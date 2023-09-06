# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS
from mmdet3d.ops import bev_pool
from mmcv.cnn import build_conv_layer
from .. import builder


def gen_dx_bx(xbound, ybound, zbound):
    '''
    'xbound': [-51.2, 51.2, 0.8],
    'ybound': [-51.2, 51.2, 0.8],
    'zbound': [-10.0, 10.0, 20.0]
    '''
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # [3]，间隔
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])    # [3]，起始点+间隔/2
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]) # [3]，间隔的数量
    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


@NECKS.register_module()
class ViewTransformerLiftSplatShoot(BaseModule):
    def __init__(self, grid_config=None, data_config=None,
                 numC_input=512, numC_Trans=64, downsample=16,
                 accelerate=False, max_drop_point_rate=0.0, use_bev_pool=True,
                 **kwargs):
        super(ViewTransformerLiftSplatShoot, self).__init__()
        if grid_config is None:
            grid_config = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [1.0, 60.0, 1.0],}
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )    # ([delta_x, delta_y, delta_z], [], [num_delta_xs, num_delta_ys, num_delta_zs])
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_config is None:
            data_config = {'input_size': (256, 704)}
        self.data_config = data_config
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape    # [D, H, W, 3]，对特征图上的每一个点，有一个对应的图像大小中的坐标
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)
        self.geom_feats = None
        self.accelerate = accelerate
        self.max_drop_point_rate = max_drop_point_rate
        self.use_bev_pool = use_bev_pool

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size'] # 图像大小
        fH, fW = ogfH // self.downsample, ogfW // self.downsample   # 特征图大小
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # [D, H, W]
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)    # 在图像大小中的x坐标
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)    # 在图像大小中的y坐标

        # D x H x W x 3，对特征图上的每一个点，有一个对应的图像大小中的坐标
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans, offset=None):
        """Determine the (x,y,z) locations (in the ego frame) of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        # 根据lss相关的变换：post_trans/pos_rots/intrinsics/rots/trans 转换为lidar坐标系下的坐标
        B, N, _ = trans.shape

        # undo post-transformation，消除图像增强以及预处理对象像素的变化
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        if offset is not None:
            _, D, H, W = offset.shape
            points[:, :, :, :, :, 2] = points[:, :, :, :, :, 2] + offset.view(B, N, D, H, W)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        # xs, ys, lamda -> xs * lamda, ys*lamda, lamda
        # 对点云中预测的宽度和高度上的栅格坐标，将其乘以深度上的栅格坐标
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        if intrins.shape[3] == 4: # for KITTI
            shift = intrins[:, :, :3, 3]
            points  = points - shift.view(B, N, 1, 1, 1, 3, 1)
            intrins = intrins[:, :, :3, :3]
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)   # 除以内参矩阵，乘以相机坐标系到车身坐标系的旋转矩阵rots
        points += trans.view(B, N, 1, 1, 1, 3)  # #加上相机坐标系到车身坐标系的平移矩阵

        # points_numpy = points.detach().cpu().numpy()
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)    # [B * N * D * H * W, C]

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long() # (geo_m_feats - start) / delta，将[-50,50] [-10 10]的范围平移到[0,100] [0,20]，计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3) # [B * N * D * H * W, 3]
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])    # (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, ..., B, B, B, B) [Nprime, 1]
        geom_feats = torch.cat((geom_feats, batch_ix), 1)   # [Nprime, 4]

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # 随机按比例drop点
        if self.max_drop_point_rate > 0.0 and self.training:
            drop_point_rate = torch.rand(1)*self.max_drop_point_rate
            kept = torch.rand(x.shape[0])>drop_point_rate
            x, geom_feats = x[kept], geom_feats[kept]

        if self.use_bev_pool:
            final = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0],
                                   self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            # shape of geom_feats [Nprime, 4], xyz in lidar coordinate and batch_idx
            # 把bev相同位置的点合在一起
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            # 给每个点赋予一个ranks，ranks相等的点在同一个batch中，也在同一个栅格中，将ranks排序并返回排序的索引以至于x,geom_feats,ranks都是按照ranks排序的。
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x    # 将x按照栅格坐标放到final中
        # collapse Z, [B, C, Z, X, Y] -> [B, C * Z, X, Y]
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def voxel_pooling_accelerated(self, rots, trans, intrins, post_rots, post_trans, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        max = 300
        # flatten indices
        if self.geom_feats is None:
            geom_feats = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # 得到在点云中的坐标
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long() # 得到在范围列表[-52.0, -51.2, ..., 51.2, 52.0]内的整数下标
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                             device=x.device, dtype=torch.long) for ix in range(B)])    # (0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...., B-1, B-1, B-1, B-1, B-1)
            geom_feats = torch.cat((geom_feats, batch_ix), 1)   # [Nprime, 4] (x, y, z, b)

            # filter out points that are outside box
            kept1 = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
                    & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
                    & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2]) # [Nprime]
            idx = torch.range(0, x.shape[0] - 1, dtype=torch.long)  # torch.range(start, end)包括start和end，(0, 1, ..., Nprime-1)
            x = x[kept1]
            idx = idx[kept1]
            geom_feats = geom_feats[kept1]

            # get tensors from the same voxel next to each other
            # 对B * N * D * H * W个位置，也就是每个batch中每个图像的每个位置，有对应的x y z B，计算x y z B，B放在最后可以让同一个batch idx中的
            ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                    + geom_feats[:, 1] * (self.nx[2] * B) \
                    + geom_feats[:, 2] * B \
                    + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks, idx = x[sorts], geom_feats[sorts], ranks[sorts], idx[sorts]   # 排序，相近的点放在一起
            repeat_id = torch.ones(geom_feats.shape[0], device=geom_feats.device, dtype=geom_feats.dtype)   # [Nkept]
            curr = 0
            repeat_id[0] = 0
            curr_rank = ranks[0]

            for i in range(1, ranks.shape[0]):
                if curr_rank == ranks[i]:
                    curr += 1
                    repeat_id[i] = curr
                else:
                    curr_rank = ranks[i]
                    curr = 0
                    repeat_id[i] = curr
            kept2 = repeat_id < max # 不能有超过max点映射到一个地方
            repeat_id, geom_feats, x, idx = repeat_id[kept2], geom_feats[kept2], x[kept2], idx[kept2]

            geom_feats = torch.cat([geom_feats, repeat_id.unsqueeze(-1)], dim=-1)   # [Nkept, 5]
            self.geom_feats = geom_feats
            self.idx = idx
        else:
            geom_feats = self.geom_feats
            idx = self.idx
            x = x[idx]

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[1], nx[0], max), device=x.device)  # [B, C, Z, X, Y, max] 
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0], geom_feats[:, 4]] = x
        final = final.sum(-1)   # [B, C, Z, X, Y]
        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)   # [B, C*Z, X, Y]

        return final

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        img_feat = x[:, self.D:(self.D + self.numC_Trans)]

        # Lift
        volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat


class SELikeModule(nn.Module):
    def __init__(self, in_channel=512, feat_channel=256, intrinsic_channel=33):
        super(SELikeModule, self).__init__()
        self.input_conv = nn.Conv2d(in_channel, feat_channel, kernel_size=1, padding=0)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(intrinsic_channel),
            nn.Linear(intrinsic_channel, feat_channel),
            nn.Sigmoid() )

    def forward(self, x, cam_params):
        x = self.input_conv(x)
        b,c,_,_ = x.shape
        y = self.fc(cam_params).view(b, c, 1, 1)
        return x * y.expand_as(x)


@NECKS.register_module()
class ViewTransformerLSSBEVDepth(ViewTransformerLiftSplatShoot):
    def __init__(self, extra_depth_net, loss_depth_weight, se_config=dict(),
                 dcn_config=dict(bias=True), **kwargs):
        super(ViewTransformerLSSBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.extra_depthnet = builder.build_backbone(extra_depth_net)   # type='ResNetForBEVDet'
        self.featnet = nn.Conv2d(self.numC_input,
                                 self.numC_Trans,
                                 kernel_size=1,
                                 padding=0)
        self.depthnet = nn.Conv2d(extra_depth_net['num_channels'][0],
                                  self.D,
                                  kernel_size=1,
                                  padding=0)
        self.dcn = nn.Sequential(*[build_conv_layer(dict(type='DCNv2',  # Deformable ConvNet V2
                                                        deform_groups=1),
                                                   extra_depth_net['num_channels'][0],
                                                   extra_depth_net['num_channels'][0],
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1,
                                                   dilation=1,
                                                   **dcn_config),
                                   nn.BatchNorm2d(extra_depth_net['num_channels'][0])
                                  ])
        self.se = SELikeModule(self.numC_input,
                               feat_channel=extra_depth_net['num_channels'][0],
                               **se_config) # 将intrinsics升维至feat_channel [b, intrinsic_channel] -> [b, feat_channel]，feature维度至feat_channel  [b, self.numC_input, h, w]，两者相加

    def forward(self, input):
        x, rots, trans, intrins, post_rots, post_trans, depth_gt = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)

        img_feat = self.featnet(x)
        depth_feat = x
        cam_params = torch.cat([intrins.reshape(B*N,-1),
                               post_rots.reshape(B*N,-1),
                               post_trans.reshape(B*N,-1),
                               rots.reshape(B*N,-1),
                               trans.reshape(B*N,-1)],dim=1)
        depth_feat = self.se(depth_feat, cam_params)    # 将intrinsics升维至feat_channel [B*N, intrinsic_channel] -> [B*N, feat_channel]，depth_feat维度至feat_channel  [B*N, self.numC_input, h, w]，两者相加
        depth_feat = self.extra_depthnet(depth_feat)[0]
        depth_feat = self.dcn(depth_feat)
        depth_digit = self.depthnet(depth_feat)
        depth_prob = self.get_depth_dist(depth_digit)   # [B*N, D, H, W]

        # Lift
        volume = depth_prob.unsqueeze(1) * img_feat.unsqueeze(2)    # [B*N, C, D, H, W]
        volume = volume.view(B, N, self.numC_Trans, self.D, H, W)   # [B, N, C, D, H, W]
        volume = volume.permute(0, 1, 3, 4, 5, 2)   # [B, N, D, H, W, C]

        # Splat
        if self.accelerate:
            bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
        else:
            geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
            bev_feat = self.voxel_pooling(geom, volume)
        return bev_feat, depth_digit