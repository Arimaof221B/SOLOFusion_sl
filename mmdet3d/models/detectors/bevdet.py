# Copyright (c) Phigent Robotics. All rights reserved.

import torch
from mmcv.runner import force_fp32
import torch.nn.functional as F

from mmdet.models import DETECTORS
from .centerpoint import CenterPoint
from .. import builder


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    def __init__(self, img_view_transformer, img_bev_encoder_backbone, img_bev_encoder_neck, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)

    def image_encoder(self,img):
        
        print(img.shape)
        
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x    # [B, N, C, H, W]

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # img[0]是[transformed_img (cam 0), transformed_adjacent_img_0 (cam 0), transformed_img (cam 1), transformed_adjacent_img_0 (cam 1), ..., ]
        x = self.image_encoder(img[0])  # [B, N, C, H, W]
        x = self.img_view_transformer([x] + img[1:])
        x = self.bev_encoder(x)
        return [x]

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0],list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        combine_type = self.test_cfg.get('combine_type','output')
        if combine_type=='output':
            return self.aug_test_combine_output(points, img_metas, img, rescale)
        elif combine_type=='feature':
            return self.aug_test_combine_feature(points, img_metas, img, rescale)
        else:
            assert False

    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


    def forward_dummy(self, points=None, img_metas=None, img_inputs=None, **kwargs):
        img_feats, _ = self.extract_feat(points, img=img_inputs, img_metas=img_metas)
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas=[dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list


@DETECTORS.register_module()
class BEVDetSequential(BEVDet):
    def __init__(self, aligned=False, distill=None, pre_process=None,
                 pre_process_neck=None, detach=True, test_adj_ids=None, **kwargs):
        super(BEVDetSequential, self).__init__(**kwargs)
        self.aligned = aligned
        self.distill = distill is not None
        if self.distill:
            self.distill_net = builder.build_neck(distill)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.pre_process_neck = pre_process_neck is not None
        if self.pre_process_neck:
            self.pre_process_neck_net = builder.build_neck(pre_process_neck)
        self.detach = detach
        self.test_adj_ids = test_adj_ids
    
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape # [B, N, 3, H, W]
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)
        imgs = torch.split(imgs,1,2)    # (B, N//2, 2, 3, H, W) -> (B, N//2, 1, 3, H, W) and (B, N//2, 1, 3, H, W), split to front and back
        imgs = [t.squeeze(2) for t in imgs] # [(B, N//2, 3, H, W), (B, N//2, 3, H, W)]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        
        # 注意为什么imgs的reshape方式和extra不一致？ 因为imgs和extra的顺序不一致，为了保证对应顺序
        
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        for img, rot, tran, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)   # ViewTransformerLiftSplatShoot中的depthnet是单层卷积，in_channels=self.numC_input, out_channels=self.D+self.numC_Trans
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)   # 将frustum的点转换为lidar坐标系下的点，[B, N, D, H, W, 3]
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])    # return x.softmax(dim=1)
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2) # [B * N, 1, D, H, W] * [B * N, C, 1, H, W]
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W) # [B, N, C, D, H, W]
            volume = volume.permute(0, 1, 3, 4, 5, 2)   # [B, N, D, H, W, D]

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)    # ()

            if self.pre_process:
                bev_feat = self.pre_process_net(bev_feat)
                if self.pre_process_neck:
                    bev_feat = self.pre_process_neck_net(bev_feat)
                else:
                    bev_feat = bev_feat[0]
            bev_feat_list.append(bev_feat)
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


@DETECTORS.register_module()
class BEVDetSequentialES(BEVDetSequential):
    def __init__(self, before=False, interpolation_mode='bilinear',**kwargs):
        super(BEVDetSequentialES, self).__init__(**kwargs)
        self.before=before
        self.interpolation_mode=interpolation_mode

    @force_fp32()
    def shift_feature(self, input, trans, rots):
        n, c, h, w = input.shape
        _,v,_ =trans[0].shape

        # generate grid
        xs = torch.linspace(0, w - 1, w, dtype=input.dtype, device=input.device).view(1, w).expand(h, w)    # ((0, 1, ..., w-1), (0, 1, ..., w-1), ..., (0, 1, ..., w-1))
        ys = torch.linspace(0, h - 1, h, dtype=input.dtype, device=input.device).view(h, 1).expand(h, w)    # ((0, 0, ..., 0), (1, 1, ..., 1), (h-1, ..., h-1))
        grid = torch.stack((xs, ys, torch.ones_like(xs)), -1).view(1, h, w, 3).expand(n, h, w, 3).view(n,h,w,3,1)   # [N, H, W, 3, 1]，特征图上各个点的坐标[x, y, 1]
        grid = grid

        # get transformation from current lidar frame to adjacent lidar frame，注意，transformation的顺序是左乘
        # 0表示current，1表示adjacent
        # transformation from current camera frame to current lidar frame (from adjacent camera frame to adjacent lidar frame, e.t. c02l0==c12l1)
        c02l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)    # 4*4变换矩阵
        c02l0[:,:,:3,:3] = rots[0]
        c02l0[:,:,:3,3] = trans[0]
        c02l0[:,:,3,3] = 1  

        # transformation from adjacent camera frame to current lidar frame
        c12l0 = torch.zeros((n,v,4,4),dtype=grid.dtype).to(grid)    # 4*4变换矩阵
        c12l0[:,:,:3,:3] = rots[1]
        c12l0[:,:,:3,3] = trans[1]
        c12l0[:,:,3,3] =1

        # transformation from current lidar frame to adjacent lidar frame (current lidar -> adjacent camera -> adjacent lidar)
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:,0,:,:].view(n,1,1,4,4)
        '''
          c02l0 * inv（c12l0）
        = c02l0 * inv(l12l0 * c12l1) 注意这里的顺序，先c12l1再l12l0合起来就是c12l0
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        # 因为只做BEV平面的align,因此把第三维去掉
        l02l1 = l02l1[:,:,:,[True,True,False,True],:][:,:,:,:,[True,True,False,True]]

        # feat2bev 是特征空间和BEV空间（lidar坐标系）之间的变换，特征空间和lidar坐标系下的bev空间是不同的
        '''
        [[dx, 0,  start_x], 
         [0,  dy, start_y], 
         [0,  0,  1]]
        '''
        feat2bev = torch.zeros((3,3),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 2] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2. # x of start point
        feat2bev[1, 2] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2. # y of start point
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1,3,3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev) # feat(current lidar) -> bev(current lidar) -> bev (adjacent lidar) -> feat (adjacent lidar) 

        # transform and normalize, normalize是因为grid_sample要求要把绝对的坐标normalize到[-1, 1]的区间内
        grid = tf.matmul(grid)  # [N, H, W, 3, 1]，特征图上各个点的坐标[x, y, 1]
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=input.dtype, device=input.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True, mode=self.interpolation_mode)
        return output


    def extract_img_feat(self, img, img_metas):
        inputs = img    # img is (imgs, rots, trans, intrins, post_rots, post_trans)
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape # [B, N, 3, H, W]
        N = N//2
        imgs = inputs[0].view(B,N,2,3,H,W)  # 分为当前帧和相邻帧
        imgs = torch.split(imgs,1,2)    # (B, N//2, 2, 3, H, W) -> (B, N//2, 1, 3, H, W) and (B, N//2, 1, 3, H, W), split to front and back
        imgs = [t.squeeze(2) for t in imgs] # [(B, N//2, 3, H, W), (B, N//2, 3, H, W)]
        rots, trans, intrins, post_rots, post_trans = inputs[1:]
        extra = [rots.view(B,2,N,3,3),
                 trans.view(B,2,N,3),
                 intrins.view(B,2,N,3,3),
                 post_rots.view(B,2,N,3,3),
                 post_trans.view(B,2,N,3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        
        # print(inputs[0].shape, len(imgs), imgs[0].shape, len(img_metas))
        
        for img, _ , _, intrin, post_rot, post_tran in zip(imgs, rots, trans, intrins, post_rots, post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            x = self.img_view_transformer.depthnet(x)   # ViewTransformerLiftSplatShoot中的depthnet是单层卷积，in_channels=self.numC_input, out_channels=self.D+self.numC_Trans
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin, post_rot, post_tran)   # 将frustum的点转换为lidar坐标系下的点，[B, N, D, H, W, 3]
            depth = self.img_view_transformer.get_depth_dist(x[:, :self.img_view_transformer.D])    # return x.softmax(dim=1)
            img_feat = x[:, self.img_view_transformer.D:(
                    self.img_view_transformer.D + self.img_view_transformer.numC_Trans)]    # [B * N, numC_Trans, H, W]

            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2) # [B * N, 1, D, H, W] * [B * N, numC_Trans, 1, H, W] -> [B * N, numC_Trans, D, H, W]
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans, self.img_view_transformer.D, H, W) # [B, N, numC_Trans, D, H, W]
            volume = volume.permute(0, 1, 3, 4, 5, 2)   # [B, N, D, H, W, numC_Trans]

            # Splat
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans, rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
        return [x]


class BEVDepth_Base():
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats, depth = self.extract_img_feat(img, img_metas)
        pts_feats = None
        return (img_feats, pts_feats, depth)


    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        B, N, H, W = depth_gt.shape
        loss_weight = (~(depth_gt == 0)).reshape(B, N, 1, H, W).expand(B, N,
                                                                       self.img_view_transformer.D,
                                                                       H, W)
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                   /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                              self.img_view_transformer.D).to(torch.long)
        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32)
        depth = depth.sigmoid().view(B, N, self.img_view_transformer.D, H, W)

        loss_depth = F.binary_cross_entropy(depth, depth_gt_logit,
                                            weight=loss_weight)
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth


@DETECTORS.register_module()
class BEVDepth(BEVDepth_Base, BEVDet):
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:])
        x = self.bev_encoder(x)
        return [x], depth

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepth4D(BEVDepth_Base, BEVDetSequentialES):
    def extract_img_feat(self, img, img_metas):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        rots, trans, intrins, post_rots, post_trans, depth_gt = inputs[1:]
        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        rots, trans, intrins, post_rots, post_trans = extra
        bev_feat_list = []
        depth_digit_list = []
        for img, _, _, intrin, post_rot, post_tran in zip(imgs, rots, trans,
                                                          intrins, post_rots,
                                                          post_trans):
            tran = trans[0]
            rot = rots[0]
            x = self.image_encoder(img)
            B, N, C, H, W = x.shape
            x = x.view(B * N, C, H, W)
            # BEVDepth
            img_feat = self.img_view_transformer.featnet(x)
            depth_feat = x
            cam_params = torch.cat([intrin.reshape(B * N, -1),
                                   post_rot.reshape(B * N, -1),
                                   post_tran.reshape(B * N, -1),
                                   rot.reshape(B * N, -1),
                                   tran.reshape(B * N, -1)], dim=1)
            depth_feat = self.img_view_transformer.se(depth_feat,
                                                      cam_params)
            depth_feat = self.img_view_transformer.extra_depthnet(depth_feat)[0]
            depth_feat = self.img_view_transformer.dcn(depth_feat)
            depth_digit = self.img_view_transformer.depthnet(depth_feat)
            depth = self.img_view_transformer.get_depth_dist(depth_digit)
            # Lift
            volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            volume = volume.view(B, N, self.img_view_transformer.numC_Trans,
                                 self.img_view_transformer.D, H, W)
            volume = volume.permute(0, 1, 3, 4, 5, 2)

            # Splat
            geom = self.img_view_transformer.get_geometry(rot, tran, intrin,
                                                          post_rot, post_tran)
            bev_feat = self.img_view_transformer.voxel_pooling(geom, volume)

            bev_feat_list.append(bev_feat)
            depth_digit_list.append(depth_digit)

        if self.before and self.pre_process:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        bev_feat_list[1] = self.shift_feature(bev_feat_list[1], trans,
                                              rots)
        if self.pre_process and not self.before:
            bev_feat_list = [self.pre_process_net(bev_feat)[0] for bev_feat
                             in bev_feat_list]
        if self.detach:
            bev_feat_list[1] = bev_feat_list[1].detach()
        if self.distill:
            bev_feat_list[1] = self.distill_net(bev_feat_list)
        bev_feat = torch.cat(bev_feat_list, dim=1)

        x = self.bev_encoder(bev_feat)
                
        return [x], depth_digit_list[0]

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        assert self.with_pts_bbox

        depth_gt = img_inputs[-1]
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses
