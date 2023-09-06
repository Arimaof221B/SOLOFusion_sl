import os
import copy
import math
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmcv.ops.nms import batched_nms
from mmdet.models import DETECTORS
from mmdet.core.post_processing.bbox_nms import multiclass_nms
from mmdet3d.ops import bev_pool
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

from .bevdet import BEVDet, BEVDetSequentialES
from .. import builder


def generate_forward_transformation_matrix(img_meta_dict):
    res = torch.eye(3)

    if 'transformation_3d_flow' in img_meta_dict:
        for transform_type in img_meta_dict['transformation_3d_flow']:
            if transform_type == "R":
                if "pcd_rotation" in img_meta_dict:
                    res = img_meta_dict['pcd_rotation'].T @ res # .T since L158 of lidar_box3d has points @ rot
            elif transform_type == "S":
                if "pcd_scale_factor" in img_meta_dict:
                    res = res * img_meta_dict['pcd_scale_factor']
            elif transform_type == "T":
                if "pcd_trans" in img_meta_dict:
                    assert torch.tensor(img_meta_dict['pcd_trans']).abs().sum() == 0, \
                        "I'm not supporting translation rn; need to convert to hom coords which is annoying"
            elif transform_type == "HF": # Horizontal is Y apparently
                if "pcd_horizontal_flip" in img_meta_dict:
                    tmp = torch.eye(3)
                    tmp[1, 1] = -1
                    res = tmp @ res
            elif transform_type == "VF":
                if "pcd_vertical_flip" in img_meta_dict:
                    tmp = torch.eye(3)
                    tmp[0, 0] = -1
                    res = tmp @ res
            else:
                raise Exception(str(img_meta_dict))

    hom_res = torch.eye(4)
    hom_res[:3, :3] = res
    return hom_res

@DETECTORS.register_module()
class SOLOFusion(BEVDet):
    def __init__(self, 
                 pre_process=None, 
                 pre_process_neck=None, 
                 
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=1, # Number of history key frames to cat
                 history_cat_conv_out_channels=None,
                 
                 use_gt_velocity=False, 

                 ### Stereo?
                 do_history_stereo_fusion=False,
                 stereo_neck=None,
                 history_stereo_prev_step=1,
                 **kwargs):
        super(SOLOFusion, self).__init__(**kwargs)

        #### Prior to history fusion, do some per-sample pre-processing.
        self.single_bev_num_channels = self.img_view_transformer.numC_Trans

        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        #### Deal with history
        self.do_history = do_history
        if self.do_history:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            
        self.use_gt_velocity = use_gt_velocity

        #### Stereo depth fusion
        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.stereo_neck = stereo_neck
            if self.stereo_neck is not None:
                self.stereo_neck = builder.build_neck(self.stereo_neck)
            self.history_stereo_prev_step = history_stereo_prev_step

        self.prev_stereo_img_feats = None # B x N x C x H x W
        self.prev_stereo_global_to_img = None # B x N x 4 x 4
        self.prev_stereo_img_forward_augs = None

        self.fp16_enabled = False

    @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        if self.do_history_stereo_fusion:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if isinstance(stereo_feats, list):
                assert len(stereo_feats) == 1 # SECONDFPN returns a trivial list
                stereo_feats = stereo_feats[0]
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=self.img_view_transformer.stereo_eps)
            return neck_feats, stereo_feats.view(B, N, *stereo_feats.shape[1:])
        else:
            return neck_feats, None

    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        """
        This was updated to be more similar to BEVDepth's original depth loss function.
        """
        B, N, H, W = depth_gt.shape
        fg_mask = (depth_gt != 0).view(-1) 
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                            self.img_view_transformer.D).to(torch.long)
        assert depth_gt.max() < self.img_view_transformer.D

        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32) # B x N x D x H x W
        depth = depth.view(B, N, self.img_view_transformer.D, H, W).softmax(dim=2)
        
        depth_gt_logit = depth_gt_logit.permute(0, 1, 3, 4, 2).view(-1, self.img_view_transformer.D)    # [B*N*H*W, D]
        depth = depth.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.img_view_transformer.D) # [B*N*H*W, D]

        loss_depth = (F.binary_cross_entropy(
                depth[fg_mask],
                depth_gt_logit[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    @force_fp32(apply_to=('rots', 'trans', 'intrins', 'post_rots', 'post_trans'))
    def process_stereo_before_fusion(self, stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans):
        # NOTE: This is written to technically support history_stereo_prev_step > 1, 
        # but I haven't actually run any experiments on that, so please double check.

        # This is meant to deal with start of sequences, initializing variables, etc. It also returns
        # The current global to img transofmrations.
        # This *does not* update the history yet.

        B, N, C, H, W = stereo_feats.shape
        # list of start_of_sequence
        start_of_sequence = torch.BoolTensor([  # start_of_sequence是布尔值，表示当前img是否是sequence的第一帧
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(stereo_feats.device)
        global_to_curr_lidar_rt = torch.stack([
            single_img_metas['global_to_curr_lidar_rt']
            for single_img_metas in img_metas]).to(rots) # B x 4 x 4
        lidar_forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas)    # 根据single_img_metas得到transformation matrix
            for single_img_metas in img_metas], dim=0).to(rots.device)

        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x N x 3 or B x N x 3 x 3
        cam_to_cam_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = post_rots
        cam_to_cam_aug[:, :, :3, 3] = post_trans

        intrins4x4 = rots.new_zeros((B, N, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = intrins

        cam_to_lidar_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = rots
        cam_to_lidar_aug[:, :, :3, 3] = trans

        ## Okay, to go from global to augmed cam (XYD)....
        # We can go from global to (X*D, X*D, D), then we need user to divide by D,
        # Then we can apply the img augs. So, we need to store both.
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug
        global_to_img = (intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
                         @ lidar_forward_augs[:, None, :, :] @ global_to_curr_lidar_rt[:, None, :, :])
        img_forward_augs = cam_to_cam_aug

        ### Then, let's check if stereo saved values are none or we're the first in the sequence.
        if self.prev_stereo_img_feats is None:
            # For the history_stereo_prev_step dimension, the "first" one will be the most recent one 
            self.prev_stereo_img_feats = stereo_feats.detach().clone()[:, None, :, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1, 1) # B x history_stereo_prev_step x N x C x H x W
            self.prev_stereo_global_to_img = global_to_img.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            self.prev_stereo_img_forward_augs = img_forward_augs.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            # self.prev_stereo_frame_idx = stereo_feats.new_zeros((B))[:, None].repeat(
            #     1, self.history_stereo_prev_step) # B x history_stereo_prev_step
        else:
            self.prev_stereo_img_feats[start_of_sequence] = stereo_feats[start_of_sequence].unsqueeze(1).detach().clone()
            self.prev_stereo_global_to_img[start_of_sequence] = global_to_img[start_of_sequence].unsqueeze(1).clone()
            self.prev_stereo_img_forward_augs[start_of_sequence] = img_forward_augs[start_of_sequence].unsqueeze(1).clone()

        # These are both B x N x 4 x 4. Want the result to be B x prev_N x curr_N x 4 x 4
        curr_unaug_cam_to_prev_unaug_cam = self.prev_stereo_global_to_img[:, 0][:, :, None, :, :] @ torch.inverse(global_to_img)[:, None, :, :, :]

        return (self.prev_stereo_img_feats[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_global_to_img[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_img_forward_augs[:, self.history_stereo_prev_step - 1],
                global_to_img,
                img_forward_augs,
                curr_unaug_cam_to_prev_unaug_cam)

    def process_stereo_for_next_timestep(self, img_metas, stereo_feats, global_to_img, img_forward_augs):
        self.prev_stereo_img_feats[:, 1:] = self.prev_stereo_img_feats[:, :-1].clone() # Move
        self.prev_stereo_img_feats[:, 0] = stereo_feats.detach().clone()
        self.prev_stereo_global_to_img[:, 1:] = self.prev_stereo_global_to_img[:, :-1].clone() # Move
        self.prev_stereo_global_to_img[:, 0] = global_to_img.clone()
        self.prev_stereo_img_forward_augs[:, 1:] = self.prev_stereo_img_forward_augs[:, :-1].clone() # Move
        self.prev_stereo_img_forward_augs[:, 0] = img_forward_augs.detach().clone()

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas):
        # print(img_metas[0].keys())
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas) 
            for single_img_metas in img_metas], dim=0).to(curr_bev)
        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
            
            self.timestamp = torch.LongTensor([
                single_img_metas['timestamp'] 
                for single_img_metas in img_metas]).to(curr_bev.device)
        
        self.timestamp[start_of_sequence] = torch.LongTensor([single_img_metas['timestamp'] for single_img_metas in img_metas]).to(curr_bev.device)[start_of_sequence]

        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
        self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        n, c, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        '''
        [[dx, 0,  start_x, 0], 
         [0,  dy, start_y, 0], 
         [0,  0,  1,       0], 
         [0,  0,  0,       1]]
        '''
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations .
        
        if not self.use_gt_velocity: 
            rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_lidar_rt
                    @ torch.inverse(forward_augs) @ feat2bev)
            
            grid = rt_flow.view(n, 1, 1, 4, 4) @ grid
        else:
            temp_grid = torch.inverse(forward_augs).view(n, 1, 1, 4, 4) @ feat2bev @ grid
            temp_grid_clone = temp_grid.clone()
            
            for nim in range(len(img_metas)):
                boxes = img_metas[nim]['gt_boxes']
                velocities = img_metas[nim]['gt_velocity']
                for b, v in zip(boxes, velocities):
                    idx = (temp_grid_clone[nim][:, :, 0] >= (b[0] - b[3])) & (temp_grid_clone[nim][:, :, 0] <= (b[0] + b[3])) & (temp_grid_clone[nim][:, :, 1] >= (b[1] - b[4])) & (temp_grid_clone[nim][:, :, 1] <= (b[1] + b[4]))
                    temp_grid[nim][idx.squeeze()][:, :2, 0] -= torch.FloatTensor(v).to(curr_bev.device) * (img_metas[nim]['timestamp'] - self.timestamp[nim])
                self.timestamp[nim] = img_metas[nim]['timestamp']
            grid = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_lidar_rt).view(n, 1, 1, 4, 4) @ temp_grid

        
        # print(len(img_metas), self.history_forward_augs.shape, curr_to_prev_lidar_rt.shape, rt_flow.shape, self.single_bev_num_channels, curr_bev.shape, self.history_bev.shape)
        #           8                    [8, 4, 4]                       [8, 4, 4]               [8, 4, 4]              80              [8, 80, 128, 128] [8, 1280, 128, 128]
        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        sampled_history_bev = F.grid_sample(self.history_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W

        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
        feats_to_return = torch.cat(
            [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (1 + T) x 81 x H x W
        
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
        
        # Update history by moving everything down one group of single_bev_num_channels channels
        # and adding in curr_bev.
        # Clone is necessary since we're doing in-place operations on self.history_bev
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, :, :].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]

        self.history_forward_augs = forward_augs.clone()
        return feats_to_return.clone()


    def extract_img_feat(self, img, img_metas, gt_bboxes_3d=None):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        imgs = inputs[0].view(B, N, 1, 3, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
  
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]

        extra = [rots.view(B, 1, N, 3, 3),
                 trans.view(B, 1, N, 3),
                 intrins.view(B, 1, N, 3, 3),
                 post_rots.view(B, 1, N, 3, 3),
                 post_trans.view(B, 1, N, 3)]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra

        bev_feat_list = []
        depth_digit_list = []
        geom_list = []
            
        curr_img_encoder_feats, curr_stereo_feats = self.image_encoder(imgs[0])
        if not self.do_history_stereo_fusion:
            bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
        else:
            prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam = \
                self.process_stereo_before_fusion(curr_stereo_feats, img_metas, rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
            bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0],
                curr_stereo_feats, prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam)
            # prev_stereo_img_feats、prev_stereo_global_to_img、prev_stereo_img_forward_augs全部向后移一位，当前的放在第0位
            self.process_stereo_for_next_timestep(img_metas, curr_stereo_feats, curr_global2img, curr_img_forward_aug)

        bev_feat = self.pre_process_net(bev_feat)[0] # singleton list
        bev_feat = self.embed(bev_feat)

        # Fuse History
        if self.do_history:
            bev_feat = self.fuse_history(bev_feat, img_metas)

        x = self.bev_encoder(bev_feat)
        # print(imgs[0].shape, bev_feat.shape, depth_digit.shape)
        # [8, 6, 3, 256, 704] [8, 160, 128, 128] [48, 112, 16, 44]
        return x, depth_digit

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

        # print(img_metas[0]['start_of_sequence'], img_metas[0]['timestamp'])
        
        
        losses = dict()
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas)

        # If we're training depth...
        depth_gt = img_inputs[-1]
        
         
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses['loss_depth'] = loss_depth
        
        # Get box losses
        bbox_outs = self.pts_bbox_head(img_feats)
        losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
        losses.update(losses_pts)

        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, _ = self.extract_img_feat(img, img_metas)
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        bbox_list = [dict(pts_bbox=bbox_pts[0])]

        return bbox_list




@DETECTORS.register_module()
class SOLOFusionSequential(BEVDet):
    def __init__(self, 
                 pre_process=None, 
                 pre_process_neck=None, 
                 
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=1, # Number of history key frames to cat
                 history_cat_conv_out_channels=None,
                 
                 do_future=True,
                 future_cat_num=1, # Number of history key frames to cat
                 future_cat_conv_out_channels=None,
                 
                 ### Stereo?
                 do_history_stereo_fusion=False,
                 stereo_neck=None,
                 history_stereo_prev_step=1,
                 **kwargs):
        super(SOLOFusionSequential, self).__init__(**kwargs)

        #### Prior to history fusion, do some per-sample pre-processing.
        self.single_bev_num_channels = self.img_view_transformer.numC_Trans

        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        #### Deal with history
        self.do_history = do_history
        self.do_future = do_future
        
        if self.do_history and not self.do_future:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
        
        elif self.do_future:
            self.interpolation_mode = interpolation_mode
            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            
            self.future_cat_num = future_cat_num
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.future_cat_num + self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None
            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            
            self.future_sweep_time = None
            self.future_bev = None
            self.future_seq_ids = None
            self.future_forward_augs = None
        

        #### Stereo depth fusion
        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.stereo_neck = stereo_neck
            if self.stereo_neck is not None:
                self.stereo_neck = builder.build_neck(self.stereo_neck)
            self.history_stereo_prev_step = history_stereo_prev_step

        self.prev_stereo_img_feats = None # B x N x C x H x W
        self.prev_stereo_global_to_img = None # B x N x 4 x 4
        self.prev_stereo_img_forward_augs = None

        self.fp16_enabled = False

    @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        if self.do_history_stereo_fusion:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if isinstance(stereo_feats, list):
                assert len(stereo_feats) == 1 # SECONDFPN returns a trivial list
                stereo_feats = stereo_feats[0]
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=self.img_view_transformer.stereo_eps)
            return neck_feats, stereo_feats.view(B, N, *stereo_feats.shape[1:])
        else:
            return neck_feats, None

    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        """
        This was updated to be more similar to BEVDepth's original depth loss function.
        """
        B, N, H, W = depth_gt.shape
        fg_mask = (depth_gt != 0).view(-1) 
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                            self.img_view_transformer.D).to(torch.long)
        assert depth_gt.max() < self.img_view_transformer.D

        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32) # B x N x D x H x W
        depth = depth.view(B, N, self.img_view_transformer.D, H, W).softmax(dim=2)
        
        depth_gt_logit = depth_gt_logit.permute(0, 1, 3, 4, 2).view(-1, self.img_view_transformer.D)    # [B*N*H*W, D]
        depth = depth.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.img_view_transformer.D) # [B*N*H*W, D]

        loss_depth = (F.binary_cross_entropy(
                depth[fg_mask],
                depth_gt_logit[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    @force_fp32(apply_to=('rots', 'trans', 'intrins', 'post_rots', 'post_trans'))
    def process_stereo_before_fusion(self, stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans):
        # NOTE: This is written to technically support history_stereo_prev_step > 1, 
        # but I haven't actually run any experiments on that, so please double check.

        # This is meant to deal with start of sequences, initializing variables, etc. It also returns
        # The current global to img transofmrations.
        # This *does not* update the history yet.

        B, N, C, H, W = stereo_feats.shape
        # list of start_of_sequence
        start_of_sequence = torch.BoolTensor([  # start_of_sequence是布尔值，表示当前img是否是sequence的第一帧
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(stereo_feats.device)
        global_to_curr_lidar_rt = torch.stack([
            single_img_metas['global_to_curr_lidar_rt']
            for single_img_metas in img_metas]).to(rots) # B x 4 x 4
        lidar_forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas)    # 根据single_img_metas得到transformation matrix
            for single_img_metas in img_metas], dim=0).to(rots.device)

        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x N x 3 or B x N x 3 x 3
        cam_to_cam_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = post_rots
        cam_to_cam_aug[:, :, :3, 3] = post_trans

        intrins4x4 = rots.new_zeros((B, N, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = intrins

        cam_to_lidar_aug = rots.new_zeros((B, N, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = rots
        cam_to_lidar_aug[:, :, :3, 3] = trans

        ## Okay, to go from global to augmed cam (XYD)....
        # We can go from global to (X*D, X*D, D), then we need user to divide by D,
        # Then we can apply the img augs. So, we need to store both.
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug
        global_to_img = (intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
                         @ lidar_forward_augs[:, None, :, :] @ global_to_curr_lidar_rt[:, None, :, :])
        img_forward_augs = cam_to_cam_aug

        ### Then, let's check if stereo saved values are none or we're the first in the sequence.
        if self.prev_stereo_img_feats is None:
            # For the history_stereo_prev_step dimension, the "first" one will be the most recent one
            self.prev_stereo_img_feats = stereo_feats.detach().clone()[:, None, :, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1, 1) # B x history_stereo_prev_step x N x C x H x W
            self.prev_stereo_global_to_img = global_to_img.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            self.prev_stereo_img_forward_augs = img_forward_augs.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            # self.prev_stereo_frame_idx = stereo_feats.new_zeros((B))[:, None].repeat(
            #     1, self.history_stereo_prev_step) # B x history_stereo_prev_step
        else:
            self.prev_stereo_img_feats[start_of_sequence] = stereo_feats[start_of_sequence].unsqueeze(1).detach().clone()
            self.prev_stereo_global_to_img[start_of_sequence] = global_to_img[start_of_sequence].unsqueeze(1).clone()
            self.prev_stereo_img_forward_augs[start_of_sequence] = img_forward_augs[start_of_sequence].unsqueeze(1).clone()

        # These are both B x N x 4 x 4. Want the result to be B x prev_N x curr_N x 4 x 4
        curr_unaug_cam_to_prev_unaug_cam = self.prev_stereo_global_to_img[:, 0][:, :, None, :, :] @ torch.inverse(global_to_img)[:, None, :, :, :]

        return (self.prev_stereo_img_feats[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_global_to_img[:, self.history_stereo_prev_step - 1], 
                self.prev_stereo_img_forward_augs[:, self.history_stereo_prev_step - 1],
                global_to_img,
                img_forward_augs,
                curr_unaug_cam_to_prev_unaug_cam)

    def process_stereo_for_next_timestep(self, img_metas, stereo_feats, global_to_img, img_forward_augs):
        self.prev_stereo_img_feats[:, 1:] = self.prev_stereo_img_feats[:, :-1].clone() # Move
        self.prev_stereo_img_feats[:, 0] = stereo_feats.detach().clone()
        self.prev_stereo_global_to_img[:, 1:] = self.prev_stereo_global_to_img[:, :-1].clone() # Move
        self.prev_stereo_global_to_img[:, 0] = global_to_img.clone()
        self.prev_stereo_img_forward_augs[:, 1:] = self.prev_stereo_img_forward_augs[:, :-1].clone() # Move
        self.prev_stereo_img_forward_augs[:, 0] = img_forward_augs.detach().clone()

    @force_fp32()
    def fuse_history(self, curr_bev, future_bev, img_metas):
        
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxx", type(future_bev))
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas) 
            for single_img_metas in img_metas], dim=0).to(curr_bev)
        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
            
            self.timestamp = torch.LongTensor([
                single_img_metas['timestamp'] 
                for single_img_metas in img_metas]).to(curr_bev.device)
        
        self.timestamp[start_of_sequence] = torch.LongTensor([single_img_metas['timestamp'] for single_img_metas in img_metas]).to(curr_bev.device)[start_of_sequence]

        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
        self.history_sweep_time -= 1 # new timestep, everything in history gets pushed back one.
        self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
        self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        n, c, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        '''
        [[dx, 0,  start_x, 0], 
         [0,  dy, start_y, 0], 
         [0,  0,  1,       0], 
         [0,  0,  0,       1]]
        '''
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations .
        rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_lidar_rt
                   @ torch.inverse(forward_augs) @ feat2bev)
        
        grid = rt_flow.view(n, 1, 1, 4, 4) @ grid
       
        
        # print(len(img_metas), self.history_forward_augs.shape, curr_to_prev_lidar_rt.shape, rt_flow.shape, self.single_bev_num_channels, curr_bev.shape, self.history_bev.shape)
        #           8                    [8, 4, 4]                       [8, 4, 4]               [8, 4, 4]              80              [8, 80, 128, 128] [8, 1280, 128, 128]
        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid = grid[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        sampled_history_bev = F.grid_sample(self.history_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        
        if not self.do_future:
            feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W

            # Reshape and concatenate features and timestep
            feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
            feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x H x W
            
            # Time conv
            feats_to_return = self.history_keyframe_time_conv(
                feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                    feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W

            # Cat keyframes & conv
            feats_to_return = self.history_keyframe_cat_conv(
                feats_to_return.reshape(
                    feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
            
            # Update history by moving everything down one group of single_bev_num_channels channels
            # and adding in curr_bev.
            # Clone is necessary since we're doing in-place operations on self.history_bev
            self.history_bev = feats_cat[:, :-self.single_bev_num_channels, :, :].detach().clone()
            self.history_sweep_time = self.history_sweep_time[:, :-1]

            self.history_forward_augs = forward_augs.clone()
            return feats_to_return.clone()
                
        
        self.history_forward_augs = forward_augs.clone()
    
    
        # ================================================================================================================================================================
    
    
        curr_to_future_lidar_rt = torch.stack([
            single_img_metas['curr_to_future_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        
        
                
        future_id = torch.stack([
            single_img_metas['future_id']
            for single_img_metas in img_metas]).to(curr_bev)
        
        
        ## Deal with first batch
        if self.future_bev is None:
            self.future_bev = curr_bev.clone()
            self.future_seq_ids = seq_ids.clone()
            self.future_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be future
            self.future_bev = curr_bev.repeat(1, self.future_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.future_sweep_time = future_id.repeat(1, self.future_cat_num)

        self.future_bev = self.future_bev.detach()

        assert self.future_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, future id and seq id should be same.
        assert (self.future_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.future_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in future with the curr_bev information
        self.future_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.future_cat_num, 1, 1)
        self.future_sweep_time -= 1 # new timestep, everything in history gets pushed back one.
        self.future_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
        self.future_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.future_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid_nn2curr = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations .
        rt_flow_nn2curr = (torch.inverse(feat2bev) @ self.future_forward_augs @ curr_to_future_lidar_rt
                   @ torch.inverse(forward_augs) @ feat2bev)
        
        grid_nn2curr = rt_flow_nn2curr.view(n, 1, 1, 4, 4) @ grid_nn2curr
       
        # normalize and sample
        grid_nn2curr = grid_nn2curr[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        

        sampled_future_bev = F.grid_sample(self.future_bev, grid.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)
        sampled_future_bev = F.grid_sample(future_bev, grid_nn2curr.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)
        sampled_future_bev = torch.cat((sampled_future_bev, sampled_future_bev), dim=1)[:, :-self.single_bev_num_channels]   # B x T * 80 x H x W

        ## Update history
        # Add in current frame to features & timestep
        self.future_sweep_time = torch.cat(
            [self.future_sweep_time.new_ones(self.future_sweep_time.shape[0], 1) * 16, self.future_sweep_time],
            dim=1)[:, :-1] # B x T -> B x T
        feats_cat = torch.cat([sampled_future_bev, curr_bev, sampled_history_bev], dim=1) # B x (T + 1 + T) * 80 x H x W

        print('======================================', feats_cat.shape)
        
        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.future_cat_num + self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
        feats_to_return = torch.cat(
            [feats_to_return, torch.cat((self.future_sweep_time, self.history_sweep_time), dim=1)[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (T + 1 + T) x 81 x H x W
        
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (T + 1 + T) x 80 x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
        
        # Update history by moving everything down one group of single_bev_num_channels channels
        # and adding in curr_bev.
        # Clone is necessary since we're doing in-place operations on self.history_bev
        self.history_bev = feats_cat[:, self.single_bev_num_channels*self.future_cat_num:-self.single_bev_num_channels, :, :].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]

        self.future_forward_augs = forward_augs.clone()
        return feats_to_return.clone()
    
    

    def extract_img_feat(self, img, img_metas, gt_bboxes_3d=None):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        N = N // 2
        imgs = inputs[0].view(B, N, 2, 3, H, W)
        imgs = torch.split(imgs, 1, dim=2)
        imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
  
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]

        extra = [rots.view(B, 2, N, 3, 3),
                 trans.view(B, 2, N, 3),
                 intrins.view(B, 2, N, 3, 3),
                 post_rots.view(B, 2, N, 3, 3),
                 post_trans.view(B, 2, N, 3)]
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra

        bev_feat_list = []
        depth_digit_list = []
        geom_list = []

                
        curr_img_encoder_feats, curr_stereo_feats = self.image_encoder(imgs[0])
        if not self.do_history_stereo_fusion:
            curr_bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
        else:            
            prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam = \
                self.process_stereo_before_fusion(curr_stereo_feats, img_metas, rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
            curr_bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0],
                curr_stereo_feats, prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam)
            # prev_stereo_img_feats、prev_stereo_global_to_img、prev_stereo_img_forward_augs全部向后移一位，当前的放在第0位
            self.process_stereo_for_next_timestep(img_metas, curr_stereo_feats, curr_global2img, curr_img_forward_aug)
        

        curr_bev_feat = self.pre_process_net(curr_bev_feat)[0] # singleton list
        curr_bev_feat = self.embed(curr_bev_feat)

        if self.do_future:

            B, N, C, imH, imW = imgs[1].shape
            future_imgs = imgs[1].view(B * N, C, imH, imW)
            future_backbone_feats = self.img_backbone(future_imgs)
            future_img_encoder_feats = self.img_neck(future_backbone_feats)
            if isinstance(future_img_encoder_feats, list):
                assert len(future_img_encoder_feats) == 1 # SECONDFPN returns a length-one list
                future_img_encoder_feats = future_img_encoder_feats[0]
            
            _, output_dim, ouput_H, output_W = future_img_encoder_feats.shape
            future_img_encoder_feats = future_img_encoder_feats.view(B, N, output_dim, ouput_H, output_W)
            
            future_bev_feat, future_depth_digit = self.img_view_transformer(
                future_img_encoder_feats, 
                rots[1], trans[1], intrins[1], post_rots[1], post_trans[1], do_stereo_fusion=False)


        # Fuse History
        if self.do_history and self.do_future:
            bev_feat = self.fuse_history(curr_bev_feat, future_bev_feat, img_metas)
        elif self.do_history:
            bev_feat = self.fuse_history(curr_bev_feat, None, img_metas)

        x = self.bev_encoder(bev_feat)
        # print(imgs[0].shape, bev_feat.shape, depth_digit.shape)
        # [8, 6, 3, 256, 704] [8, 160, 128, 128] [48, 112, 16, 44]
        return x, depth_digit

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

        # print(img_metas[0]['start_of_sequence'], img_metas[0]['timestamp'])
        losses = dict()
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas)

        # If we're training depth...
        depth_gt = img_inputs[-1] 
        B,N,H,W = depth_gt.shape
        depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses['loss_depth'] = loss_depth
        
        # Get box losses
        bbox_outs = self.pts_bbox_head(img_feats)
        losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
        losses.update(losses_pts)

        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, _ = self.extract_img_feat(img, img_metas)
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        bbox_list = [dict(pts_bbox=bbox_pts[0])]

        return bbox_list






@DETECTORS.register_module()
class SOLOFusionSequential2(BEVDet):
    def __init__(self, 
                 pre_process=None, 
                 pre_process_neck=None, 
                 
                 do_history=True,
                 interpolation_mode='bilinear',
                 history_cat_num=1, # Number of history key frames to cat
                 history_cat_conv_out_channels=None,
                 
                 do_future=False,
                 future_cat_num=1, # Number of history key frames to cat
                 
                 use_gt_velocity=False, 
                 
                 ### Stereo?
                 do_history_stereo_fusion=False,
                 stereo_neck=None,
                 history_stereo_prev_step=1,
                 
                 do_future_stereo_fusion=False,
                 future_stereo_prev_step=1,
                 
                 **kwargs):
        super(SOLOFusionSequential2, self).__init__(**kwargs)

        #### Prior to history fusion, do some per-sample pre-processing.
        self.single_bev_num_channels = self.img_view_transformer.numC_Trans

        # Lightweight MLP
        self.embed = nn.Sequential(
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_bev_num_channels, self.single_bev_num_channels, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(self.single_bev_num_channels),
            nn.ReLU(inplace=True))

        # Preprocessing like BEVDet4D
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)

        #### Deal with history
        self.do_history = do_history
        self.do_future = do_future
        
        if self.do_history and not self.do_future:
            self.interpolation_mode = interpolation_mode

            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
        
        elif self.do_future:
            self.interpolation_mode = interpolation_mode
            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            
            self.future_cat_num = future_cat_num
            # Embed each sample with its relative temporal offset with current timestep
            self.history_keyframe_time_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(self.single_bev_num_channels),
                nn.ReLU(inplace=True))

            # Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                nn.Conv2d(self.single_bev_num_channels * (self.future_cat_num + self.history_cat_num + 1),
                        history_cat_conv_out_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.BatchNorm2d(history_cat_conv_out_channels),
                nn.ReLU(inplace=True))

            self.history_sweep_time = None

            self.history_bev = None
            self.history_seq_ids = None
            self.history_forward_augs = None
            
            self.future_bev = None
            self.future_seq_ids = None
            self.future_forward_augs = None
        

        #### Stereo depth fusion
        self.do_history_stereo_fusion = do_history_stereo_fusion
        if self.do_history_stereo_fusion:
            self.stereo_neck = stereo_neck
            if self.stereo_neck is not None:
                self.stereo_neck = builder.build_neck(self.stereo_neck)
            self.history_stereo_prev_step = history_stereo_prev_step
            
        self.prev_stereo_img_feats = None # B x N x C x H x W
        self.prev_stereo_global_to_img = None # B x N x 4 x 4
        self.prev_stereo_img_forward_augs = None
        
        self.do_future_stereo_fusion = do_future_stereo_fusion
        if self.do_future_stereo_fusion:
            self.future_stereo_prev_step = future_stereo_prev_step
            
        self.future_prev_stereo_img_feats = None # B x N x C x H x W
        self.future_prev_stereo_global_to_img = None # B x N x 4 x 4
        self.future_prev_stereo_img_forward_augs = None
        
        
        self.use_gt_velocity = use_gt_velocity
        

        self.fp16_enabled = False

    @auto_fp16()
    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        backbone_feats = self.img_backbone(imgs)
        
        neck_feats = self.img_neck(backbone_feats)
        if isinstance(neck_feats, list):
            assert len(neck_feats) == 1 # SECONDFPN returns a length-one list
            neck_feats = neck_feats[0]
            
        _, output_dim, ouput_H, output_W = neck_feats.shape
        neck_feats = neck_feats.view(B, N, output_dim, ouput_H, output_W)

        if self.do_history_stereo_fusion:
            backbone_feats_detached = [tmp.detach() for tmp in backbone_feats]
            stereo_feats = self.stereo_neck(backbone_feats_detached)
            if isinstance(stereo_feats, list):
                assert len(stereo_feats) == 1 # SECONDFPN returns a trivial list
                stereo_feats = stereo_feats[0]
            stereo_feats = F.normalize(stereo_feats, dim=1, eps=self.img_view_transformer.stereo_eps)
            return neck_feats, stereo_feats.view(B, N, *stereo_feats.shape[1:])
        else:
            return neck_feats, None

    @force_fp32()
    def get_depth_loss(self, depth_gt, depth):
        """
        This was updated to be more similar to BEVDepth's original depth loss function.
        """
        B, N, H, W = depth_gt.shape
        fg_mask = (depth_gt != 0).view(-1) 
        depth_gt = (depth_gt - self.img_view_transformer.grid_config['dbound'][0])\
                /self.img_view_transformer.grid_config['dbound'][2]
        depth_gt = torch.clip(torch.floor(depth_gt), 0,
                            self.img_view_transformer.D).to(torch.long)
        assert depth_gt.max() < self.img_view_transformer.D

        depth_gt_logit = F.one_hot(depth_gt.reshape(-1),
                                   num_classes=self.img_view_transformer.D)
        depth_gt_logit = depth_gt_logit.reshape(B, N, H, W,
                                                self.img_view_transformer.D).permute(
            0, 1, 4, 2, 3).to(torch.float32) # B x N x D x H x W
        depth = depth.view(B, N, self.img_view_transformer.D, H, W).softmax(dim=2)
        
        depth_gt_logit = depth_gt_logit.permute(0, 1, 3, 4, 2).view(-1, self.img_view_transformer.D)    # [B*N*H*W, D]
        depth = depth.permute(0, 1, 3, 4, 2).contiguous().view(-1, self.img_view_transformer.D) # [B*N*H*W, D]

        loss_depth = (F.binary_cross_entropy(
                depth[fg_mask],
                depth_gt_logit[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum()))
        loss_depth = self.img_view_transformer.loss_depth_weight * loss_depth
        return loss_depth

    @force_fp32(apply_to=('rots', 'trans', 'intrins', 'post_rots', 'post_trans'))
    def process_stereo_before_fusion(self, stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans, future_stereo_feats=None):
        # NOTE: This is written to technically support history_stereo_prev_step > 1, 
        # but I haven't actually run any experiments on that, so please double check.

        # This is meant to deal with start of sequences, initializing variables, etc. It also returns
        # The current global to img transofmrations.
        # This *does not* update the history yet.

        B, N, C, H, W = stereo_feats.shape
        # list of start_of_sequence
        start_of_sequence = torch.BoolTensor([  # start_of_sequence是布尔值，表示当前img是否是sequence的第一帧
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(stereo_feats.device)
        
        global_to_curr_lidar_rt = torch.stack([
            single_img_metas['global_to_curr_lidar_rt']
            for single_img_metas in img_metas]).to(rots[0]) # B x 4 x 4
          
        lidar_forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas)    # 根据single_img_metas得到transformation matrix
            for single_img_metas in img_metas], dim=0).to(rots[0].device)


        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x N x 3 or B x N x 3 x 3
        cam_to_cam_aug = rots[0].new_zeros((B, N, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = post_rots[0]
        cam_to_cam_aug[:, :, :3, 3] = post_trans[0]

        intrins4x4 = rots[0].new_zeros((B, N, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = intrins[0]

        cam_to_lidar_aug = rots[0].new_zeros((B, N, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = rots[0]
        cam_to_lidar_aug[:, :, :3, 3] = trans[0]

        ## Okay, to go from global to augmed cam (XYD)....
        # We can go from global to (X*D, X*D, D), then we need user to divide by D,
        # Then we can apply the img augs. So, we need to store both.
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug
        global_to_img = (intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
                         @ lidar_forward_augs[:, None, :, :] @ global_to_curr_lidar_rt[:, None, :, :])
        img_forward_augs = cam_to_cam_aug

        ### Then, let's check if stereo saved values are none or we're the first in the sequence.
        if self.prev_stereo_img_feats is None:
            # For the history_stereo_prev_step dimension, the "first" one will be the most recent one
            self.prev_stereo_img_feats = stereo_feats.detach().clone()[:, None, :, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1, 1) # B x history_stereo_prev_step x N x C x H x W
            self.prev_stereo_global_to_img = global_to_img.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            self.prev_stereo_img_forward_augs = img_forward_augs.clone()[:, None, :, :, :].repeat(
                1, self.history_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            # self.prev_stereo_frame_idx = stereo_feats.new_zeros((B))[:, None].repeat(
            #     1, self.history_stereo_prev_step) # B x history_stereo_prev_step
        else:
            self.prev_stereo_img_feats[start_of_sequence] = stereo_feats[start_of_sequence].unsqueeze(1).detach().clone()
            self.prev_stereo_global_to_img[start_of_sequence] = global_to_img[start_of_sequence].unsqueeze(1).clone()
            self.prev_stereo_img_forward_augs[start_of_sequence] = img_forward_augs[start_of_sequence].unsqueeze(1).clone()

        # These are both B x N x 4 x 4. Want the result to be B x prev_N x curr_N x 4 x 4
        curr_unaug_cam_to_prev_unaug_cam = self.prev_stereo_global_to_img[:, 0][:, :, None, :, :] @ torch.inverse(global_to_img)[:, None, :, :, :]
        
        if not self.do_future:
            return (self.prev_stereo_img_feats[:, self.history_stereo_prev_step - 1], 
                    self.prev_stereo_global_to_img[:, self.history_stereo_prev_step - 1], 
                    self.prev_stereo_img_forward_augs[:, self.history_stereo_prev_step - 1],
                    global_to_img,
                    img_forward_augs,
                    curr_unaug_cam_to_prev_unaug_cam)
            
        
        B, N, C, H, W = future_stereo_feats.shape

        global_to_future_lidar_rt = torch.stack([
            single_img_metas['global_to_future_lidar_rt']
            for single_img_metas in img_metas]).to(rots[1]) # B x 4 x 4

        ### First, let's figure out global to img.
        ## Let's make them into 4x4s for my sanity
        # Each of the rots, etc are B x N x 3 or B x N x 3 x 3
        cam_to_cam_aug = rots[1].new_zeros((B, N, 4, 4))
        cam_to_cam_aug[:, :, 3, 3] = 1
        cam_to_cam_aug[:, :, :3, :3] = post_rots[1]
        cam_to_cam_aug[:, :, :3, 3] = post_trans[1]

        intrins4x4 = rots[1].new_zeros((B, N, 4, 4))
        intrins4x4[:, :, 3, 3] = 1
        intrins4x4[:, :, :3, :3] = intrins[1]

        cam_to_lidar_aug = rots[1].new_zeros((B, N, 4, 4))
        cam_to_lidar_aug[:, :, 3, 3] = 1
        cam_to_lidar_aug[:, :, :3, :3] = rots[1]
        cam_to_lidar_aug[:, :, :3, 3] = trans[1]

        ## Okay, to go from global to augmed cam (XYD)....
        # We can go from global to (X*D, X*D, D), then we need user to divide by D,
        # Then we can apply the img augs. So, we need to store both.
        # Global -> Lidar unaug -> lidar aug -> cam space unaug -> cam xyd unaug
        future_global_to_img = (intrins4x4 @ torch.inverse(cam_to_lidar_aug) 
                         @ lidar_forward_augs[:, None, :, :] @ global_to_future_lidar_rt[:, None, :, :])
        
        future_img_forward_augs = cam_to_cam_aug

        ### Then, let's check if stereo saved values are none or we're the first in the sequence.
        if self.future_prev_stereo_img_feats is None:
            # For the history_stereo_prev_step dimension, the "first" one will be the most recent one
            self.future_prev_stereo_img_feats = future_stereo_feats.detach().clone()[:, None, :, :, :, :].repeat(
                1, self.future_stereo_prev_step, 1, 1, 1, 1) # B x history_stereo_prev_step x N x C x H x W
            self.future_prev_stereo_global_to_img = future_global_to_img.clone()[:, None, :, :, :].repeat(
                1, self.future_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            self.future_prev_stereo_img_forward_augs = future_img_forward_augs.clone()[:, None, :, :, :].repeat(
                1, self.future_stereo_prev_step, 1, 1, 1) # B x history_stereo_prev_step x N x 4 x 4
            # self.prev_stereo_frame_idx = stereo_feats.new_zeros((B))[:, None].repeat(
            #     1, self.history_stereo_prev_step) # B x history_stereo_prev_step
        else:
            self.future_prev_stereo_img_feats[start_of_sequence] = future_stereo_feats[start_of_sequence].unsqueeze(1).detach().clone()
            self.future_prev_stereo_global_to_img[start_of_sequence] = future_global_to_img[start_of_sequence].unsqueeze(1).clone()
            self.future_prev_stereo_img_forward_augs[start_of_sequence] = future_img_forward_augs[start_of_sequence].unsqueeze(1).clone()

        # These are both B x N x 4 x 4. Want the result to be B x prev_N x curr_N x 4 x 4
        future_unaug_cam_to_future_prev_unaug_cam = self.future_prev_stereo_global_to_img[:, 0][:, :, None, :, :] @ torch.inverse(future_global_to_img)[:, None, :, :, :]
        
        return (self.prev_stereo_img_feats[:, self.history_stereo_prev_step - 1], 
                    self.prev_stereo_global_to_img[:, self.history_stereo_prev_step - 1], 
                    self.prev_stereo_img_forward_augs[:, self.history_stereo_prev_step - 1],
                    global_to_img,
                    img_forward_augs,
                    curr_unaug_cam_to_prev_unaug_cam), \
                (self.future_prev_stereo_img_feats[:, self.future_stereo_prev_step - 1], 
                    self.future_prev_stereo_global_to_img[:, self.future_stereo_prev_step - 1], 
                    self.future_prev_stereo_img_forward_augs[:, self.future_stereo_prev_step - 1],
                    future_global_to_img,
                    future_img_forward_augs,
                    future_unaug_cam_to_future_prev_unaug_cam)
        

    def process_stereo_for_next_timestep(self, img_metas, stereo_feats, global_to_img, img_forward_augs, future_stereo_feats=None, future_global_to_img=None, future_img_forward_augs=None):
        self.prev_stereo_img_feats[:, 1:] = self.prev_stereo_img_feats[:, :-1].clone() # Move
        self.prev_stereo_img_feats[:, 0] = stereo_feats.detach().clone()
        self.prev_stereo_global_to_img[:, 1:] = self.prev_stereo_global_to_img[:, :-1].clone() # Move
        self.prev_stereo_global_to_img[:, 0] = global_to_img.clone()
        self.prev_stereo_img_forward_augs[:, 1:] = self.prev_stereo_img_forward_augs[:, :-1].clone() # Move
        self.prev_stereo_img_forward_augs[:, 0] = img_forward_augs.detach().clone()
        
        if self.do_future:
            self.future_prev_stereo_img_feats[:, 1:] = self.future_prev_stereo_img_feats[:, :-1].clone() # Move
            self.future_prev_stereo_img_feats[:, 0] = future_stereo_feats.detach().clone()
            self.future_prev_stereo_global_to_img[:, 1:] = self.future_prev_stereo_global_to_img[:, :-1].clone() # Move
            self.future_prev_stereo_global_to_img[:, 0] = future_global_to_img.clone()
            self.future_prev_stereo_img_forward_augs[:, 1:] = self.future_prev_stereo_img_forward_augs[:, :-1].clone() # Move
            self.future_prev_stereo_img_forward_augs[:, 0] = future_img_forward_augs.detach().clone()

    @force_fp32()
    def fuse_history(self, curr_bev, future_curr_bev, img_metas):
        
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = torch.stack([
            generate_forward_transformation_matrix(single_img_metas) 
            for single_img_metas in img_metas], dim=0).to(curr_bev)
        curr_to_prev_lidar_rt = torch.stack([
            single_img_metas['curr_to_prev_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        
        

        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)
            
            self.timestamp = torch.LongTensor([
                single_img_metas['timestamp'] 
                for single_img_metas in img_metas]).to(curr_bev.device)
        
        self.timestamp[start_of_sequence] = torch.LongTensor([single_img_metas['timestamp'] for single_img_metas in img_metas]).to(curr_bev.device)[start_of_sequence]

        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in history with the curr_bev information
        self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
        self.history_sweep_time -= 1 # new timestep, everything in history gets pushed back one.
        self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
        self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        ## Get grid idxs & grid2bev first.
        n, c, h, w = curr_bev.shape

        # Generate grid
        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
        grid = torch.stack(
            (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,4,1)

        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        '''
        [[dx, 0,  start_x, 0], 
         [0,  dy, start_y, 0], 
         [0,  0,  1,       0], 
         [0,  0,  0,       1]]
        '''
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.dx[0]
        feat2bev[1, 1] = self.img_view_transformer.dx[1]
        feat2bev[0, 3] = self.img_view_transformer.bx[0] - self.img_view_transformer.dx[0] / 2.
        feat2bev[1, 3] = self.img_view_transformer.bx[1] - self.img_view_transformer.dx[1] / 2.
        feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations .
        
        if not self.use_gt_velocity:
            rt_flow_curr2prev = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_lidar_rt
                    @ torch.inverse(forward_augs) @ feat2bev)
            
            grid_curr2prev = rt_flow_curr2prev.view(n, 1, 1, 4, 4) @ grid
        else:
            temp_grid = torch.inverse(forward_augs).view(n, 1, 1, 4, 4) @ feat2bev @ grid
            temp_grid_clone = temp_grid.clone()
            
            for nim in range(len(img_metas)):
                boxes = img_metas[nim]['gt_boxes']
                velocities = img_metas[nim]['gt_velocity']
                for b, v in zip(boxes, velocities):
                    idx = (temp_grid_clone[nim][:, :, 0] >= (b[0] - b[3])) & (temp_grid_clone[nim][:, :, 0] <= (b[0] + b[3])) & (temp_grid_clone[nim][:, :, 1] >= (b[1] - b[4])) & (temp_grid_clone[nim][:, :, 1] <= (b[1] + b[4]))
                    temp_grid[nim][idx.squeeze()][:, :2, 0] -= torch.FloatTensor(v).to(curr_bev.device) * (img_metas[nim]['timestamp'] - self.timestamp[nim])
                self.timestamp[nim] = img_metas[nim]['timestamp']
            grid_curr2prev = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_lidar_rt).view(n, 1, 1, 4, 4) @ temp_grid
       
        
        # print(len(img_metas), self.history_forward_augs.shape, curr_to_prev_lidar_rt.shape, rt_flow.shape, self.single_bev_num_channels, curr_bev.shape, self.history_bev.shape)
        #           8                    [8, 4, 4]                       [8, 4, 4]               [8, 4, 4]              80              [8, 80, 128, 128] [8, 1280, 128, 128]
        # normalize and sample
        normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
        grid_curr2prev = grid_curr2prev[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        sampled_history_bev = F.grid_sample(self.history_bev, grid_curr2prev.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        
        if not self.do_future:
            feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W

            # Reshape and concatenate features and timestep
            feats_to_return = feats_cat.reshape(
                feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
            feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x H x W
            
            # Time conv
            feats_to_return = self.history_keyframe_time_conv(
                feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                    feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 x H x W

            # Cat keyframes & conv
            feats_to_return = self.history_keyframe_cat_conv(
                feats_to_return.reshape(
                    feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
            
            # Update history by moving everything down one group of single_bev_num_channels channels
            # and adding in curr_bev.
            # Clone is necessary since we're doing in-place operations on self.history_bev
            self.history_bev = feats_cat[:, :-self.single_bev_num_channels, :, :].detach().clone()
            self.history_sweep_time = self.history_sweep_time[:, :-1]

            self.history_forward_augs = forward_augs.clone()
            return feats_to_return.clone()
                
        
        self.history_forward_augs = forward_augs.clone()
    
    
        # ================================================================================================================================================================

        future_id = torch.tensor([
            single_img_metas['future_id']
            for single_img_metas in img_metas]).to(curr_bev)
    
        curr_to_future_lidar_rt = torch.stack([
            single_img_metas['curr_to_future_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        
        
        future_to_future_prev_lidar_rt = torch.stack([
            single_img_metas['future_to_future_prev_lidar_rt']
            for single_img_metas in img_metas]).to(curr_bev)
                
        
        ## Deal with first batch
        if self.future_bev is None:
            self.future_bev = future_curr_bev.clone()
            self.future_seq_ids = seq_ids.clone()
            self.future_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be future
            self.future_bev = future_curr_bev.repeat(1, self.future_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.future_sweep_time = curr_bev.new_ones(curr_bev.shape[0], self.future_cat_num) * future_id.reshape(-1, 1)

        self.future_bev = self.future_bev.detach()

        assert self.future_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, future id and seq id should be same.
        assert (self.future_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
            "{}, {}, {}".format(self.future_seq_ids, seq_ids, start_of_sequence)
        
        ## Replace all the new sequences' positions in future with the curr_bev information
        self.future_bev[start_of_sequence] = future_curr_bev[start_of_sequence].repeat(1, self.future_cat_num, 1, 1)
        self.future_sweep_time -= 1 # new timestep, everything in history gets pushed back one.
        self.future_sweep_time[start_of_sequence] = (curr_bev.new_ones(curr_bev.shape[0], self.future_cat_num) * future_id.reshape(-1, 1))[start_of_sequence] # zero the new sequence timestep starts
        self.future_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
        self.future_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        
        # 将future_prev转换到future
        rt_flow_future2future_prev = (torch.inverse(feat2bev) @ self.future_forward_augs @ future_to_future_prev_lidar_rt @ torch.inverse(forward_augs) @ feat2bev)
        grid_future2future_prev = rt_flow_future2future_prev.view(n, 1, 1, 4, 4) @ grid
       
        # normalize and sample
        grid_future2future_prev = grid_future2future_prev[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        
        sampled_future_bev = F.grid_sample(self.future_bev, grid_future2future_prev.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)
        self.future_bev = torch.cat((future_curr_bev, sampled_future_bev), dim=1)[:, :-self.single_bev_num_channels]
        
        
        
        # 将future转换到curr
        rt_flow_curr2future = (torch.inverse(feat2bev) @ forward_augs @ curr_to_future_lidar_rt @ torch.inverse(forward_augs) @ feat2bev)
        grid_curr2future = rt_flow_curr2future.view(n, 1, 1, 4, 4) @ grid
       
        # normalize and sample
        grid_curr2future = grid_curr2future[:,:,:,:2,0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0
        sampled_future_bev = F.grid_sample(self.future_bev, grid_curr2future.to(curr_bev.dtype), align_corners=True, mode=self.interpolation_mode)

        ## Update future
        # Add in current frame to features & timestep
        self.future_sweep_time = torch.cat([future_id.reshape(-1, 1), self.future_sweep_time], dim=1)[:, :-1] # B x T -> B x T
        feats_cat = torch.cat([sampled_future_bev, curr_bev, sampled_history_bev], dim=1) # B x (T + 1 + T) * 80 x H x W

        
        # Reshape and concatenate features and timestep
        feats_to_return = feats_cat.reshape(
            feats_cat.shape[0], self.future_cat_num + self.history_cat_num + 1, self.single_bev_num_channels, feats_cat.shape[2], feats_cat.shape[3]) # B x (1 + T) x 80 x H x W
        feats_to_return = torch.cat(
            [feats_to_return, torch.cat((self.future_sweep_time, self.history_sweep_time), dim=1)[:, :, None, None, None].repeat(
                1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
            ], dim=2) # B x (T + 1 + T) x 81 x H x W
        
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (T + 1 + T) x 80 x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, feats_to_return.shape[3], feats_to_return.shape[4])) # B x C x H x W
        
        # Update history by moving everything down one group of single_bev_num_channels channels
        # and adding in curr_bev.
        # Clone is necessary since we're doing in-place operations on self.history_bev
        self.history_bev = feats_cat[:, (self.single_bev_num_channels*self.future_cat_num):-self.single_bev_num_channels, :, :].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]

        self.future_forward_augs = forward_augs.clone()
        
        return feats_to_return.clone()
    
    

    def extract_img_feat(self, img, img_metas, gt_bboxes_3d=None):
        inputs = img
        """Extract features of images."""
        B, N, _, H, W = inputs[0].shape
        rots, trans, intrins, post_rots, post_trans = inputs[1:6]
        if self.do_future:
            N = N // 2
            imgs = inputs[0].view(B, N, 2, 3, H, W)
            imgs = torch.split(imgs, 1, dim=2)
            imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W
            
            extra = [rots.view(B, 2, N, 3, 3),
                    trans.view(B, 2, N, 3),
                    intrins.view(B, 2, N, 3, 3),
                    post_rots.view(B, 2, N, 3, 3),
                    post_trans.view(B, 2, N, 3)]
        else:
            imgs = inputs[0].view(B, N, 1, 3, H, W)
            imgs = torch.split(imgs, 1, dim=2)
            imgs = [tmp.squeeze(2) for tmp in imgs] # List of imgs each B x N x 3 x H x W

            extra = [rots.view(B, 1, N, 3, 3),
                    trans.view(B, 1, N, 3),
                    intrins.view(B, 1, N, 3, 3),
                    post_rots.view(B, 1, N, 3, 3),
                    post_trans.view(B, 1, N, 3)]
        
        extra = [torch.split(t, 1, dim=1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra] # each B x N x 3 (x 3)
        rots, trans, intrins, post_rots, post_trans = extra

        bev_feat_list = []
        depth_digit_list = []
        geom_list = []
                
        curr_img_encoder_feats, curr_stereo_feats = self.image_encoder(imgs[0])
        
        if self.do_future:
            future_img_encoder_feats, future_stereo_feats = self.image_encoder(imgs[1])

        
        if not self.do_history_stereo_fusion:
            curr_bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0])
            if self.do_future:
                future_bev_feat, future_depth_digit = self.img_view_transformer(
                future_img_encoder_feats, 
                rots[1], trans[1], intrins[1], post_rots[1], post_trans[1], do_stereo_fusion=False)
        else:
            if self.do_future:           
                (prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam), \
                    (future_prev_stereo_feats, future_prev_global2img, future_prev_img_forward_aug, future_global2img, future_img_forward_aug, future_unaug_cam_to_prev_unaug_cam) = \
                    self.process_stereo_before_fusion(curr_stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans, future_stereo_feats)
            else:
                prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam = \
                    self.process_stereo_before_fusion(curr_stereo_feats, img_metas, rots, trans, intrins, post_rots, post_trans, None)
                
            curr_bev_feat, depth_digit = self.img_view_transformer(
                curr_img_encoder_feats, 
                rots[0], trans[0], intrins[0], post_rots[0], post_trans[0],
                curr_stereo_feats, prev_stereo_feats, prev_global2img, prev_img_forward_aug, curr_global2img, curr_img_forward_aug, curr_unaug_cam_to_prev_unaug_cam)
            
            if self.do_future:
                future_bev_feat, future_depth_digit = self.img_view_transformer(
                    future_img_encoder_feats, 
                    rots[1], trans[1], intrins[1], post_rots[1], post_trans[1],
                    future_stereo_feats, future_prev_stereo_feats, future_prev_global2img, future_prev_img_forward_aug, future_global2img, future_img_forward_aug, future_unaug_cam_to_prev_unaug_cam)
                # prev_stereo_img_feats、prev_stereo_global_to_img、prev_stereo_img_forward_augs全部向后移一位，当前的放在第0位
                self.process_stereo_for_next_timestep(img_metas, curr_stereo_feats, curr_global2img, curr_img_forward_aug, future_stereo_feats, future_global2img, future_img_forward_aug)
            else:
                self.process_stereo_for_next_timestep(img_metas, curr_stereo_feats, curr_global2img, curr_img_forward_aug)
        

        curr_bev_feat = self.pre_process_net(curr_bev_feat)[0] # singleton list
        curr_bev_feat = self.embed(curr_bev_feat)

        if self.do_future:
            future_bev_feat = self.pre_process_net(future_bev_feat)[0] # singleton list
            future_bev_feat = self.embed(future_bev_feat)


        # Fuse History
        if self.do_history and self.do_future:
            bev_feat = self.fuse_history(curr_bev_feat, future_bev_feat, img_metas)
        elif self.do_history:
            bev_feat = self.fuse_history(curr_bev_feat, None, img_metas)
        else:
            bev_feat = curr_bev_feat

        x = self.bev_encoder(bev_feat)
        # print(imgs[0].shape, bev_feat.shape, depth_digit.shape)
        # [8, 6, 3, 256, 704] [8, 160, 128, 128] [48, 112, 16, 44]
        return x, depth_digit

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

        # print(img_metas[0]['start_of_sequence'], img_metas[0]['timestamp'])
        losses = dict()
        img_feats, depth = self.extract_img_feat(img_inputs, img_metas)

        # If we're training depth...
        depth_gt = img_inputs[-1] 
        B,N,H,W = depth_gt.shape
        # depth_gt = torch.split(depth_gt.view(B,2,N//2,H,W), 1, 1)[0].squeeze(1)
        
        
        loss_depth = self.get_depth_loss(depth_gt, depth)
        losses['loss_depth'] = loss_depth
        
        # Get box losses
        bbox_outs = self.pts_bbox_head(img_feats)
        losses_pts = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, bbox_outs)
        losses.update(losses_pts)

        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        img_feats, _ = self.extract_img_feat(img, img_metas)
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)

        bbox_list = [dict(pts_bbox=bbox_pts[0])]

        return bbox_list
