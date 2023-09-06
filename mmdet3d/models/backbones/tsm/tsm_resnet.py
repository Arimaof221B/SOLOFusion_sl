# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Temporal Shift Module w/ ResNet-50 and ResNet-101.

Based on:
  TSM: Temporal Shift Module for Efficient Video Understanding
  Ji Lin, Chuang Gan, Song Han
  https://arxiv.org/pdf/1811.08383.pdf.
"""

from collections import OrderedDict
from torch import nn
import torch
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Sequence, Union
from absl import logging

import typing_extensions

from .tsm_utils import prepare_inputs, prepare_outputs, apply_temporal_shift
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from ...builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


class NormalizeFn(typing_extensions.Protocol):

  def __call__(self, x, is_training: bool):
    pass


class TSMResNetBlockV1(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        stride: int,
        use_projection: bool,
        tsm_mode: str,
        channel_shift_fraction: float = 0.125,
        num_frames: int = 8,
        dilation: int = 1,
        use_bottleneck: bool = False,
        norm_cfg = dict(type='BN'),
        style='pytorch'
        ):
        """
        Initializes the TSMResNetBlock module.

        Args:
        output_channels: Number of output channels.
        stride: Stride used in convolutions.
        use_projection: Whether to use a projection for the shortcut.
        tsm_mode: Mode for TSM ('gpu' or 'tpu' or 'deflated_0.x').
        normalize_fn: Function used for normalization.
        channel_shift_fraction: The fraction of temporally shifted channels. If
            `channel_shift_fraction` is 0, the block is the same as a normal ResNet
            block.
        num_frames: Size of frame dimension in a single batch example.
        rate: dilation rate.
        use_bottleneck: use a bottleneck (resnet-50 and above),
        name: The name of the module.
        """
        super().__init__()
        self._out_channels = (out_channels if use_bottleneck else out_channels // 4)
        self._bottleneck_channels = out_channels // 4
        self._stride = stride
        self._dilation = dilation
        self._use_projection = use_projection
        self._tsm_mode = tsm_mode
        self._channel_shift_fraction = channel_shift_fraction
        self._num_frames = num_frames
        self._use_bottleneck = use_bottleneck
        
        
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1
        
        if use_projection:
            self.downsample = nn.Sequential(nn.Conv2d(
                                            in_channels=in_channels,
                                            out_channels=self._out_channels,
                                            kernel_size=1,
                                            stride=self._stride,
                                            padding=0,
                                            bias=False), 
                                            nn.BatchNorm2d(self._out_channels))
        
        self.norm1_name, norm1 = build_norm_layer(norm_cfg, self._bottleneck_channels, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, self._bottleneck_channels, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(norm_cfg, self._out_channels, postfix=3)
        
        self.conv1 = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=self._bottleneck_channels,
                kernel_size=1 if self._use_bottleneck else 3,
                stride=self.conv1_stride if self._use_bottleneck else self._stride,
                padding=0 if self._use_bottleneck else 1, 
                bias=False
            )

        # self.bn1 = nn.BatchNorm2d(self._bottleneck_channels)
        self.add_module(self.norm1_name, norm1)
        
        self.conv2 = nn.Conv2d(
                in_channels = self._bottleneck_channels, 
                out_channels=self._bottleneck_channels,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=1,
                dilation=self._dilation,
                bias=False
            )
        
        # self.bn2 = nn.BatchNorm2d(self._bottleneck_channels)
        self.add_module(self.norm2_name, norm2)
        
        self.conv3 = nn.Conv2d(
            in_channels=self._bottleneck_channels, 
            out_channels=self._out_channels,
            kernel_size=1 if self._use_bottleneck else 3,
            stride=1,
            padding=0 if self._use_bottleneck else 1,
            bias=False,
        )
        
        # self.bn3 = nn.BatchNorm2d(self._out_channels)
        self.add_module(self.norm3_name, norm3)
        

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)


    def forward(self, inputs): 
        """Connects the ResNetBlock module into the graph.

        Args:
        inputs: A 4-D float array of shape `[B, H, W, C]`.
        is_training: Whether to use training mode.

        Returns:
        A 4-D float array of shape
        `[B * num_frames, new_h, new_w, output_channels]`.
        """
        # ResNet V2 uses pre-activation, where the batch norm and relu are before
        # convolutions, rather than after as in ResNet V1.
        preact = inputs

        if self._use_projection:
            shortcut = self.downsample(preact)
        else:
            shortcut = inputs

        # Eventually applies Temporal Shift Module.
        if self._channel_shift_fraction != 0:
            preact = apply_temporal_shift(
                preact,
                tsm_mode=self._tsm_mode,
                num_frames=self._num_frames,
                channel_shift_fraction=self._channel_shift_fraction)

        # First convolution.
        residual = self.conv1(preact)
        residual = self.norm1(residual)
        residual = F.relu(residual)

        if self._use_bottleneck:
        # Second convolution.
            residual = self.conv2(residual)
            residual = self.norm2(residual)
            residual = F.relu(residual)

        # Third convolution.
        residual = self.conv3(residual)
        residual = self.norm3(residual)

        # NOTE: we do not use block multiplier.
        output = shortcut + residual
        output = F.relu(output)
        return output


class TSMResNetBlockV2(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        out_channels: int,
        stride: int,
        use_projection: bool,
        tsm_mode: str,
        channel_shift_fraction: float = 0.125,
        num_frames: int = 8,
        dilation: int = 1,
        use_bottleneck: bool = False,
        ):
        super().__init__()
    
    def forward(self):
        pass


class TSMResNetUnit(nn.Module):
    """Block group for TSM ResNet."""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        num_blocks: int,
        stride: int,
        tsm_mode: str,
        num_frames: int,
        channel_shift_fraction: float = 0.125,
        dilation: int = 1,
        use_bottleneck: bool = False,
        norm_cfg=dict(type='BN'),
        style='pytorch', 
        version: str = 'V1'
    ):
        """Creates a TSMResNet Unit.

        Args:
            output_channels: Number of output channels.
            num_blocks: Number of ResNet blocks in the unit.
            stride: Stride of the unit.
            tsm_mode: Which temporal shift module to use.
            num_frames: Size of frame dimension in a single batch example.
            normalize_fn: Function used for normalization.
            channel_shift_fraction: The fraction of temporally shifted channels. If
            `channel_shift_fraction` is 0, the block is the same as a normal ResNet
            block.
            rate: dilation rate.
            use_bottleneck: use a bottleneck (resnet-50 and above).
            name: The name of the module.
        """
        super().__init__()
        self._num_blocks = num_blocks
        self._stride = stride
        self._tsm_mode = tsm_mode
        self._channel_shift_fraction = channel_shift_fraction
        self._num_frames = num_frames
        self._dilation = dilation
        self._use_bottleneck = use_bottleneck
        
        
        for idx_block in range(self._num_blocks):
            if version == 'V1' or version == 'v1':
                block = TSMResNetBlockV1(
                    in_channels, 
                    out_channels,
                    stride=(self._stride if idx_block == 0 else 1),
                    dilation=(max(self._dilation // 2, 1) if idx_block == 0 else self._dilation),
                    use_projection=(idx_block == 0),
                    tsm_mode=self._tsm_mode,
                    channel_shift_fraction=self._channel_shift_fraction,
                    num_frames=self._num_frames, 
                    use_bottleneck=self._use_bottleneck, 
                    norm_cfg=norm_cfg, 
                    style=style
                )
            elif version == 'V2' or version == 'v2':
                block = TSMResNetBlockV2(
                    in_channels, 
                    out_channels,
                    stride=(self._stride if idx_block == 0 else 1),
                    dilation=(max(self._dilation // 2, 1) if idx_block == 0 else self._dilation),
                    use_projection=(idx_block == 0),
                    tsm_mode=self._tsm_mode,
                    channel_shift_fraction=self._channel_shift_fraction,
                    num_frames=self._num_frames, 
                    use_bottleneck=self._use_bottleneck, 
                    norm_cfg=norm_cfg
                )
            else:
                raise()
            in_channels = out_channels
            setattr(self, f"block_{idx_block}", block)

    def forward(self, x):
        """Connects the module to inputs.

        Args:
            inputs: A 4-D float array of shape `[B * num_frames, H, W, C]`.
            is_training: Whether to use training mode.

        Returns:
            A 4-D float array of shape
            `[B * num_frames, H // stride, W // stride, output_channels]`.
        """
        
        for idx_block in range(self._num_blocks): 
            x = getattr(self, f"block_{idx_block}")(x)
        
        return x


class TSMResNetV1(nn.Module):
    """TSM based on ResNet V2 as described in https://arxiv.org/abs/1603.05027."""

    # Endpoints of the model in order.
    VALID_ENDPOINTS = (
        'tsm_resnet_stem',
        'tsm_resnet_unit_1',
        'tsm_resnet_unit_2',
        'tsm_resnet_unit_3',
        'tsm_resnet_unit_4',
        'last_conv',
        'Embeddings',
    )

    def __init__(
        self,
        depth: int = 18,
        num_frames: int = 16,
        channel_shift_fraction: Union[float, Sequence[float]] = 0.125,
        frozen_stages: int = -1, 
        width_mult: int = 1,
        output_stride: int = 8,
        tsm_mode: str = 'gpu', 
        is_deflated: bool = False, 
        with_cp: bool = False, 
        out_indices = [2, 3], 
        norm_cfg = dict(type='BN'), 
        norm_eval=True, 
        style='pytorch'
        
    ):
        """Constructs a ResNet model.

        Args:
            normalize_fn: Function used for normalization.
            depth: Depth of the desired ResNet.
            num_frames: Number of frames (used in TPU mode).
            channel_shift_fraction: Fraction of channels that are temporally shifted,
            if `channel_shift_fraction` is 0, a regular ResNet is returned.
            width_mult: Whether or not to use a width multiplier.

        Raises:
            ValueError: If `channel_shift_fraction` or `depth` has invalid value.
        """
        super().__init__()

        if isinstance(channel_shift_fraction, float):
            channel_shift_fraction = [channel_shift_fraction] * 4

        if not all([0. <= x <= 1.0 for x in channel_shift_fraction]):
            raise ValueError(f'channel_shift_fraction ({channel_shift_fraction})'
                            ' all have to be in [0, 1].')

        
        self._channels = (256, 512, 1024, 2048)

        num_blocks = {
            18: (2, 2, 2, 2),
            34: (3, 4, 6, 3),
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
            200: (3, 24, 36, 3),
        }
        if depth not in num_blocks:
            raise ValueError(
                f'`depth` should be in {list(num_blocks.keys())} ({depth} given).')
        
        self._num_frames = num_frames
        self._num_blocks = num_blocks[depth]

        self._width_mult = width_mult
        self._channel_shift_fraction = channel_shift_fraction
        self._use_bottleneck = (depth >= 50)
        
        self.is_deflated = is_deflated
        self._with_cp = with_cp
        self._out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        
        
        self.tsm_resnet_stem = nn.Conv2d(3, 64 * self._width_mult, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64 * self._width_mult)
        
        
        norm_cfg=dict(type='BN')
        self.norm_cfg = norm_cfg
        
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64 * self._width_mult, postfix=1)
        self.add_module(self.norm1_name, norm1)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        in_channels = 64 * self._width_mult
        
        if output_stride == 4:
            strides = (1, 1, 1, 1)
            dilations = (1, 2, 4, 8)
        elif output_stride == 8:
            strides = (1, 2, 1, 1)
            dilations = (1, 1, 2, 4)
        elif output_stride == 16:
            strides = (1, 2, 2, 1)
            dilations = (1, 1, 1, 2)
        elif output_stride == 32:
            strides = (1, 2, 2, 2)
            dilations = (1, 1, 1, 1)
        else:
            raise ValueError('unsupported output stride')
        
        for unit_id, (channels, num_blocks, stride, dilation) in enumerate(
            zip(self._channels, self._num_blocks, strides, dilations)):
            unit = TSMResNetUnit(
                in_channels=in_channels, 
                out_channels=channels * self._width_mult,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                channel_shift_fraction=self._channel_shift_fraction[unit_id],
                num_frames=num_frames,
                tsm_mode=tsm_mode,
                use_bottleneck=self._use_bottleneck,
                style=style
            )
            setattr(self, f'tsm_resnet_unit_{unit_id + 1}', unit)
            in_channels = channels * self._width_mult

        self._freeze_stages()
    
    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)
    

    def forward(
        self,
        x,
        final_endpoint: str = 'Embeddings',
        alpha_deflation: float = 0.3,
        out_num_frames: Optional[int] = None,
    ):
        """Connects the TSM ResNetV2 module into the graph.

        Args:
            inputs: The input may be in one of two shapes; if the shape is `[B, T, C, H, W]`, 
            this module assumes the backend is a GPU (setting
            `tsm_mode='gpu'`) and `T` is treated the time dimension, with `B` being
            the batch dimension. This mode cannot be used when `is_deflated` is
            `true`. In this mode, the num_frames parameter passed to the constructor
            is ignored. If the shape is `[B, C, H, W]`, then the batch dimension is
            assumed to be of the form [B*T, C, H, W], where `T` is the number of
            frames in each video. This value may be set by passing `num_frames=n` to
            the constructor. The default value is `n=16` (beware this default is not
            the same as the default for the `TSMResNetBlock`, which has a default of
            8 frames). In this case, the module assumes it is being run on a TPU,
            and emits instructions that are more efficient for that case,
            using`tsm_mode`='tpu'` for the downstream blocks.
            is_training: Whether to use training mode.
            final_endpoint: Up to which endpoint to run / return.
            is_deflated: Whether or not to use the deflated version of the network.
            alpha_deflation: Deflation parameter to use for dealing with the padding
            effect.
            out_num_frames: Whether time is on first axis, for TPU performance
            output_stride: Stride of the final feature grid; possible values are
            4, 8, 16, or 32.  32 is the standard for TSM-ResNet. Others strides are
            achieved by converting strided to un-strided convolutions later in the
            network, while increasing the dilation rate for later layers.

        Returns:
            Network output at location `final_endpoint`. A float array which shape
            depends on `final_endpoint`.

        Raises:
            ValueError: If `final_endpoint` is not recognized.
        """

        # Prepare inputs for TSM.
        if self.is_deflated:
            if len(x.shape) != 4:
                raise ValueError(
                    'In deflated mode inputs should be given as [B, H, W, 3]')
            logging.warning(
                    'Deflation is an experimental feature and the API might change.')
            tsm_mode = f'deflated_{alpha_deflation}'
            num_frames = 1
        else:
            x, tsm_mode, num_frames = prepare_inputs(x)
            num_frames = num_frames or out_num_frames or self._num_frames

        self._final_endpoint = final_endpoint
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError(f'Unknown final endpoint {self._final_endpoint}')

        # Stem convolution.
        x = self.tsm_resnet_stem(x)
        x = self.norm1(x)
        x = self.maxpool(x)
        
        if self._final_endpoint == 'tsm_resnet_stem':
            x = prepare_outputs(x, tsm_mode, num_frames, reduce_mean=False)
            return x

        outs = []
        
        # Residual block.
        for unit_id, channels in enumerate(self._channels):
            end_point = f'tsm_resnet_unit_{unit_id + 1}'
            layer = getattr(self, end_point)
            if self._with_cp:
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)
            if unit_id in self._out_indices:
                outs.append(x)
            
            if self._final_endpoint == end_point:
                x = prepare_outputs(x, tsm_mode, num_frames, reduce_mean=False)
                return x

        x = F.relu(x)

        end_point = 'last_conv'
        if self._final_endpoint == end_point:
            x = prepare_outputs(x, tsm_mode, num_frames, reduce_mean=False)
            return x

        return tuple(outs)
        # x = torch.mean(x, dim=(2, 3))
        # # Prepare embedding outputs for TSM (temporal average of features).
        # x = prepare_outputs(x, tsm_mode, num_frames, reduce_mean=True)
        # assert self._final_endpoint == 'Embeddings'
        # return x
    
    
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    
    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
    

@BACKBONES.register_module()
class TSMResNet(nn.Module):
    def __init__(self,
        depth: int = 18,
        num_frames: int = 16,
        channel_shift_fraction: Union[float, Sequence[float]] = 0.125,
        frozen_stages=-1, 
        width_mult: int = 1,
        output_stride: int = 8,
        tsm_mode: str = 'gpu', 
        version: str = 'V1', 
        pretrained: str = '/home/arima/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth', 
        with_cp: bool = False, 
        out_indices = [2, 3], 
        norm_cfg=dict(type='BN'), 
        norm_eval=True, 
        style='pytorch'):
        super().__init__()
        
        if version == 'V1' or version == 'v1':
            self.model = TSMResNetV1(depth=depth,
                                    num_frames=num_frames,
                                    channel_shift_fraction=channel_shift_fraction,
                                    frozen_stages=frozen_stages, 
                                    width_mult=width_mult,
                                    output_stride=output_stride,
                                    tsm_mode=tsm_mode, 
                                    with_cp=with_cp, 
                                    out_indices=out_indices, 
                                    norm_cfg=norm_cfg, 
                                    norm_eval=norm_eval, 
                                    style=style)
        
        
        print("------------------loading pretrained weights------------------")
        state_dict = torch.load(pretrained)
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('fc'):
                continue
            elif k.startswith('conv1'):
                new_k = k.replace('conv1', 'tsm_resnet_stem')
            elif k.startswith('layer'):
                new_k = k.replace('layer', 'tsm_resnet_unit_')
                new_k = new_k.split('.')
                new_k[1] = 'block_' + new_k[1]
                new_k = '.'.join(new_k)
            else:
                new_k = k
            new_state_dict[new_k] = v
        
        self.model.load_state_dict(new_state_dict)
    
    def forward(self, x):
        return self.model(x)