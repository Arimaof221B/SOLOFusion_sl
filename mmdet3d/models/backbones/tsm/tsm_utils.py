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

"""Utils functions for TSM."""

from typing import Tuple
import torch
from torch.nn import functional as F


def prepare_inputs(inputs):
    """Deduces input mode for TSM."""
    # Deduce if we run on TPU based on input shape.
    if len(inputs.shape) == 5:
        # Input is given in the standard [B, T, 3, H, W] format.
        tsm_mode = 'gpu'
        num_frames = inputs.shape[1]
        inputs = inputs.flatten(0, 1)
    else:
        # Input is given in the [T * B, 3, H, W] format.
        tsm_mode = 'tpu'
        num_frames = None
    return inputs, tsm_mode, num_frames


def prepare_outputs(
    outputs,
    tsm_mode: str,
    num_frames: int,
    reduce_mean: bool = True,
):
    """Processes output of TSM to undo the merging of batch and time."""
    # Get the shape without the batch/time dimension (for TSM batch and time are
    # merged in the first dimension).
    shape_no_bt = list(outputs.shape[1:])
    if tsm_mode == 'gpu':
        # Outputs are of the shape [B * num_frames, ..., n_channels].
        outputs = outputs.reshape(-1, num_frames, *shape_no_bt)
        if reduce_mean:
            outputs = torch.mean(outputs, dim=[1] + list(range(2, len(shape_no_bt) + 1)))
    elif tsm_mode.startswith('deflated'):
        # In deflated mode, outputs are already in the right format.
        pass
    else:
        raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                            f'\'deflated_0.x\' ({tsm_mode} given)')
    return outputs  # pytype: disable=bad-return-type  # numpy-scalars


def apply_temporal_shift(
    x,
    tsm_mode: str,
    num_frames: int,
    channel_shift_fraction: float = 0.125,
):
    """Performs a temporal shift: https://arxiv.org/abs/1811.08383 with mode."""
    if tsm_mode == 'gpu':
        outputs = temporal_shift(x, num_frames, channel_shift_fraction)
    elif tsm_mode.startswith('deflated'):
        alpha = float(tsm_mode.split('_')[1])
        outputs = temporal_shift_image_mode(x, channel_shift_fraction, alpha)
    else:
        raise ValueError('`tsm_mode` should be \'tpu\' or \'gpu\' or '
                            f'\'deflated_0.x\' ({tsm_mode} given)')
    return outputs


def temporal_shift_image_mode(x, channel_shift_fraction=0.125, alpha=0.3):
    """Temporal shift applied on single image (to emulate a fixed video)."""
    # B, H, W, C = batch_size, im_height, im_width, channels.
    # Input is (B, H, W, C).
    origin_shape = tuple(x.shape)
    n_channels = origin_shape[-1]
    n_shift = int(n_channels * channel_shift_fraction)
    # Alpha emulates the effect of the padding when using a single frame.
    shifted_backward = alpha * x[:, :, :, -n_shift: ]
    shifted_forward = alpha * x[:, :, :, : n_shift]
    no_shift = x[:, :, :, n_shift: -n_shift]
    shifted_x = torch.cat([shifted_backward, no_shift, shifted_forward], dim=3)
    return shifted_x


def temporal_shift_gpu(
    x,
    num_frames: int,
    channel_shift_fraction: float = 0.125,
):
    """Performs a temporal shift: https://arxiv.org/abs/1811.08383."""
    # B, T, H, W, C = batch_size, num_frames, im_height, im_width, channels.
    # Input is (B * T, C, H, W).
    origin_shape = tuple(x.shape)
    reshaped_x = x.reshape(-1, num_frames, *origin_shape[1:])
    n_channels = origin_shape[1]
    n_shift = int(n_channels * channel_shift_fraction)

    new_shape = tuple(reshaped_x.shape)

    shifted_backward = reshaped_x[:, 1:, -n_shift: ]
    shifted_backward_padding = ((0, 0), (0, 1), (0, 0))
    shifted_backward = F.pad(shifted_backward, shifted_backward_padding)
    
    shifted_forward = reshaped_x[:, :-1, : n_shift]
    shifted_forward_padding = ((0, 0), (1, 0), (0, 0))
    shifted_forward = F.pad(shifted_forward, shifted_forward_padding)
    
    no_shift = reshaped_x[:, :, n_shift:-n_shift]
    
    shifted_x = torch.concatenate([shifted_backward, no_shift, shifted_forward], dim=2)
    shifted_x = shifted_x.reshape(-1, *origin_shape[1:])
    
    return shifted_x


def temporal_shift(x, num_frames, channel_shift_fraction=0.125): 
    nt, c, h, w = x.shape
    x = x.reshape(-1, num_frames, c, h, w)
    n_shift = int(c * channel_shift_fraction)
    out = torch.zeros_like(x)
        
    out[:, : -1, : n_shift] = x[:, 1:, : n_shift]               # forward
    out[:, :, n_shift: -n_shift] = x[:, :, n_shift: -n_shift]   # no_shift
    out[:, 1:, -n_shift:] = x[:, : -1, -n_shift:]                # backward

    return out.reshape(nt, c, h, w)
    