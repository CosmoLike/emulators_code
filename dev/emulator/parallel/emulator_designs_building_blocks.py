"""Grouped (per-bin) nn building block: the per-bin conv.

Only the conv gets a grouped twin (see emulator_designs.py for why the
per-bin ResMLP family was removed).
"""

import torch.nn as nn

from ..activations import activation_fcn


class GroupedCNNBlock(nn.Module):
  """
  Per-group 1D convolution: split the input into n_groups contiguous
  segments of length group_len and convolve each independently (a
  grouped Conv1d, groups=n_groups). No kernel crosses a group
  boundary -- so with a per-tomographic-bin layout, each bin's theta
  curve is refined without smoothing across the bin-boundary jumps a
  global conv would blur.

  Two convs with a nonlinearity between (so the per-group `channels`
  filters are useful, like CNNBlock). Input/output: (B, n_groups *
  group_len).

  Arguments:
    n_groups    = independent segments (= number of bins).
    group_len   = length of each segment (the padded per-bin length
                  = max bin size).
    kernel_size = kernel width (odd; same-padding keeps group_len).
    channels    = conv filters per group.
    act         = activation factory.
  """
  def __init__(self, n_groups, group_len, kernel_size=11,
               channels=16, act=activation_fcn):
    super().__init__()
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    pad = (kernel_size - 1) // 2
    self.n_groups  = n_groups
    self.group_len = group_len
    # 1 input channel per group -> `channels` filters per group;
    # groups=n_groups keeps every bin's conv independent.
    self.conv_in  = nn.Conv1d(in_channels=n_groups,
                              out_channels=n_groups * channels,
                              kernel_size=kernel_size,
                              padding=pad,
                              groups=n_groups)
    self.act_mid  = act(group_len)        # within-bin position act
    self.conv_out = nn.Conv1d(in_channels=n_groups * channels,
                              out_channels=n_groups,
                              kernel_size=kernel_size,
                              padding=pad,
                              groups=n_groups)
    self.act_out  = act(n_groups * group_len)

  def forward(self, x):
    # (B, n_groups*group_len) -> (B, n_groups, group_len): each
    # group becomes one channel the grouped conv treats alone.
    h = x.view(x.size(0), self.n_groups, self.group_len)
    h = self.conv_in(h)         # (B, n_groups*channels, group_len)
    h = self.act_mid(h)         # nonlinearity (channels matter)
    h = self.conv_out(h)        # (B, n_groups, group_len)
    h = h.view(x.size(0), -1)   # (B, n_groups*group_len)
    return self.act_out(h)
