"""Grouped (per-bin) nn building block: the per-bin convolution.

This module holds the one grouped twin of the package's conv blocks,
GroupedCNNBlock: a 1D convolution split into independent per-group
sub-convolutions, one group per tomographic bin, so no kernel weight
ever connects two bins. Only the conv stage has a grouped twin; the
per-bin ResMLP family was tested and removed (see
parallel/emulator_designs.py for the reasoning and the post-mortem
reference).

PS: grouped conv = a Conv1d whose `groups` option partitions input
and output channels into `groups` independent blocks (weights exist
only within a block); same-padding = zero-padding chosen so the
output length equals the input length; factory = a callable invoked
once per layer to build a fresh module (act(dim) -> nn.Module), so
layers never share learnable parameters.
"""

import torch.nn as nn

from ..activations import activation_fcn


class GroupedCNNBlock(nn.Module):
  """
  Per-group 1D convolution: split the input into n_groups contiguous
  segments of length group_len and convolve each independently (a
  grouped Conv1d, groups = n_groups). No kernel crosses a group
  boundary, so with a per-tomographic-bin layout each bin's theta
  curve is refined without smoothing across the bin-boundary jumps a
  global conv would blur.

  Two convs with a nonlinearity between: an expand conv (1 -> F
  filters per group) and a collapse conv (F -> 1 per group). The
  middle activation is what makes the F filters useful -- two
  stacked linear convs would compose into a single 1 -> 1 kernel
  and waste the channels (the expand-then-collapse rule, as in the
  root package's conv blocks).

  Forward shape flow:

      x  (B, G*L)                flattened per-bin layout
         │  view                 (B, G, L): one channel per bin
         ▼
      h  (B, G, L)
         │  conv_in (groups=G)   1 -> F filters per bin, width k
         ▼
      h  (B, G*F, L)
         │  act_mid              the mid nonlinearity (see above)
         ▼
      h  (B, G*F, L)
         │  conv_out (groups=G)  F -> 1 per bin, width k
         ▼
      h  (B, G, L)
         │  view                 back to the flat layout
         ▼
      h  (B, G*L)
         │  act_out              output activation
         ▼
      out (B, G*L)

      (legend: B = batch rows; G = n_groups, the tomographic bins;
       L = group_len, the padded per-bin length (the longest bin's
       kept theta count -- LSST-Y1 example: 26); F = channels, the
       conv filters per bin; k = kernel_size, same-padded so L is
       preserved.)

  Arguments:
    n_groups    = independent segments (= number of bins).
    group_len   = length of each segment (the padded per-bin length
                  = max bin size).
    kernel_size = kernel width (odd; same-padding keeps group_len).
    channels    = conv filters per group (F above).
    act         = activation factory, invoked as act(dim) once per
                  activation site (a fresh module each time, never
                  a shared instance). Defaults to activation_fcn,
                  the paper's H.
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
    h = self.act_mid(h)         # mid nonlinearity (channels matter)
    h = self.conv_out(h)        # (B, n_groups, group_len)
    h = h.view(x.size(0), -1)   # (B, n_groups*group_len)
    return self.act_out(h)
