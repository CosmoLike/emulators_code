"""Grouped (per-bin) nn building blocks."""

import torch
import torch.nn as nn

from ..activations import activation_fcn
from .activations import GroupedActivation


class GroupedLinear(nn.Module):
  """
  G independent Linear(in, out) layers run as one batched matmul
  over the group axis G. Weights stacked into a single (G, in, out)
  tensor, bias into (G, out).
  """
  def __init__(self, n_groups, in_features, out_features):
    super().__init__()
    # build G ordinary nn.Linear layers just to borrow their init,
    # then stack their weights/biases and discard them. l.weight is
    # (out, in); .t() -> (in, out) for the einsum; stack adds the
    # group axis.
    lins = []
    for _ in range(n_groups):
      lins.append(nn.Linear(in_features=in_features,
                            out_features=out_features))
    weights, biases = [], []
    for l in lins:
      weights.append(l.weight.detach().t())
      biases.append(l.bias.detach())
    self.weight = nn.Parameter(torch.stack(weights))
    self.bias   = nn.Parameter(torch.stack(biases))

  def forward(self, x):
    # x: (G, B, in) -- G heads, each with the same B input rows.
    # self.weight: (G, in, out) -- G stacked weight matrices.
    #
    # einsum("gbi,gio->gbo", x, weight):
    #   g = group/head axis. In both inputs and the output, so a
    #       batch axis: head g uses only weight[g], all G in parallel
    #       (one batched matmul).
    #   b = the B samples. Only in x and the output -> kept.
    #   i = input features. In both inputs but not the output, so
    #       einsum sums over it -> the matmul's contracted dimension.
    #   o = output features. Only in weight and the output -> kept.
    #
    # Element-by-element y[g] = x[g] @ W[g] for each group g, all G
    # at once:
    #   y[g, b, o] = sum_i  x[g, b, i] * weight[g, i, o]
    # the same as this triple loop, but on the GPU at once:
    #   for g in range(G):
    #     for b in range(B):
    #       for o in range(out):
    #         y[g,b,o] = sum(x[g,b,i]*weight[g,i,o]
    #                        for i in range(in))
    y = torch.einsum("gbi,gio->gbo", x, self.weight)   # (G,B,out)

    # self.bias: (G, out). unsqueeze(1) -> (G, 1, out) to broadcast
    # across the B axis (every sample in head g gets head g's bias).
    return y + self.bias.unsqueeze(1)


class GroupedAffine(nn.Module):
  """G independent Affine (per-group learnable scale + shift)."""
  def __init__(self, n_groups):
    super().__init__()
    # gain inits to 1, bias to 0 (as in the scalar Affine), with an
    # extra group axis and trailing 1s to broadcast over (G, B, D).
    self.gain = nn.Parameter(torch.ones(n_groups, 1, 1))
    self.bias = nn.Parameter(torch.zeros(n_groups, 1, 1))

  def forward(self, x):                 # x: (G, B, D)
    return x * self.gain + self.bias


class GroupedResBlock(nn.Module):
  """
  The grouped twin of ResBlock: n_layers grouped Linear(D,D), each
  followed by a grouped Affine "norm" and grouped activation, skip
  added before the last norm/act (pre-activation residual).
  Identical logic to ResBlock, with the group axis threaded
  through.
  """
  def __init__(self, n_groups, size, n_layers=2):
    super().__init__()
    lins, norms, acts = [], [], []
    for _ in range(n_layers):
      lins.append(GroupedLinear(n_groups, size, size))
      norms.append(GroupedAffine(n_groups))
      acts.append(GroupedActivation(n_groups, size))
    self.lins  = nn.ModuleList(lins)
    self.norms = nn.ModuleList(norms)
    self.acts  = nn.ModuleList(acts)

  def forward(self, x):                 # x: (G, B, D)
    xskip = x
    out = x
    n = len(self.lins)
    for i in range(n):
      out = self.lins[i](out)
      if i == n - 1:                    # add skip before last
        out = out + xskip               #   norm + activation
      out = self.acts[i](self.norms[i](out))
    return out


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
