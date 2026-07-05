"""Per-bin (grouped) CNN emulator model: the archived parallel variant.

This subpackage holds the per-bin experiment: refine each tomographic
bin's segment of the data vector independently, with no information
crossing a bin boundary. Only the conv stage survives, as
ParallelResCNN (a shared ResMLP trunk followed by a grouped per-bin
conv). There is deliberately no per-bin ResMLP: splitting the trunk
per bin throws away the parameter sharing of the cosmology map and
re-learns that hard shared map once per bin. That design was tested
on 2026-06-25 and rejected (worse than a single ResMLP at matched
parameters; the post-mortem lives in
notes/per-bin-parallel-resmlp-plan.md).

ParallelResCNN is therefore kept as the reference implementation of
the grouped-conv batching pattern (groups = n_bins as per-bin
parallelism on the GPU), not as a production head. The production
conv head is emulator_designs.ResCNN at the package root: bins as
channels, a gated zero-initialized correction, and the fixed
full-whitened <-> diagonal basis buffers. This class predates that
redesign and keeps the original straight-pipeline shape (the conv is
an inner layer, not a gated correction), so results stay comparable
with the notebook experiments it came from.

PS: bin = one (redshift-pair, xi+/xi-) block of the data vector, a
contiguous run of theta angles; grouped conv = a Conv1d whose
`groups` option splits its channels into independent
sub-convolutions (no weights connect different groups); padded
layout = every bin stored at the longest bin's kept length, so one
rectangular tensor holds all bins; whitened = transformed so each
component has unit variance (the network's working basis).
"""

import torch.nn as nn

from ..activations import activation_fcn
from ..emulator_designs_building_blocks import Affine, ResBlock
from .emulator_designs_building_blocks import GroupedCNNBlock


class ParallelResCNN(nn.Module):
  """
  Shared ResMLP trunk + a per-bin grouped 1D-conv stage: each
  tomographic bin's theta curve is refined independently, so no
  kernel smooths across the bin-boundary jumps a global conv would
  blur.

  The conv works on a padded per-bin layout: n_bins segments of
  length max_bin (the longest bin's kept theta count), giving the
  grouped conv one uniform per-group length. The padding slots
  (max_bin minus each real bin size) are absorbed by the
  surrounding linear layers; the final Linear maps the padded
  representation back to the real data-vector length.

  Forward shape flow:

      x  (B, input_dim)          whitened parameters
         │  Linear               input_dim -> W
         ▼
      h  (B, W)                  trunk features
         │  n_blocks x ResBlock  width-preserving residual blocks
         ▼
      h  (B, W)
         │  Linear               expand, W -> n_bins*max_bin
         ▼
      p  (B, n_bins*max_bin)     padded per-bin layout
         │  GroupedCNNBlock      per-bin conv; no cross-bin mixing
         ▼
      p  (B, n_bins*max_bin)
         │  Linear               project, n_bins*max_bin -> n_keep
         ▼
      y  (B, n_keep)             diagonal-whitened data vector
         │  Affine               per-element learnable gain + bias
         ▼
      out (B, n_keep)

      (legend: B = batch rows; input_dim = number of cosmological
       parameters; W = int_dim_res, the trunk width; n_bins =
       number of tomographic (pair, xi+/xi-) bins; max_bin = the
       longest bin's kept theta count -- LSST-Y1 example: 30 bins
       of up to 26 angles; n_keep = output_dim, the kept
       data-vector length the network emulates.)

  The targets must stay in theta order within each bin for the conv
  to see the angular axis, so this model requires the
  DiagonalGeometry whitening (scale-only, no rotation) -- a
  full-eigenbasis whitening would scramble the axis the kernel
  slides along. Unlike the production ResCNN, there are no
  basis-change buffers here: the whole pipeline lives in the
  diagonal basis.

  Arguments:
    input_dim   = number of cosmological parameters.
    output_dim  = data-vector length to emulate (= n_keep).
    int_dim_res = internal width of the residual trunk.
    geom        = output geometry carrying bin_sizes (attached by
                  build_shear_angle_map); only the per-bin split is
                  read from it. Must be a DiagonalGeometry (see
                  above).
    kernel_size = conv kernel width (odd, same-padded).
    channels    = conv filters per bin inside GroupedCNNBlock.
    n_blocks    = residual blocks in the trunk.
    block_opts  = ResBlock options (None -> {}); its "act" is also
                  handed to the grouped conv, so conv stage and
                  trunk share one activation family. Defaults to
                  activation_fcn (the paper's H) when block_opts
                  sets no "act".

  needs_geom / needs_bins are capability flags EmulatorExperiment
  reads: geom injected at build, and build_shear_angle_map run on
  the data geometry before the model is built.
  """
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, channels=16, n_blocks=3,
               block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}
    n_bins  = len(geom.bin_sizes)
    max_bin = max(geom.bin_sizes)
    cnn_dim = n_bins * max_bin           # padded per-bin layout

    # one activation family for trunk and conv stage: the ResBlock
    # "act" option (when set) also drives the grouped conv, so a
    # YAML-chosen activation is not silently ignored by the head.
    act = block_opts.get("act", activation_fcn)

    layers = []
    layers.append(nn.Linear(in_features=input_dim, out_features=int_dim_res))

    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))

    # expand to the padded per-bin layout.
    layers.append(nn.Linear(in_features=int_dim_res, out_features=cnn_dim))

    # per-bin (grouped) convolution -- no cross-bin mixing.
    layers.append(GroupedCNNBlock(n_groups=n_bins,
                                  group_len=max_bin,
                                  kernel_size=kernel_size,
                                  channels=channels,
                                  act=act))
    # project the padded layout to the real data vector.
    layers.append(nn.Linear(in_features=cnn_dim, out_features=output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
