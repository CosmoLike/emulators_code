"""Per-bin CNN emulator model.

Only the CNN correction gets a per-bin variant: a grouped conv refines
each tomographic bin independently (no smoothing across bin-boundary
jumps), on a shared ResMLP trunk. There is deliberately NO per-bin
ResMLP: splitting the trunk per bin throws away the cosmology-map
parameter sharing and re-learns the hard shared map once per bin --
the tested-and-rejected design (it came out worse than one ResMLP at
matched parameters).
"""

import torch.nn as nn

from ..emulator_designs_building_blocks import Affine, ResBlock
from .emulator_designs_building_blocks import GroupedCNNBlock


class ParallelResCNN(nn.Module):
  """
  ResMLP trunk + a per-bin 1D-CNN correction head: like ResCNN, but
  the conv is grouped so each tomographic bin is refined
  independently (no smoothing across the bin-boundary jumps).

  The CNN works on a padded per-bin layout -- n_bins segments of
  length max_bin (the largest bin's kept count) -- giving the
  grouped conv a uniform per-group length. The padding (max_bin
  minus each real bin size) is absorbed by the surrounding linears;
  the final Linear maps the padded n_bins*max_bin representation to
  the real data-vector length.

  Needs geom.bin_sizes (run build_shear_angle_map(geom) first) and
  a DiagonalGeometry (theta order kept within each bin).

  Arguments: as ResCNN, plus geom (for the per-bin split).
  """
  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, channels=16, n_blocks=3,
               block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}
    n_bins  = len(geom.bin_sizes)
    max_bin = max(geom.bin_sizes)
    cnn_dim = n_bins * max_bin           # padded per-bin layout

    layers = []
    layers.append(nn.Linear(in_features=input_dim, out_features=int_dim_res))

    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))

    # expand to the padded per-bin layout.
    layers.append(nn.Linear(in_features=int_dim_res, out_features=cnn_dim))

    # per-bin (grouped) convolution -- no cross-bin mixing.
    layers.append(GroupedCNNBlock(n_bins, max_bin,
                                  kernel_size=kernel_size,
                                  channels=channels))
    # project the padded layout to the real data vector.
    layers.append(nn.Linear(in_features=cnn_dim, out_features=output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)
