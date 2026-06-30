"""Per-bin parallel emulator models."""

import torch
import torch.nn as nn

from ..emulator_designs_building_blocks import Affine, ResBlock
from .emulator_designs_building_blocks import (
  GroupedLinear, GroupedAffine, GroupedResBlock, GroupedCNNBlock)


class ParallelResMLP(nn.Module):
  """
  Per-bin emulator, batched: G = len(geom.bin_sizes) independent
  ResMLP heads (one per (xi+/-, source pair) bin) computed in
  parallel via grouped layers. Every head sees the full input and
  predicts its bin; the heads are stacked on a group axis so all G
  run in one batched matmul instead of a Python loop.

  Output widths differ per bin, so every head is built to the padded
  width max(bin_sizes); the forward slices each head's real bin_size
  and concats. The slice/concat is pure memory movement (no matmul),
  costing nothing.

  Same parameters and math as the loop version, laid out so the GPU
  does the heads simultaneously.

  Arguments:
    input_dim   = number of model inputs.
    output_dim  = full emulated length (n_keep) = sum(bin_sizes).
    int_dim_res = internal width per head.
    geom        = geometry carrying geom.bin_sizes (the per-bin
                  kept counts, in dest_idx order).
    n_blocks    = grouped residual blocks per head.
    block_opts  = {"n_layers": ...} for each GroupedResBlock.
  """
  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               n_blocks=3, block_opts=None):
    super().__init__()
    sizes = list(geom.bin_sizes)
    assert sum(sizes) == output_dim, (
      f"sum(bin_sizes)={sum(sizes)} != output_dim="
      f"{output_dim}; run build_shear_angle_map(geom)")
    if block_opts is None:
      block_opts = {}
    self.sizes   = sizes               # real per-head widths
    self.n_heads = len(sizes)
    self.max_bin = max(sizes)          # padded head width
    G = self.n_heads
    n_layers = block_opts.get("n_layers", 2)

    # input projection: input_dim -> int_dim_res, all heads.
    self.in_proj  = GroupedLinear(G, input_dim, int_dim_res)
    # the residual trunk (identical shape across heads).
    blocks = []
    for _ in range(n_blocks):
      blocks.append(GroupedResBlock(G, int_dim_res, n_layers=n_layers))
    self.blocks = nn.ModuleList(blocks)
    # output projection to the padded width, then a final affine.
    self.out_proj = GroupedLinear(G, int_dim_res, self.max_bin)
    self.out_aff  = GroupedAffine(G)

  def forward(self, x):                 # x: (B, input_dim)
    G = self.n_heads
    # Broadcast the same input to all G heads. unsqueeze(0) inserts
    # a size-1 axis: (B, in) -> (1, B, in); expand(G, -1, -1)
    # stretches it to length G (each -1 leaves an axis as is) ->
    # (G, B, in). expand is a view (stride 0 on the new axis), so the
    # input is not copied G times; GroupedLinear's einsum broadcasts.
    h = x.unsqueeze(0).expand(G, -1, -1)
    h = self.in_proj(h)                 # (G, B, D)
    for blk in self.blocks:
      h = blk(h)                        # (G, B, D)
    h = self.out_aff(self.out_proj(h))  # (G, B, max_bin)
    # Each head owns its first sizes[g] outputs (the rest padding).
    # Slice those and concat in head order (= bin = dest_idx order)
    # -> (B, n_keep). The only per-head loop, just slicing -- no
    # matmul.
    slices = []
    for g in range(G):
      slices.append(h[g, :, :self.sizes[g]])
    return torch.cat(slices, dim=-1)


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
