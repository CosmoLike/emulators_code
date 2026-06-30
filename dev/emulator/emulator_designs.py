"""Standard emulator models (ResMLP, ResCNN).

Full networks mapping whitened parameters to the whitened data vector.
ResMLP is the baseline: input projection, a stack of identical ResBlocks,
output projection, final Affine. ResCNN adds a 1D-CNN correction appendix
on a ResMLP trunk: the trunk predicts in the full (cov-eigenbasis)
whitening, fixed buffers map its output into theta order for a conv to
correct structure along the angular axis, and a learnable gate adds the
correction back -- so swapping ResMLP -> ResCNN changes only the model.
Per-bin parallel variants live in parallel/.

Whitened = rotated into the covariance eigenbasis and scaled to unit
variance, leaving the components decorrelated and equally hard to fit;
done by the geometry classes (geometries_parameter / geometries_output).
"""

import torch
import torch.nn as nn

from .activations import activation_fcn
from .emulator_designs_building_blocks import (
  Affine, ResBlock, CNNBlock)


class ResMLP(nn.Module):
  """
  Full emulator: input projection, a stack of identical residual
  blocks, output projection, final learnable affine.

  Arguments:
    input_dim   = number of cosmological parameters
    output_dim  = length of the data vector
    int_dim_res = internal (residual) width
    n_blocks    = number of residual blocks
    block_opts  = dict of ResBlock options (n_layers,
                   norm, act), the same for every block

  block_opts defaults to None, not {}: a default argument is
  created once and shared across calls, so a mutable dict would
  leak between them. All blocks share one configuration, capping
  the hyperparameter count.
  """
  def __init__(self, 
               input_dim, 
               output_dim, 
               int_dim_res,
               n_blocks=3, 
               block_opts=None):
    super().__init__()
    
    # Default to {} (not in the signature: a mutable default is
    # created once and would leak between calls).
    if block_opts is None:
      block_opts = {}
    layers = []

    # param dim -> internal width
    layers.append(nn.Linear(in_features=input_dim, out_features=int_dim_res))

    # n_blocks identical residual blocks at the internal width;
    # **block_opts unpacks the dict into keyword args per ResBlock.
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))

    # internal width -> data-vector dim
    layers.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))

    # final learnable scale and shift
    layers.append(Affine())

    # Sequential registers every module, so the temporary list is fine.
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    return self.model(x)


class ResCNN(nn.Module):
  """
  ResMLP trunk + a 1D-CNN correction appendix. The trunk is
  identical to the standalone ResMLP and predicts in the full
  (cov-eigenbasis) whitened basis, so its loss stays the
  well-conditioned chi2 = ||pred - target||^2 (identity Hessian).

  The CNN is an additive correction in the diagonal view (theta
  order, per-element /sigma). A 1D conv slides one shared kernel
  along the length axis to exploit adjacency, useful only when
  neighbouring entries are neighbouring theta. The full-whitened
  basis mixes all thetas per component and scrambles the angular
  order, so a conv there has no locality. The diagonal view keeps
  theta order (per-element /sigma, no rotation), so the conv sees
  the real angular axis and corrects theta-local structure (the
  smooth shape and oscillations of xi across neighbouring bins,
  left by the trunk as correlated residuals along theta). forward
  maps the trunk output into that view, the conv corrects it, then
  maps the correction back and adds it through a learnable gate --
  so the bulk map keeps the eigenbasis conditioning (the ResMLP is
  permutation invariant, needing no theta order) and only the small
  correction pays the theta-order conditioning cost.

  The two basis-change maps are precomputed and stored as fixed
  buffers, named for the bases: f = full-whitened (the eigenbasis
  the trunk predicts in), d = diagonal (theta order, each element
  scaled by its marginal sigma, the DiagonalGeometry view).
  Subscripts read in multiply order: y_full @ W_fd goes f -> d,
  correction @ W_df goes d -> f (W_df = W_fd inverse). Buffers, not
  live geometry calls in forward, stay safe under torch.compile
  reduce-overhead / CUDA graphs: a tensor lazily built in forward
  gets captured in the graph's static pool and overwritten next
  run.

  Target and loss use the full-whitening DataVectorGeometry, as the
  standalone ResMLP, so swapping ResMLP -> ResCNN changes the model
  only, not the whitening (no confound).

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = full-whitening DataVectorGeometry; its evecs /
                   sqrt_ev (and derived sigma) define the two
                   basis-change buffers.
    kernel_size  = CNN kernel width (odd); forwarded to CNNBlock.
    channels     = CNN filter count; forwarded to CNNBlock.
    n_blocks     = residual blocks in the trunk.
    n_blocks_cnn = stacked CNN correction blocks (default 1).
    gate_init    = initial value of the scalar scaling the
                   correction. Small (default 0.1) to start near the
                   pure ResMLP; not 0 -- a 0 gate strands the CNN
                   with no gradient, so it never learns.
    block_opts   = ResBlock options (None -> {}); its "act" is also
                   handed to the CNN head, so head and trunk share
                   one activation family. Defaults to activation_fcn
                   (the paper's H) when block_opts sets no "act".
  """
  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, channels=16, n_blocks=3,
               n_blocks_cnn=1, gate_init=0.1, block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}

    # ResMLP main path: standalone ResMLP layer stack, output in the
    # full-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # CNN appendix: axis-aware blocks in theta order. The head takes
    # the trunk's activation: block_opts["act"] carries the run's
    # chosen activation (the --activation flag, injected by
    # EmulatorExperiment), falling back to activation_fcn (the
    # paper's H, the ResBlock default). Without this, a non-default
    # activation would reach the trunk only and the head stay on H.
    cnn_act = block_opts.get("act", activation_fcn)
    cnn = []
    for _ in range(n_blocks_cnn):
      cnn.append(CNNBlock(output_dim, kernel_size=kernel_size,
                          channels=channels, act=cnn_act))
    self.cnn = nn.ModuleList(cnn)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # Frozen basis-change buffers (move with .to(device), not
    # trained). x @ W_fd maps f -> d, x @ W_df maps d -> f. sigma =
    # DiagonalGeometry per-element scale sqrt(diag cov);
    # evecs/sqrt_ev give the full basis.
    #   full-whitened y -> physical -> theta order (/sigma):
    #     W_fd = diag(sqrt_ev) evecs.T diag(1/sigma)
    #   theta-order correction -> physical -> full-whitened:
    #     W_df = diag(sigma) evecs diag(1/sqrt_ev)  (= W_fd^{-1})
    evecs   = geom.evecs.detach()
    sqrt_ev = geom.sqrt_ev.detach()
    sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
    self.register_buffer(
      "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
    self.register_buffer(
      "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])

  def forward(self, x):
    # trunk prediction in the full-whitened basis (the bulk map).
    y = self.mlp(x)                   # (B, out_dim)
    h = y @ self.W_fd                 # f -> d, conv sees angular axis
    for blk in self.cnn:
      h = blk(h)                      # axis-aware correction
    # correction back to full-whitened (d -> f), gated, added.
    return y + self.gate * (h @ self.W_df)
