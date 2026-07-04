"""Standard emulator models (ResMLP, ResCNN, ResTRF).

Full networks mapping whitened parameters to the whitened data vector.
ResMLP is the baseline: input projection, a stack of identical ResBlocks,
output projection, final Affine. ResCNN and ResTRF add a correction
appendix on a ResMLP trunk: the trunk predicts in the full
(cov-eigenbasis) whitening, fixed buffers map its output into theta order,
a structured head corrects it there -- a 1D conv along the angular axis
(ResCNN), or a transformer whose tokens are the tomographic bins
(ResTRF) -- and a learnable gate adds the correction back, so swapping
the architecture changes only the model. Per-bin conv variants live in
parallel/.

Whitened = rotated into the covariance eigenbasis and scaled to unit
variance, leaving the components decorrelated and equally hard to fit;
done by the geometry classes (geometries_parameter / geometries_output).
"""

import torch
import torch.nn as nn

from .activations import activation_fcn
from .emulator_designs_building_blocks import (
  Affine, ResBlock, BinLinear, TRFBlock)


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
  ResMLP trunk + a bins-as-channels 1D-CNN correction appendix. The
  trunk is identical to the standalone ResMLP and predicts in the
  full (cov-eigenbasis) whitened basis, so its loss stays the
  well-conditioned chi2 = ||pred - target||^2 (identity Hessian).

  The CNN is an additive correction in the diagonal view (theta
  order, per-element /sigma; the full-whitened basis scrambles the
  angular order, so a conv there has no locality). The theta-order
  dv splits into its (xi+/-, source-pair) tomographic bins, and the
  bins become the conv's CHANNELS: one Conv1d(n_bins -> n_bins,
  kernel_size) slides a single kernel along theta over the whole
  data vector at once. At every theta position each output bin
  reads a kernel_size-wide window of ALL bins -- theta-local AND
  cross-bin (the bins share one angular grid, so channel mixing
  couples different bins at like angular scales, up to per-bin mask
  offsets). No channel expansion: the head's tensors never grow
  beyond the (padded) dv size, so the bandwidth wall the old
  expand-to-C-filters head hit cannot occur by construction. Each
  block is one conv + one activation (the nonlinearity between
  stacked blocks -- without it two convs fold into a single
  kernel); the only head hyperparameters are kernel_size and
  n_blocks_cnn.

  Bins differ in kept length, so each is padded to max_bin inside a
  fixed index buffer (pad_idx scatters the n_keep theta-order
  entries into the padded (n_bins, max_bin) layout and gathers the
  corrections back; pad slots stay zero). The bin split comes from
  geom.bin_sizes (attached by build_shear_angle_map; the needs_bins
  flag makes EmulatorExperiment run it).

  The head starts as an exact identity: the last conv is
  zero-initialized, so corr = 0 and the model IS its trunk at epoch
  1 (the zero-init-residual-branch start; gradients reach the
  zeroed conv through the nonzero gate at step 1).

  The two basis-change maps are precomputed and stored as fixed
  buffers, named for the bases: f = full-whitened (the eigenbasis
  the trunk predicts in), d = diagonal (theta order, each element
  scaled by its marginal sigma). Subscripts read in multiply order:
  y_full @ W_fd goes f -> d, correction @ W_df goes d -> f (W_df =
  W_fd inverse). Buffers, not live geometry calls in forward, stay
  safe under torch.compile CUDA graphs.

  Target and loss use the full-whitening DataVectorGeometry, as the
  standalone ResMLP, so swapping ResMLP -> ResCNN changes the model
  only, not the whitening (no confound).

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = full-whitening DataVectorGeometry carrying
                   bin_sizes; its evecs / sqrt_ev define the basis
                   buffers.
    kernel_size  = conv kernel width (odd, same-padded).
    n_blocks     = residual blocks in the trunk.
    n_blocks_cnn = stacked conv+activation correction blocks.
    gate_init    = initial value of the scalar scaling the
                   correction. Small (default 0.1) to start near the
                   pure ResMLP; not 0 -- a 0 gate strands the CNN
                   with no gradient, so it never learns.
    block_opts   = ResBlock options (None -> {}); its "act" is also
                   handed to the CNN head, so head and trunk share
                   one activation family. Defaults to activation_fcn
                   (the paper's H) when block_opts sets no "act".

  needs_geom / needs_bins are capability flags EmulatorExperiment
  reads: geom injected (basis buffers + bin sizes), compile_mode
  defaulted to "default", and build_shear_angle_map run on the data
  geometry before the model is built.
  """
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               kernel_size=11, n_blocks=3, n_blocks_cnn=1,
               gate_init=0.1, block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}
    assert kernel_size % 2 == 1, (
      "kernel_size must be odd so same-padding keeps the length")
    assert hasattr(geom, "bin_sizes"), (
      "ResCNN needs geom.bin_sizes -- run build_shear_angle_map"
      "(geom) first (EmulatorExperiment does this for models with "
      "the needs_bins flag)")

    # ResMLP main path: standalone ResMLP layer stack, output in the
    # full-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # the bin split: per-bin kept counts, contiguous in theta order,
    # and the fixed scatter/gather index into the padded layout (bin
    # g's j-th entry at g*max_bin + j; see the class docstring).
    sizes = []
    for s in geom.bin_sizes:
      sizes.append(int(s))
    self.n_bins  = len(sizes)
    self.max_bin = max(sizes)
    pos = []
    for g in range(self.n_bins):
      for j in range(sizes[g]):
        pos.append(g * self.max_bin + j)
    self.register_buffer(
      "pad_idx", torch.tensor(pos, dtype=torch.long))

    # the head: n_blocks_cnn x (one bins-as-channels conv + one
    # activation). The activation is the run's (block_opts["act"],
    # the --activation choice injected by EmulatorExperiment),
    # falling back to activation_fcn (the paper's H); act(max_bin)
    # gives per-position parameters, broadcast over the bin axis.
    cnn_act = block_opts.get("act", activation_fcn)
    pad = (kernel_size - 1) // 2
    convs, acts = [], []
    for _ in range(n_blocks_cnn):
      convs.append(nn.Conv1d(in_channels=self.n_bins,
                             out_channels=self.n_bins,
                             kernel_size=kernel_size,
                             padding=pad))
      acts.append(cnn_act(self.max_bin))
    self.convs = nn.ModuleList(convs)
    self.acts  = nn.ModuleList(acts)

    # zero-init the LAST conv: corr = 0 at init (the activation maps
    # 0 -> 0), so the model starts as its trunk exactly; the zeroed
    # conv gets real gradients through the nonzero gate at step 1,
    # earlier blocks wake one step later.
    nn.init.zeros_(self.convs[-1].weight)
    nn.init.zeros_(self.convs[-1].bias)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # Frozen basis-change buffers (move with .to(device), not
    # trained). x @ W_fd maps f -> d, x @ W_df maps d -> f. sigma =
    # per-element scale sqrt(diag cov); evecs/sqrt_ev the full basis.
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
    y = self.mlp(x)                   # (B, n_keep)
    h = y @ self.W_fd                 # f -> d, theta order
    # scatter into the padded per-bin layout: each bin one channel.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    c = padded.view(-1, self.n_bins, self.max_bin)
    n = len(self.convs)
    for i in range(n):
      c = self.acts[i](self.convs[i](c))   # cross-bin, theta-local
    # gather the real entries back out of the padding, return to the
    # full-whitened basis, add through the gate.
    corr = c.reshape(-1, self.n_bins * self.max_bin)[:, self.pad_idx]
    return y + self.gate * (corr @ self.W_df)


class ResTRF(nn.Module):
  """
  ResMLP trunk + a bin-token transformer correction appendix. The
  trunk is the standalone ResMLP, predicting in the full
  (cov-eigenbasis) whitening; the head maps its output into theta
  order (ResCNN's fixed W_fd / W_df buffers), splits it into the
  (xi+/-, source-pair) tomographic bins, and runs a transformer
  whose TOKENS are those bins: attention shares information across
  bins, then each bin's own MLP stack specializes its correction
  (see TRFBlock for the two deviations from a textbook block). A
  per-bin conv (parallel/) refines within bins but never across
  them; attention is the head for CROSS-bin structure in the
  trunk's residuals.

  The bin split comes from geom.bin_sizes (attached by
  build_shear_angle_map; EmulatorExperiment runs it when the
  needs_bins flag is set). Bins differ in length, so each is padded
  to max_bin inside a fixed index buffer (pad_idx scatters the
  n_keep theta-order entries into the padded (G, max_bin) layout
  and gathers the corrections back; the pad positions stay zero and
  drop at the gather).

  The head starts as an exact identity: the final per-bin
  projection (out) is zero-initialized, so corr = 0 and the model
  IS its trunk at epoch 1 -- the same zero-init-residual-branch
  start as the conv heads, with the same wake-up chain (out's
  weights get real gradients through the nonzero gate at step 1).

  needs_geom / needs_bins are capability flags EmulatorExperiment
  reads: geom injected (basis buffers + bin sizes), compile_mode
  defaulted to "default", and build_shear_angle_map run on the data
  geometry before the model is built.

  Arguments:
    input_dim    = number of cosmological parameters.
    output_dim   = data-vector length to emulate (= n_keep).
    int_dim_res  = internal width of the residual trunk.
    geom         = full-whitening DataVectorGeometry carrying
                   bin_sizes; its evecs / sqrt_ev define the basis
                   buffers.
    int_dim_trf  = token embedding width (divisible by n_heads).
    n_heads      = attention heads per TRFBlock.
    n_blocks     = residual blocks in the trunk.
    n_blocks_trf = stacked transformer blocks.
    n_mlp_blocks = depth of each bin's private MLP stack inside
                   every TRFBlock.
    gate_init    = initial correction-gate scale (small, not 0 --
                   a 0 gate strands the head with no gradient).
    block_opts   = ResBlock options (None -> {}); its "act" also
                   reaches the TRF MLPs, so head and trunk share
                   one activation family.
  """
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               int_dim_trf=32, n_heads=4, n_blocks=4,
               n_blocks_trf=1, n_mlp_blocks=2, gate_init=0.1,
               block_opts=None):
    super().__init__()
    if block_opts is None:
      block_opts = {}
    assert hasattr(geom, "bin_sizes"), (
      "ResTRF needs geom.bin_sizes -- run build_shear_angle_map"
      "(geom) first (EmulatorExperiment does this for models with "
      "the needs_bins flag)")

    # ResMLP main path: standalone ResMLP layer stack, output in the
    # full-whitened basis (well conditioned).
    mlp = [nn.Linear(in_features=input_dim, out_features=int_dim_res)]
    for _ in range(n_blocks):
      mlp.append(ResBlock(int_dim_res, **block_opts))
    mlp.append(nn.Linear(in_features=int_dim_res, out_features=output_dim))
    mlp.append(Affine())
    self.mlp = nn.Sequential(*mlp)

    # the bin split: per-bin kept counts, contiguous in theta order.
    sizes = []
    for s in geom.bin_sizes:
      sizes.append(int(s))
    self.n_bins  = len(sizes)
    self.max_bin = max(sizes)
    # pad_idx maps each kept theta-order position to its slot in the
    # padded (n_bins, max_bin) layout: bin g's j-th entry sits at
    # g*max_bin + j, the tail slots of short bins stay zero. One
    # fixed buffer serves both directions -- scatter to pad, gather
    # to unpad.
    pos = []
    for g in range(self.n_bins):
      for j in range(sizes[g]):
        pos.append(g * self.max_bin + j)
    self.register_buffer(
      "pad_idx", torch.tensor(pos, dtype=torch.long))

    # the head: per-bin embedding -> n_blocks_trf transformer blocks
    # -> per-bin output projection, all bin-unique (BinLinear). The
    # trunk's activation reaches the TRF MLPs too.
    trf_act = block_opts.get("act", activation_fcn)
    self.embed = BinLinear(self.n_bins, self.max_bin, int_dim_trf)
    trf = []
    for _ in range(n_blocks_trf):
      trf.append(TRFBlock(int_dim_trf, n_bins=self.n_bins,
                          n_heads=n_heads,
                          n_mlp_blocks=n_mlp_blocks,
                          act=trf_act))
    self.trf = nn.ModuleList(trf)
    self.out = BinLinear(self.n_bins, int_dim_trf, self.max_bin)

    # zero-init the output projection: corr = 0 at init, so the
    # model starts as its trunk exactly (the zero-init-residual-
    # branch trick; gradients reach the zeroed layer through the
    # nonzero gate, the rest of the head wakes one step later).
    nn.init.zeros_(self.out.weight)
    nn.init.zeros_(self.out.bias)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # Frozen basis-change buffers, exactly ResCNN's: x @ W_fd maps
    # full-whitened -> theta order (/sigma), x @ W_df maps back.
    evecs   = geom.evecs.detach()
    sqrt_ev = geom.sqrt_ev.detach()
    sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
    self.register_buffer(
      "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
    self.register_buffer(
      "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])

  def forward(self, x):
    # trunk prediction in the full-whitened basis (the bulk map).
    y = self.mlp(x)                   # (B, n_keep)
    h = y @ self.W_fd                 # f -> d, theta order
    # scatter into the padded per-bin layout: new_zeros makes the
    # (B, n_bins*max_bin) canvas (pad slots stay 0), the pad_idx
    # assignment places the n_keep real entries.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    # (B, G*max_bin) -> (B, G, max_bin): each bin one token row.
    t = padded.view(-1, self.n_bins, self.max_bin)
    t = self.embed(t)                 # (B, G, int_dim_trf)
    for blk in self.trf:
      t = blk(t)                      # cross-bin attention + MLPs
    t = self.out(t)                   # (B, G, max_bin)
    # gather the real entries back out of the padded layout
    # (dropping the pad slots), then return to the full-whitened
    # basis and add through the gate.
    corr = t.reshape(-1, self.n_bins * self.max_bin)[:, self.pad_idx]
    return y + self.gate * (corr @ self.W_df)
