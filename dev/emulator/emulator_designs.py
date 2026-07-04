"""Standard emulator models (ResMLP, ResCNN, ResTRF).

Full networks mapping whitened cosmological parameters to the whitened
data vector. Where this file sits in the training pipeline:

  cosmological parameters
     │   geometries_parameter.py  center, rotate, unit-scale (whiten in)
     ▼
  whitened inputs
     │   emulator_designs.py      ResMLP, ResCNN, or ResTRF (this file)
     ▼
  whitened data vector
     │   geometries_output.py     un-whiten + scatter to full length
     ▼
  physical residual vs truth
     │   loss_functions.py        contract with the inverse covariance
     ▼
  chi2 = r^T Cinv r

ResMLP is the baseline: input projection, a stack of identical
ResBlocks, output projection, final Affine. ResCNN and ResTRF add a
correction appendix on a ResMLP trunk: the trunk predicts in the full
(cov-eigenbasis) whitening, fixed buffers map its output into theta
order, a structured head corrects it there -- a 1D conv along the
angular axis (ResCNN), or a transformer whose tokens are the
tomographic bins (ResTRF) -- and a learnable gate adds the correction
back, so swapping the architecture changes only the model. Per-bin
conv variants live in parallel/.

Whitened = rotated into the covariance eigenbasis and scaled to unit
variance, leaving the components decorrelated and equally hard to fit;
done by the geometry classes (geometries_parameter /
geometries_output).
"""

import torch
import torch.nn as nn

from .activations import activation_fcn
from .emulator_designs_building_blocks import (
  Affine, ResBlock, TRFBlock, FiLMGenerator, rescale_kernel_size)


class ResMLP(nn.Module):
  """
  The baseline emulator: input projection, a stack of identical
  residual blocks, output projection, final learnable affine.

    x  (B, input_dim)     whitened cosmological parameters
       │  Linear: input_dim -> int_dim_res
       ▼
       │  n_blocks x ResBlock, all at width int_dim_res
       ▼
       │  Linear: int_dim_res -> output_dim;  Affine
       ▼
    dv (B, output_dim)    whitened data vector

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
  The forward pass, shapes and all:

    x  (B, input_dim)           whitened parameters
       │  self.mlp              the ResMLP trunk
       ▼
    y  (B, n_keep)              full-whitened dv (also the skip)
       │  @ W_fd                f -> d: theta order, /sigma
       ▼
    h  (B, n_keep)
       │  pad_idx scatter       pad slots stay zero
       ▼
    c  (B, n_bins, max_bin)     tomographic bins = conv channels
       │  n_blocks_cnn x [Conv1d + act]
       ▼
       │  pad_idx gather        drop the pad slots
       ▼
    corr (B, n_keep)
       │  @ W_df                d -> f, back to full whitening
       ▼
    out = y + gate * corr       corr = 0 at init (identity start)

  (legend: B = batch rows; n_keep = kept data-vector length, the
  unmasked entries the model emulates = output_dim; n_bins = number
  of tomographic (xi+/-, source-pair) bins; max_bin = the longest
  bin's kept theta count = the padded bin width; f / d = the
  full-whitened / diagonal-theta bases, see the W_fd buffers below.)

  The CNN is an additive correction in the diagonal view (theta
  order, per-element /sigma; the full-whitened basis scrambles the
  angular order, so a conv there has no locality). The theta-order
  dv splits into its (xi+/-, source-pair) tomographic bins, and the
  bins become the conv's channels: one Conv1d(n_bins -> n_bins,
  kernel_size) slides a single kernel along theta over the whole
  data vector at once. At every theta position each output bin
  reads a kernel_size-wide window of all the bins -- theta-local
  and cross-bin (the bins share one angular grid, so channel mixing
  couples different bins at like angular scales, up to per-bin mask
  offsets). No channel expansion: the head's tensors never grow
  beyond the (padded) dv size, so the bandwidth wall the old
  expand-to-C-filters head hit cannot occur by construction. Each
  block is one conv + one activation (the nonlinearity between
  stacked blocks -- without it two convs fold into a single
  kernel). The head hyperparameters are kernel_size (+ the
  rescale_kernel flag), n_blocks_cnn, groups, separable, film, and
  gate_init.

  groups restricts that channel mixing along the one physical cut
  the channel order offers. The channels are the bins in dv order
  -- the xi+ pairs first, then the xi- pairs (cosmolike's layout,
  reconstructed by build_shear_angle_map) -- and a grouped conv
  splits the channels into `groups` consecutive blocks that never
  mix:

    channels:   xi+ pair 1 .. P │ xi- pair 1 .. P
                                │
    groups=1:   no cut -- every output bin reads every bin (the
                default: full cross-bin mixing)
    groups=2:   cut at the │ -- xi+ never mixes with xi-, but
                bins still mix freely within their branch

  (legend: P = n_bins/2 source pairs per xi branch; per-block conv
  parameters = n_bins * (n_bins/groups) * kernel_size + n_bins, so
  the cut also halves the head's conv weights. The boundary is
  validated against geom.pm_kept at build: bin_sizes drops
  fully-masked bins, so a wholly-masked bin on one branch would
  silently shift the cut -- that fails loudly instead.)

  separable factors the remaining sum's two jobs -- smoothing
  along theta and mixing channels -- into two cheaper layers per
  block:

    c  (B, C, max_bin)
       │  depthwise Conv1d(C -> C, k, groups=C): each channel its
       │  own k-tap theta filter, no mixing        C*k weights
       ▼
       │  pointwise Conv1d(C -> C, 1, groups=groups): mixes the
       │  channels at each theta position          C*(C/groups)
       ▼
    act  (one activation per block, after the pointwise)

  versus the plain block's joint C*(C/groups)*k. No activation
  sits between the two layers, so the pair composes into a single
  constrained conv -- weights w[o, c, t] = pointwise[o, c] *
  depthwise[c, t] -- i.e. a low-rank factorization of the plain
  block's sum, not a different operation. The assumption the
  factorization adds: the theta-smoothing profile a channel needs
  does not depend on which channel it mixes into (plausible for
  covariance-driven leakage at like angular scales; the standard
  depthwise-separable trade). The zero-init identity start moves
  to the last block's pointwise layer.

  (legend: C = n_bins, the channels; k = the per-block kernel
  width after any rescale; B = batch rows.)

  Bins differ in kept length, so each is padded to max_bin (the
  longest bin's kept theta count) inside a fixed index buffer
  (pad_idx scatters the n_keep theta-order entries into the padded
  (n_bins, max_bin) layout and gathers the corrections back; pad
  slots stay zero). The bin split comes from geom.bin_sizes
  (attached by build_shear_angle_map; the needs_bins flag makes
  EmulatorExperiment run it).

  The head starts as an exact identity: the last conv is
  zero-initialized, so corr = 0 and the model equals its trunk at
  epoch 1 (the zero-init-residual-branch start; gradients reach the
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
    kernel_size  = conv kernel width (odd, same-padded), tuned as
                   if the head had ONE block. With rescale_kernel
                   it states the target receptive field; without,
                   it is used verbatim for every block.
    rescale_kernel = False (default): every block uses kernel_size
                   as given. True: the per-block kernel shrinks
                   with depth so the n_blocks_cnn-deep stack keeps
                   a single kernel_size-wide block's view --
                   receptive field n*(k-1)+1 >= kernel_size, see
                   rescale_kernel_size. Depth then buys
                   nonlinearity at a fixed total view (and
                   near-flat head parameters) instead of
                   over-growing the receptive field. The resolved
                   width is stored as self.kernel_size.
    groups       = channel-mixing restriction: 1 (default, dense
                   mixing) or 2 (xi+ never mixes with xi-). See
                   the groups paragraph and graph in the class
                   docstring; other values error, and the xi
                   boundary is validated against geom.pm_kept at
                   build.
    separable    = False (default): one joint conv per block.
                   True: factor each block into a depthwise
                   theta filter + a pointwise channel mix (a
                   low-rank factorization of the same sum, ~k/2
                   times fewer weights). See the separable
                   paragraph and graph in the class docstring.
    film         = False (default): the head is one fixed map,
                   blind to the cosmology. True: re-inject the
                   parameters into every block as a per-channel
                   affine, conv -> gamma(x)*c + beta(x) -> act,
                   with one identity-initialized FiLMGenerator
                   per block (Linear(input_dim, 2*n_bins) --
                   ~2*n_bins*input_dim parameters each, see the
                   generator's docstring and
                   notes/film-conditioning.md). The gate stays as
                   the outer valve; FiLM modulates inside the
                   blocks, letting the cosmology choose which
                   bins to amplify and by how much.
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
               kernel_size=11, rescale_kernel=False, groups=1,
               separable=False, film=False, n_blocks=3,
               n_blocks_cnn=1, gate_init=0.1, block_opts=None):
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
    # rescale_kernel: kernel_size was tuned for a single block, so
    # shrink the per-block kernel with depth to keep that block's
    # view -- receptive field n*(k-1)+1 >= kernel_size, see
    # rescale_kernel_size -- instead of over-growing it.
    if rescale_kernel:
      kernel_size = rescale_kernel_size(kernel_size=kernel_size,
                                        n_blocks_cnn=n_blocks_cnn)
    # the resolved per-block width, inspectable after a rescale.
    self.kernel_size = int(kernel_size)

    # groups: only the xi-branch cut is a physical channel
    # boundary here (see the docstring). Validate the layout
    # against the geometry rather than assume it: each bin is a
    # contiguous run of kept elements sharing one pm (0 = xi+,
    # 1 = xi-), so the run starts give the per-bin branch; the
    # first half of the bins must all be xi+ and the second half
    # xi- (a fully-masked bin on one branch would silently shift
    # the boundary -- fail loudly instead).
    assert groups in (1, 2), (
      "ResCNN groups must be 1 (dense) or 2 (xi+ never mixes "
      "with xi-); the channels are single bins, so no other cut "
      "has a physical meaning")
    if groups == 2:
      assert hasattr(geom, "pm_kept") and self.n_bins % 2 == 0, (
        "groups=2 needs geom.pm_kept (build_shear_angle_map) and "
        "an even bin count")
      pm_bins = []
      start = 0
      for s in sizes:
        pm_bins.append(int(geom.pm_kept[start]))
        start += s
      half = self.n_bins // 2
      assert (all(pm == 0 for pm in pm_bins[:half])
              and all(pm == 1 for pm in pm_bins[half:])), (
        "groups=2 needs the first half of the bins to be xi+ and "
        "the second half xi- (a fully-masked bin on one branch "
        f"breaks the split); per-bin branches here: {pm_bins}")

    pad = (kernel_size - 1) // 2
    convs, acts = [], []
    for _ in range(n_blocks_cnn):
      if separable:
        # depthwise-separable factorization (see the class
        # docstring): a per-channel k-tap theta filter (groups =
        # n_bins: no mixing), then a pointwise 1x1 channel mix
        # honoring `groups`. No activation between the two -- the
        # pair is a low-rank factorization of the plain block's
        # conv, and the block's one activation follows as usual.
        # Sequential keeps forward unchanged (convs[i] is callable
        # either way).
        convs.append(nn.Sequential(
          nn.Conv1d(in_channels=self.n_bins,
                    out_channels=self.n_bins,
                    kernel_size=kernel_size,
                    padding=pad,
                    groups=self.n_bins),
          nn.Conv1d(in_channels=self.n_bins,
                    out_channels=self.n_bins,
                    kernel_size=1,
                    groups=groups)))
      else:
        convs.append(nn.Conv1d(in_channels=self.n_bins,
                               out_channels=self.n_bins,
                               kernel_size=kernel_size,
                               padding=pad,
                               groups=groups))
      acts.append(cnn_act(self.max_bin))
    self.convs = nn.ModuleList(convs)
    self.acts  = nn.ModuleList(acts)

    # zero-init the last mixing layer: corr = 0 at init (the
    # activation maps 0 -> 0), so the model starts as its trunk
    # exactly; the zeroed layer gets real gradients through the
    # nonzero gate at step 1, earlier layers wake one step later.
    # In a separable block the zero lives on the pointwise (second)
    # conv; the depthwise filter keeps its init (zeroing both would
    # zero the pointwise's input and stall its wake-up).
    last = self.convs[-1][1] if separable else self.convs[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

    # FiLM (film=True): one identity-initialized generator per
    # block predicts a per-bin (gamma, beta) from the parameters,
    # re-injecting cosmology the head otherwise never sees. At
    # init gamma = 1 / beta = 0, so the identity start above is
    # untouched. None (default) = the fixed, parameter-blind head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_cnn):
        gens.append(FiLMGenerator(n_cond=input_dim,
                                  n_channels=self.n_bins))
      self.film_gens = nn.ModuleList(gens)

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
    # (reminder: W_fd = f -> d, full-whitened -> diagonal theta
    # order; W_df = d -> f, its inverse. The subscripts read in
    # multiply order: x @ W_fd starts in f and lands in d.)
    h = y @ self.W_fd                 # f -> d, theta order
    # scatter into the padded per-bin layout: each bin one channel.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    c = padded.view(-1, self.n_bins, self.max_bin)
    n = len(self.convs)
    for i in range(n):
      c = self.convs[i](c)                 # cross-bin, theta-local
      if self.film_gens is not None:
        # FiLM re-injection: a per-bin affine whose coefficients
        # depend on the parameters (identity at init). unsqueeze
        # broadcasts (B, n_bins) over the theta axis -- the
        # modulation is per channel, never per position.
        gamma, beta = self.film_gens[i](x)
        c = gamma.unsqueeze(-1) * c + beta.unsqueeze(-1)
      c = self.acts[i](c)
    # gather the real entries back out of the padding, return to the
    # full-whitened basis (reminder: @ W_df goes d -> f), add
    # through the gate.
    corr = c.reshape(-1, self.n_bins * self.max_bin)[:, self.pad_idx]
    return y + self.gate * (corr @ self.W_df)


class ResTRF(nn.Module):
  """
  ResMLP trunk + a bin-token transformer correction appendix. The
  trunk is the standalone ResMLP, predicting in the full
  (cov-eigenbasis) whitening; the head maps its output into theta
  order (ResCNN's fixed W_fd / W_df buffers), splits it into the
  (xi+/-, source-pair) tomographic bins, and runs a transformer
  whose tokens are those bins. The forward pass, shapes and all:

    x  (B, input_dim)           whitened parameters
       │  self.mlp              the ResMLP trunk
       ▼
    y  (B, n_keep)              full-whitened dv (also the skip)
       │  @ W_fd                f -> d: theta order, /sigma
       ▼
       │  pad_idx scatter       pad slots stay zero
       ▼
    t0 (B, n_bins, max_bin)     one token per bin, width max_bin
       │  n_blocks_trf x TRFBlock
       │                        cross-bin attention + per-bin MLPs
       ▼
    t  (B, n_bins, max_bin)
       │  corr = t - t0         what the blocks added (0 at init)
       │  pad_idx gather, @ W_df    d -> f, back to full whitening
       ▼
    out = y + gate * corr

  (legend: B = batch rows; n_keep = kept data-vector length, the
  unmasked entries the model emulates = output_dim; n_bins = number
  of tomographic (xi+/-, source-pair) bins; max_bin = the longest
  bin's kept theta count = the padded token width; f / d = the
  full-whitened / diagonal-theta bases, see the W_fd buffers below.)

  Attention shares information across bins, then each bin's own MLP
  stack specializes its correction (see TRFBlock for the two
  deviations from a textbook block). A per-bin conv (parallel/)
  refines within bins but never across them; attention is the head
  for cross-bin structure in the trunk's residuals.

  The bin split comes from geom.bin_sizes (attached by
  build_shear_angle_map; EmulatorExperiment runs it when the
  needs_bins flag is set). Bins differ in length, so each is padded
  to max_bin (the longest bin's kept theta count) inside a fixed
  index buffer (pad_idx scatters the n_keep theta-order entries
  into the padded (G, max_bin) layout and gathers the corrections
  back; the pad positions stay zero and drop at the gather).

  The tokens live at their natural width: max_bin, the padded bin
  length. There is deliberately no embedding layer in and no output
  projection out -- those adapters are what a transformer needs
  when its sequence is synthetic (a flat latent split into tokens,
  as in the published CMB design, where they were the parameter-
  heaviest layers); here the sequence structure is physical, so the
  raw bin segments are the tokens and the blocks' output is already
  in dv layout. The correction is corr = blocks(h) - h: every
  TRFBlock is exactly the identity at init (its branch outputs are
  zero-initialized, see TRFBlock), so corr = 0 and the model equals
  its trunk at epoch 1 -- the same zero-init identity start as the
  conv heads, with the same wake-up chain (the zeroed branch layers
  get real gradients through the nonzero gate at step 1).

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
    n_heads      = attention heads per TRFBlock; must divide the
                   token width max_bin (the LSST-Y1 cosmic-shear
                   run keeps max_bin = 26 theta points per bin,
                   allowing n_heads = 1, 2, or 13; default 2).
    n_blocks     = residual blocks in the trunk.
    n_blocks_trf = stacked transformer blocks.
    n_mlp_blocks = depth of each bin's private MLP stack inside
                   every TRFBlock.
    gate_init    = initial correction-gate scale (small, not 0 --
                   a 0 gate strands the head with no gradient).
    shared_mlp   = False (default): per-bin unique MLPs. True: one
                   MLP shared by every bin -- the textbook block,
                   the ablation isolating the unique-MLP deviation
                   (see TRFBlock's permutation-equivariance caveat).
    film         = False (default): the head is one fixed map,
                   blind to the cosmology. True: after every
                   TRFBlock, modulate the token stream with a
                   per-token affine gamma(x)*t + beta(x) from an
                   identity-initialized FiLMGenerator (one per
                   block; tokens play the conv head's channel
                   role, broadcast over the token width). The
                   stream carries the correction (corr = stream -
                   t0), so the cosmology chooses which bins'
                   corrections to amplify; identity init keeps
                   corr = 0 at epoch 1. See FiLMGenerator and
                   notes/film-conditioning.md.
    block_opts   = ResBlock options (None -> {}); its "act" also
                   reaches the TRF MLPs, so head and trunk share
                   one activation family.
  """
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, int_dim_res, geom,
               n_heads=2, n_blocks=4, n_blocks_trf=1,
               n_mlp_blocks=2, gate_init=0.1, shared_mlp=False,
               film=False, block_opts=None):
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

    # the head: n_blocks_trf transformer blocks straight on the
    # padded bin tokens (width = max_bin; no embedding, no output
    # projection). Every block is the identity at init, so
    # blocks(h) - h = 0 exactly. The trunk's activation reaches the
    # TRF MLPs too.
    trf_act = block_opts.get("act", activation_fcn)
    trf = []
    for _ in range(n_blocks_trf):
      trf.append(TRFBlock(self.max_bin, n_tokens=self.n_bins,
                          n_heads=n_heads,
                          n_mlp_blocks=n_mlp_blocks,
                          act=trf_act, shared_mlp=shared_mlp))
    self.trf = nn.ModuleList(trf)

    # FiLM (film=True): one identity-initialized generator per TRF
    # block predicts a per-token (gamma, beta) from the parameters,
    # re-injecting cosmology the head otherwise never sees. At init
    # gamma = 1 / beta = 0, so blocks(t0) == t0 and corr = 0 still
    # hold exactly. None (default) = the fixed, parameter-blind
    # head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_trf):
        gens.append(FiLMGenerator(n_cond=input_dim,
                                  n_channels=self.n_bins))
      self.film_gens = nn.ModuleList(gens)

    # learnable scalar gate on the correction (small init, not 0).
    self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    # Frozen basis-change buffers, exactly ResCNN's (reminder:
    # W_fd = f -> d, full-whitened -> diagonal theta order /sigma;
    # W_df = d -> f, its inverse -- subscripts in multiply order).
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
    # (reminder: W_fd = f -> d, full-whitened -> diagonal theta
    # order; W_df = d -> f, its inverse. The subscripts read in
    # multiply order: x @ W_fd starts in f and lands in d.)
    h = y @ self.W_fd                 # f -> d, theta order
    # scatter into the padded per-bin layout: new_zeros makes the
    # (B, n_bins*max_bin) canvas (pad slots stay 0), the pad_idx
    # assignment places the n_keep real entries.
    padded = h.new_zeros(h.shape[0], self.n_bins * self.max_bin)
    padded[:, self.pad_idx] = h
    # (B, G*max_bin) -> (B, G, max_bin): each bin one token row, at
    # its natural width -- the blocks run directly on these.
    t0 = padded.view(-1, self.n_bins, self.max_bin)
    t = t0
    n = len(self.trf)
    for i in range(n):
      t = self.trf[i](t)              # cross-bin attention + MLPs
      if self.film_gens is not None:
        # FiLM re-injection: a per-token affine whose coefficients
        # depend on the parameters (identity at init). unsqueeze
        # broadcasts (B, n_bins) over the token width -- per bin,
        # never per position.
        gamma, beta = self.film_gens[i](x)
        t = gamma.unsqueeze(-1) * t + beta.unsqueeze(-1)
    # the correction is what the blocks added: every block is the
    # identity at init, so t - t0 = 0 exactly at epoch 1 (the
    # identity start, with no output projection needed to host it).
    # Gather the real entries back out of the padded layout
    # (dropping the pad slots), then return to the full-whitened
    # basis (reminder: @ W_df goes d -> f) and add through the gate.
    corr = (t - t0).reshape(
      -1, self.n_bins * self.max_bin)[:, self.pad_idx]
    return y + self.gate * (corr @ self.W_df)
