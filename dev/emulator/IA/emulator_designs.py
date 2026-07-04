"""Factored intrinsic-alignment template models."""

import torch
import torch.nn as nn

from ..activations import activation_fcn
from ..emulator_designs_building_blocks import (
  Affine, ResBlock, TRFBlock, FiLMGenerator, rescale_kernel_size)


class NLATemplateMLP(nn.Module):
  """
  Factored NLA emulator: maps the 11 non-A1_1 params (cosmo + A1_2)
  to three whitened templates [GG, GI, II]. The IA amplitude A1_1 is
  applied in closed form by the loss (xi = GG + A1_1*GI +
  A1_1^2*II), so it never enters the network -- making the A1_1
  generalization exact.

  Input layout (NLAInputGeometry.encode): last column is the raw
  A1_1 (for the loss); the model uses only [:, :-1]. output_dim =
  n_keep (one template width); emits 3*n_keep, reshapes to
  (B, 3, n_keep).
  """
  def __init__(self, input_dim, output_dim, int_dim_res,
               n_blocks=4, block_opts=None):
    """Build the residual trunk and the 3-template output head.

    Arguments:
      input_dim   = full encoded input width (12 = 11 model
                    features + the appended A1_1 column).
      output_dim  = one template's length (n_keep, the unmasked dv
                    size); 3 are emitted.
      int_dim_res = internal residual width.
      n_blocks    = number of residual blocks.
      block_opts  = ResBlock options dict (None -> {}).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    self.n_keep = output_dim
    # n_in = real input width: drop the 1 appended A1_1 column
    # (the loss's input, not the net's).
    self.n_in   = input_dim - 1
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    # one output projection emitting all three templates stacked.
    layers.append(nn.Linear(in_features=int_dim_res, out_features=3 * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """Map cosmo + A1_2 to the three whitened templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last column is
          A1_1 (ignored), [:, :-1] the whitened cosmo + A1_2
          features the templates depend on.

    Returns:
      (B, 3, n_keep): the whitened templates [GG, GI, II].
    """
    h = self.model(x[:, :self.n_in])           # (B, 3*n_keep)
    # view reshapes without copying: the flat (B, 3*n_keep) row
    # splits into (B, 3, n_keep) -- first n_keep entries GG, next
    # GI, last II. (view needs contiguous memory, which a Linear
    # output is, so the reshape is free.)
    return h.view(x.shape[0], 3, self.n_keep)   # (B, 3, n_keep)


class TemplateMLP(nn.Module):
  """
  Factored IA emulator: maps the non-amplitude parameters (cosmo +
  photo-z + the IA evolution powers eta) to n_templates whitened
  templates. The IA amplitudes are applied in closed form by the
  loss, so they never enter the network -- making the amplitude
  generalization exact and prior-width-independent.

  Input layout (AmplitudeFactorGeometry.encode): last n_amps columns
  are the raw amplitudes (for the loss); the model uses only
  [:, :-n_amps]. output_dim = n_keep (one template width); emits
  n_templates*n_keep, reshapes to (B, n_templates, n_keep). NLA:
  n_amps=1, n_templates=3; TATT: n_amps=3, n_templates=10.

    x  (B, input_dim)            encoded params; the last n_amps
       │                         columns = raw amplitudes, read by
       │                         the loss only
       │  drop the amplitudes [:, :-n_amps]; ResMLP layer stack
       ▼
    h  (B, T * n_keep)           all templates, flat
       │  view: one slice per template (no copy)
       ▼
    t  (B, T, n_keep)            whitened templates, coeff_fn order
       │  in the LOSS (TemplateFactoredChi2), not in the model:
       │  xi = sum_t coeff_t(A) * t_t, then chi2 vs the truth
       ▼
    chi2

  (legend: B = batch rows; input_dim = encoded width including the
  n_amps appended amplitude columns; T = n_templates (3 nla / 10
  tatt); n_keep = kept dv length = output_dim, one template's width;
  coeff_t(A) = the closed-form amplitude polynomial of the raw
  amplitudes A -- nla_coeffs / tatt_coeffs in IA/loss_functions.py.)

  factored = True is a capability flag (like the losses'
  needs_params): EmulatorExperiment reads it to pick the
  AmplitudeFactorGeometry input encoding and the template-combining
  loss, so a new factored model opts in by setting the flag rather
  than by being added to an isinstance check.
  """
  factored = True

  def __init__(self, input_dim, output_dim, n_amps,
               n_templates, int_dim_res, n_blocks=4,
               block_opts=None):
    """Build the residual trunk and the template output head.

    Arguments:
      input_dim   = full encoded input width (non-amplitude
                    features + the n_amps appended amplitudes).
      output_dim  = one template's length (n_keep, the unmasked dv
                    size); n_templates are emitted.
      n_amps      = appended amplitude columns to drop from the
                    input (1 NLA, 3 TATT).
      n_templates = templates to emit (3 NLA, 10 TATT); must match
                    the coeff_fn's length.
      int_dim_res = internal residual width.
      n_blocks    = number of residual blocks.
      block_opts  = ResBlock options dict (None -> {}).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    self.n_keep      = output_dim
    self.n_templates = n_templates
    # n_in = real input width: drop the n_amps amplitude columns.
    self.n_in = input_dim - n_amps
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    layers.append(nn.Linear(in_features=int_dim_res,
                            out_features=n_templates * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

  def forward(self, x):
    """Map the non-amplitude params to the whitened templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last n_amps
          columns are the amplitudes (ignored), [:, :-n_amps] the
          whitened cosmo + photo-z + eta features the templates
          depend on.

    Returns:
      (B, n_templates, n_keep): the whitened templates, in coeff_fn
      order (template 0 carries the no-IA / center part).
    """
    h = self.model(x[:, :self.n_in])
    # view reshapes the flat (B, n_templates*n_keep) output into
    # (B, n_templates, n_keep) without copying -- each template's
    # n_keep values one slice along axis 1.
    return h.view(x.shape[0], self.n_templates, self.n_keep)


class TemplateResCNN(nn.Module):
  """
  Factored IA emulator with a bins-as-channels 1D-CNN correction
  head: the TemplateMLP trunk emits n_templates whitened templates,
  then a single-kernel conv stack corrects them before the loss
  combines them. The amplitude polynomial is untouched (the loss
  still forms xi = sum_t c_t * template_t from the appended raw
  amplitudes), so the correction inherits the factored design's
  exactness in the amplitudes.

  Why correct the templates, not the combined xi: correcting after
  the combine would need the amplitudes in the network, surrendering
  the exact generalization the factoring buys.

    x  (B, input_dim)            encoded params; the last n_amps
       │                         columns = raw amplitudes, loss-only
       │  trunk = the TemplateMLP stack on [:, :-n_amps]
       │  (phase "head": run under no_grad, trunk frozen)
       ▼
    y  (B, T, n_keep)            whitened templates (also the skip;
       │                         phase "trunk" returns y here)
       │  @ W_fd                 f -> d: theta order, /sigma
       ▼
       │  pad_idx scatter        pad slots stay zero
       ▼
    c  (B, T*n_bins, max_bin)    (template, bin) pairs = channels
       │  n_blocks_cnn x [Conv1d + act]
       ▼
       │  pad_idx gather         drop the pad slots
       ▼
    corr (B, T, n_keep)
       │  @ W_df                 d -> f, back to full whitening
       ▼
    out = y + gate * corr        per-template gate (T scalars);
                                 corr = 0 at init (identity start)

  (legend: B = batch rows; T = n_templates (3 nla / 10 tatt);
  n_keep = kept dv length = output_dim, one template's width;
  n_bins = tomographic (xi+/-, source-pair) bins; max_bin = the
  longest bin's kept theta count = the padded width; f / d = the
  full-whitened / diagonal-theta bases, see the W_fd buffer
  comments; phases = set_train_phase, see its docstring.)

  The head is ResCNN's bins-as-channels design with the templates
  joining the channel axis: each template's theta-order dv splits
  into its (xi+/-, source-pair) bins, and the conv channels are the
  (template, bin) pairs -- one Conv1d(n_templates*n_bins ->
  n_templates*n_bins, kernel_size) slides a single kernel along
  theta over everything at once, so the correction is theta-local,
  cross-BIN, and cross-TEMPLATE in one map. No channel expansion:
  the head's tensors never grow beyond the (padded) templates'
  size, so the bandwidth wall the old expand-to-C-filters head hit
  cannot occur by construction. Each block is one conv + one
  activation. The head hyperparameters are kernel_size (+ the
  rescale_kernel flag), n_blocks_cnn, groups, separable, film, and
  gate_init. The
  A1 exactness is untouched: it lives in the loss's combine, and
  the head emits amplitude-blind templates (true for every groups
  choice -- the head acts before the combine).

  groups restricts that cross-bin + cross-template mixing along
  the two physical cuts the channel order offers. The channels are
  template-major (template, bin) pairs, and within each template
  the bins run xi+ pairs then xi- pairs (cosmolike's dv layout,
  reconstructed by build_shear_angle_map); a grouped conv splits
  the channels into `groups` consecutive blocks that never mix:

    channels (nla names drawn; T templates in general):

      GG xi+ │ GG xi- ┃ GI xi+ │ GI xi- ┃ II xi+ │ II xi-

    groups=1:    no cut -- every output reads every (template,
                 bin) pair (the default: full mixing)
    groups=T:    cuts at the ┃ -- GG / GI / II corrected in
                 isolation, bins still mix within a template
    groups=2*T:  cuts at ┃ and │ -- templates isolated AND xi+
                 isolated from xi-; only bins of the same
                 (template, branch) mix

  (legend: T = n_templates (3 nla / 10 tatt); each drawn block
  holds n_bins/2 = 15 source pairs; per-block conv parameters =
  n_ch * (n_ch/groups) * kernel_size + n_ch with n_ch = T*n_bins,
  so the cuts divide the conv weights by `groups`. Physics
  framing: groups=1 encodes the hypothesis that the trunk's GG /
  GI / II residuals share structure -- they come from the same
  underlying power-spectrum integrals -- and groups=T / 2*T are
  its ablations. The xi boundary is validated against geom.pm_kept
  at build: bin_sizes drops fully-masked bins, so a wholly-masked
  bin on one branch would silently shift the cut -- that fails
  loudly instead.)

  separable factors each block's remaining sum -- smoothing along
  theta and mixing channels -- into a depthwise per-channel k-tap
  theta filter (n_ch*k weights) followed by a pointwise 1x1
  channel mix honoring `groups` (n_ch*(n_ch/groups) weights),
  versus the plain block's joint n_ch*(n_ch/groups)*k. With no
  activation between them the pair composes into one constrained
  conv, w[o, c, t] = pointwise[o, c] * depthwise[c, t] -- a
  low-rank factorization of the same sum, ~k/2 times fewer
  weights; the zero-init identity start moves to the last block's
  pointwise. Full graph and the assumption it adds: see ResCNN's
  separable paragraph (n_ch = n_templates * n_bins here).

  The basis handling is ResCNN's: templates live in the full
  (cov-eigenbasis) whitening, which scrambles theta, so fixed
  buffers map each template to the diagonal view (theta order,
  per-element /sigma) for the conv and back (W_fd / W_df, see
  ResCNN), and the fixed pad_idx buffer scatters each template into
  the padded (n_bins, max_bin) layout and gathers the corrections
  back. Buffers, not live geometry calls in forward, so
  torch.compile CUDA graphs stay safe. The gate is per template
  (n_templates scalars, not one): the templates carry very
  different whitened magnitudes (GG holds the center, II is a
  small quadratic piece), so each learns its own correction scale;
  at gate = 0 the model is exactly the TemplateMLP trunk.

  Input layout is TemplateMLP's (last n_amps columns are the raw
  amplitudes, dropped from the trunk input); output is
  (B, n_templates, n_keep), what TemplateFactoredChi2 consumes --
  so swapping the architecture (name: resmlp -> rescnn at ia: nla)
  changes only the model.

  The head starts as an exact identity (the last block's output
  layer is zero-initialized, and the final activation maps 0 -> 0),
  so at epoch 1 the model IS its trunk -- no random-weight
  perturbation. That also enables two-phase training
  (train_args.trunk_epochs > 0, orchestrated by run_emulator via
  set_train_phase): first train the trunk alone with the head
  bypassed (pure-TemplateMLP cost per epoch), then freeze the trunk
  (run under no_grad, no trunk backward) and let the head learn
  only the residual, starting from the identity so the loss is
  continuous across the switch.

  factored / needs_geom / needs_bins are capability flags
  EmulatorExperiment reads: factored picks the
  AmplitudeFactorGeometry input encoding and the template-combining
  loss; needs_geom injects geom and defaults compile_mode to
  "default" (reduce-overhead's CUDA-graph capture trips on the
  gated skip-add); needs_bins runs build_shear_angle_map on the
  data geometry (it attaches bin_sizes) before the model is built.
  """
  factored   = True
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, n_amps,
               n_templates, int_dim_res, geom, kernel_size=11,
               rescale_kernel=False, groups=1, separable=False,
               film=False, n_blocks=4, n_blocks_cnn=1,
               gate_init=0.1, block_opts=None):
    """Build the template trunk, the conv head, the buffers.

    Arguments:
      input_dim    = full encoded input width (non-amplitude
                     features + the n_amps appended amplitudes).
      output_dim   = one template's length (n_keep); n_templates
                     are emitted and corrected.
      n_amps       = appended amplitude columns to drop from the
                     input (1 NLA, 3 TATT).
      n_templates  = templates to emit (3 NLA, 10 TATT); must match
                     the coeff_fn's length.
      int_dim_res  = internal residual width of the trunk.
      geom         = full-whitening DataVectorGeometry carrying
                     bin_sizes; its evecs / sqrt_ev define the
                     basis-change buffers.
      kernel_size  = conv kernel width (odd, same-padded), tuned
                     as if the head had ONE block. With
                     rescale_kernel it states the target receptive
                     field; without, it is used verbatim for every
                     block.
      rescale_kernel = False (default): every block uses
                     kernel_size as given. True: the per-block
                     kernel shrinks with depth so the
                     n_blocks_cnn-deep stack keeps a single
                     kernel_size-wide block's view (receptive
                     field n*(k-1)+1 >= kernel_size, see
                     rescale_kernel_size); the resolved width is
                     stored as self.kernel_size.
      groups       = channel-mixing restriction: 1 (default,
                     dense mixing), n_templates (GG / GI / II
                     never mix), or 2*n_templates (templates
                     isolated AND xi+ isolated from xi-). See the
                     groups paragraph and graph in the class
                     docstring; other values error, and the xi
                     boundary is validated against geom.pm_kept
                     at build.
      separable    = False (default): one joint conv per block.
                     True: factor each block into a depthwise
                     theta filter + a pointwise channel mix (a
                     low-rank factorization of the same sum,
                     ~k/2 times fewer weights). See the separable
                     paragraph in the class docstring.
      film         = False (default): the head is one fixed map,
                     blind to the cosmology. True: re-inject the
                     NON-AMPLITUDE parameters into every block as
                     a per-(template, bin)-channel affine, conv ->
                     gamma(z)*c + beta(z) -> act, one
                     identity-initialized FiLMGenerator per block
                     (Linear(n_in, 2*n_ch); see its docstring and
                     notes/film-conditioning.md). The conditioning
                     is x[:, :n_in] -- the amplitudes NEVER enter,
                     so the head stays amplitude-blind and the
                     closed-form amplitude exactness survives. The
                     per-template gate stays as the outer valve.
      n_blocks     = residual blocks in the trunk.
      n_blocks_cnn = stacked conv+activation correction blocks.
      gate_init    = initial per-template correction scale. Small
                     (default 0.1) to start near the pure trunk;
                     not 0 -- a 0 gate strands the CNN with no
                     gradient, so it never learns.
      block_opts   = ResBlock options (None -> {}); its "act" is
                     also handed to the CNN head, so head and trunk
                     share one activation family (falls back to
                     activation_fcn, the paper's H).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    assert kernel_size % 2 == 1, (
      "kernel_size must be odd so same-padding keeps the length")
    assert hasattr(geom, "bin_sizes"), (
      "TemplateResCNN needs geom.bin_sizes -- run "
      "build_shear_angle_map(geom) first (EmulatorExperiment does "
      "this for models with the needs_bins flag)")
    self.n_keep      = output_dim
    self.n_templates = n_templates
    # n_in = real input width: drop the n_amps amplitude columns.
    self.n_in = input_dim - n_amps

    # trunk: the TemplateMLP layer stack, emitting all templates in
    # the full-whitened basis (well conditioned).
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    layers.append(nn.Linear(in_features=int_dim_res,
                            out_features=n_templates * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

    # the bin split: per-bin kept counts, contiguous in theta order,
    # and the fixed scatter/gather index into the padded layout (bin
    # g's j-th entry at g*max_bin + j), applied per template.
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

    # the head: n_blocks_cnn x (one conv + one activation), with the
    # (template, bin) pairs as the channels -- a single kernel over
    # everything (see the class docstring). Takes the trunk's
    # activation so head and trunk share one family; act(max_bin)
    # gives per-position parameters, broadcast over the channels.
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

    # groups: the channels are template-major (template, bin)
    # pairs, so only the template cut (groups = n_templates) and
    # the template+branch cut (groups = 2*n_templates) land on
    # physical boundaries (see the docstring). For the branch cut,
    # validate the bin layout against the geometry rather than
    # assume it: each bin is a contiguous run of kept elements
    # sharing one pm (0 = xi+, 1 = xi-); every template block
    # repeats the same bin order, so checking the bin list once
    # covers all templates.
    assert groups in (1, n_templates, 2 * n_templates), (
      f"TemplateResCNN groups must be 1 (dense), n_templates "
      f"({n_templates}: templates never mix) or 2*n_templates "
      f"({2 * n_templates}: templates never mix AND xi+ never "
      "mixes with xi-); the channels are template-major, so no "
      "other cut has a physical meaning")
    if groups == 2 * n_templates:
      assert hasattr(geom, "pm_kept") and self.n_bins % 2 == 0, (
        "the branch cut needs geom.pm_kept "
        "(build_shear_angle_map) and an even bin count")
      pm_bins = []
      start = 0
      for s in sizes:
        pm_bins.append(int(geom.pm_kept[start]))
        start += s
      half = self.n_bins // 2
      assert (all(pm == 0 for pm in pm_bins[:half])
              and all(pm == 1 for pm in pm_bins[half:])), (
        "the branch cut needs the first half of each template's "
        "bins to be xi+ and the second half xi- (a fully-masked "
        "bin on one branch breaks the split); per-bin branches "
        f"here: {pm_bins}")

    pad = (kernel_size - 1) // 2
    n_ch = n_templates * self.n_bins
    convs, acts = [], []
    for _ in range(n_blocks_cnn):
      if separable:
        # depthwise-separable factorization (see the class
        # docstring): a per-channel k-tap theta filter (groups =
        # n_ch: no mixing), then a pointwise 1x1 channel mix
        # honoring `groups`. No activation between the two -- the
        # pair is a low-rank factorization of the plain block's
        # conv; the block's one activation follows as usual.
        # Sequential keeps forward unchanged (convs[i] is callable
        # either way).
        convs.append(nn.Sequential(
          nn.Conv1d(in_channels=n_ch,
                    out_channels=n_ch,
                    kernel_size=kernel_size,
                    padding=pad,
                    groups=n_ch),
          nn.Conv1d(in_channels=n_ch,
                    out_channels=n_ch,
                    kernel_size=1,
                    groups=groups)))
      else:
        convs.append(nn.Conv1d(in_channels=n_ch,
                               out_channels=n_ch,
                               kernel_size=kernel_size,
                               padding=pad,
                               groups=groups))
      acts.append(cnn_act(self.max_bin))
    self.convs = nn.ModuleList(convs)
    self.acts  = nn.ModuleList(acts)

    # one learnable gate per template, (n_templates, 1) so it
    # broadcasts over (B, n_templates, n_keep).
    self.gate = nn.Parameter(
      torch.full((n_templates, 1), float(gate_init)))

    # Zero-init the LAST conv, so the head starts as an exact
    # identity on the model output: the activation maps 0 -> 0, so a
    # zeroed conv gives corr = 0 and out = trunk exactly -- no
    # random-weight perturbation at epoch 1 (or at a phase handoff).
    # Gradients still reach the zeroed conv (d corr/d w depends on
    # its INPUT, not its weights, times the nonzero gate), so it
    # grows from 0 as soon as a correction helps; earlier blocks
    # wake up one step later. The standard zero-init-residual-branch
    # trick. Only the last conv is zeroed -- zeroing all would kill
    # every gradient path. In a separable block the zero lives on
    # the pointwise (second) conv; the depthwise filter keeps its
    # init (zeroing both would zero the pointwise's input and
    # stall its wake-up).
    last = self.convs[-1][1] if separable else self.convs[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

    # FiLM (film=True): one identity-initialized generator per
    # block predicts a per-(template, bin) (gamma, beta) from the
    # NON-amplitude parameters (n_cond = n_in: the amplitudes never
    # enter, so the head stays amplitude-blind and the closed-form
    # amplitude exactness survives). At init gamma = 1 / beta = 0,
    # so the identity start above is untouched. None (default) =
    # the fixed, parameter-blind head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_cnn):
        gens.append(FiLMGenerator(n_cond=self.n_in,
                                  n_channels=n_ch))
      self.film_gens = nn.ModuleList(gens)

    # training phase, set by set_train_phase: "joint" (default,
    # everything trains), "trunk" (head frozen AND bypassed -- the
    # model runs as a pure TemplateMLP at TemplateMLP cost), "head"
    # (trunk frozen and run under no_grad -- backward touches the
    # head only). A plain Python attribute: torch.compile guards on
    # it and recompiles once per phase switch.
    self._phase = "joint"

    # Frozen basis-change buffers, exactly ResCNN's: x @ W_fd maps
    # full-whitened -> theta order (/sigma), x @ W_df maps back
    # (W_df = W_fd^{-1}). sigma = per-element sqrt(diag cov).
    evecs   = geom.evecs.detach()
    sqrt_ev = geom.sqrt_ev.detach()
    sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
    self.register_buffer(
      "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
    self.register_buffer(
      "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])

  def set_train_phase(self, phase):
    """Switch the two-phase training mode (run_emulator calls this).

    Freezes/unfreezes the parameter groups and sets the forward
    behavior:
      "joint" = everything trains, head active (the default).
      "trunk" = head frozen and BYPASSED: forward returns the bare
                templates, so phase-1 epochs cost exactly a
                TemplateMLP (no head compute, no head gradients).
                With the zero-init head this changes nothing
                numerically -- corr was already 0.
      "head"  = trunk frozen and run under no_grad: backward
                touches only the conv head + gates, so phase-2
                epochs skip the whole trunk backward. The head
                starts from its zero-init identity, so the loss is
                continuous across the switch.

    Arguments:
      phase = "joint" | "trunk" | "head".
    """
    if phase not in ("joint", "trunk", "head"):
      raise ValueError(f"unknown train phase {phase!r}; "
                       "use 'joint', 'trunk', or 'head'")
    self._phase = phase
    trunk_on = phase in ("joint", "trunk")
    head_on  = phase in ("joint", "head")
    for p in self.model.parameters():
      p.requires_grad_(trunk_on)
    for p in self.convs.parameters():
      p.requires_grad_(head_on)
    for p in self.acts.parameters():
      p.requires_grad_(head_on)
    if self.film_gens is not None:
      # the FiLM generators are head parameters: frozen with the
      # head in the trunk phase, trained with it in the head phase.
      for p in self.film_gens.parameters():
        p.requires_grad_(head_on)
    self.gate.requires_grad_(head_on)

  def forward(self, x):
    """Map the non-amplitude params to conv-corrected templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last n_amps
          columns are the amplitudes (ignored here, read by the
          loss), [:, :-n_amps] the whitened features.

    Returns:
      (B, n_templates, n_keep): the corrected whitened templates,
      in coeff_fn order.
    """
    B = x.shape[0]
    # trunk templates in the full-whitened basis. In the "head"
    # phase the trunk is frozen, so skip building its autograd
    # graph: no trunk activations stored, no trunk backward.
    if self._phase == "head":
      with torch.no_grad():
        y = self.model(x[:, :self.n_in]).view(
          B, self.n_templates, self.n_keep)  # (B, T, n_keep)
    else:
      y = self.model(x[:, :self.n_in]).view(
        B, self.n_templates, self.n_keep)    # (B, T, n_keep)
    # "trunk" phase: the head is frozen at its zero-init identity,
    # so its output is known to be y -- skip the compute entirely.
    if self._phase == "trunk":
      return y
    # theta order per template (the matmul broadcasts over (B, T)),
    # then scatter into the padded per-bin layout: the (template,
    # bin) pairs become the conv channels. (reminder: W_fd = f -> d,
    # full-whitened -> diagonal theta order; W_df = d -> f, its
    # inverse -- subscripts read in multiply order.)
    h = y @ self.W_fd                         # (B, T, n_keep) theta
    padded = h.new_zeros(B, self.n_templates,
                         self.n_bins * self.max_bin)
    padded[..., self.pad_idx] = h
    c = padded.view(B, self.n_templates * self.n_bins,
                    self.max_bin)
    n = len(self.convs)
    for i in range(n):
      c = self.convs[i](c)                    # cross-bin+template
      if self.film_gens is not None:
        # FiLM re-injection: a per-(template, bin) affine whose
        # coefficients depend on the NON-amplitude parameters
        # (identity at init; amplitude-blind by construction --
        # the slice below is the same one the trunk consumes).
        # unsqueeze broadcasts (B, T*n_bins) over the theta axis.
        gamma, beta = self.film_gens[i](x[:, :self.n_in])
        c = gamma.unsqueeze(-1) * c + beta.unsqueeze(-1)
      c = self.acts[i](c)
    # gather the real entries back out of the padding (per
    # template), return to the full-whitened basis (reminder:
    # @ W_df goes d -> f), add through the per-template gate.
    c = c.view(B, self.n_templates, self.n_bins * self.max_bin)
    corr = c[..., self.pad_idx]               # (B, T, n_keep)
    return y + self.gate * (corr @ self.W_df)


class TemplateResTRF(nn.Module):
  """
  Factored IA emulator with a transformer correction head: the
  TemplateMLP trunk emits n_templates whitened templates, and a
  transformer whose TOKENS are the (template, bin) pairs corrects
  them before the loss combines them in closed form. The amplitude
  polynomial is untouched, so the correction inherits the factored
  design's exactness in the amplitudes (that exactness lives in the
  loss's combine; the head only ever sees amplitude-blind
  templates).

  The tokens live at their NATURAL width (max_bin, the padded bin
  length), with no embedding in and no projection out -- the same
  no-adapter design as ResTRF, and the same pairs-as-tokens move as
  the conv head's pairs-as-channels (TemplateResCNN). Attention
  therefore runs across ALL n_templates*n_bins tokens at once:
  cross-bin AND cross-template in one map; each token's own MLP
  stack (BinLinear; the deviation from the textbook shared FFN,
  which also replaces the positional encoding) specializes its
  correction. Bins differ in length, so each is padded to max_bin
  inside a fixed pad_idx buffer (scatter to pad, gather to unpad;
  pad slots stay zero).

    x  (B, input_dim)            encoded params; the last n_amps
       │                         columns = raw amplitudes, loss-only
       │  trunk = the TemplateMLP stack on [:, :-n_amps]
       │  (phase "head": run under no_grad, trunk frozen)
       ▼
    y  (B, T, n_keep)            whitened templates (also the skip;
       │                         phase "trunk" returns y here)
       │  @ W_fd                 f -> d: theta order, /sigma
       ▼
       │  pad_idx scatter        pad slots stay zero
       ▼
    t0 (B, T*n_bins, max_bin)    one token per (template, bin) pair
       │  n_blocks_trf x TRFBlock
       │                         cross-bin AND cross-template
       │                         attention + per-token MLPs
       ▼
    t  (B, T*n_bins, max_bin)
       │  corr = t - t0          what the blocks added (0 at init)
       │  pad_idx gather, @ W_df     d -> f, back to full whitening
       ▼
    out = y + gate * corr        per-template gate (T scalars)

  (legend: B = batch rows; T = n_templates (3 nla / 10 tatt);
  n_keep = kept dv length = output_dim, one template's width;
  n_bins = tomographic (xi+/-, source-pair) bins; max_bin = the
  longest bin's kept theta count = the padded token width; f / d =
  the full-whitened / diagonal-theta bases, see the W_fd buffer
  comments; phases = set_train_phase, see TemplateResCNN's.)

  The correction is corr = blocks(h) - h: every TRFBlock is exactly
  the identity at init (zero-initialized branch outputs, see
  TRFBlock), so corr = 0 and the model equals its trunk at epoch 1 --
  enabling the two-phase schedule (train_args.trunk_epochs,
  orchestrated by run_emulator via set_train_phase): first the
  trunk alone with the head bypassed (pure-TemplateMLP cost), then
  the trunk frozen under no_grad while the head learns only the
  residual, loss-continuous at the handoff.

  factored / needs_geom / needs_bins are capability flags
  EmulatorExperiment reads: factored picks AmplitudeFactorGeometry
  + the template-combining loss; needs_geom injects geom and
  defaults compile_mode to "default"; needs_bins runs
  build_shear_angle_map on the data geometry (it attaches
  bin_sizes) before the model is built.
  """
  factored   = True
  needs_geom = True
  needs_bins = True

  def __init__(self, input_dim, output_dim, n_amps,
               n_templates, int_dim_res, geom, n_heads=2,
               n_blocks=4, n_blocks_trf=1, n_mlp_blocks=2,
               gate_init=0.1, shared_mlp=False, film=False,
               block_opts=None):
    """Build the template trunk, the TRF head, the buffers.

    Arguments:
      input_dim    = full encoded input width (non-amplitude
                     features + the n_amps appended amplitudes).
      output_dim   = one template's length (n_keep); n_templates
                     are emitted and corrected.
      n_amps       = appended amplitude columns to drop from the
                     input (1 NLA, 3 TATT).
      n_templates  = templates to emit (3 NLA, 10 TATT); must match
                     the coeff_fn's length.
      int_dim_res  = internal residual width of the trunk.
      geom         = full-whitening DataVectorGeometry carrying
                     bin_sizes; its evecs / sqrt_ev define the
                     basis buffers.
      n_heads      = attention heads per TRFBlock; must divide the
                     token width max_bin (the LSST-Y1 cosmic-shear
                     run keeps max_bin = 26 theta points per bin,
                     allowing n_heads = 1, 2, or 13; default 2).
      n_blocks     = residual blocks in the trunk.
      n_blocks_trf = stacked transformer blocks.
      n_mlp_blocks = depth of each token's private MLP stack inside
                     every TRFBlock.
      gate_init    = initial per-template correction scale (small,
                     not 0 -- a 0 gate strands the head with no
                     gradient).
      shared_mlp   = False (default): per-token unique MLPs. True:
                     one MLP shared by every (template, bin) token
                     -- the textbook block, the ablation isolating
                     the unique-MLP deviation (see TRFBlock's
                     permutation-equivariance caveat).
      film         = False (default): the head is one fixed map,
                     blind to the cosmology. True: after every
                     TRFBlock, modulate the token stream with a
                     per-(template, bin)-token affine gamma(z)*t +
                     beta(z) from an identity-initialized
                     FiLMGenerator (one per block), conditioned on
                     x[:, :n_in] only -- the amplitudes never
                     enter, so the head stays amplitude-blind and
                     the closed-form amplitude exactness survives.
                     Identity init keeps corr = 0 at epoch 1. See
                     FiLMGenerator and notes/film-conditioning.md.
      block_opts   = ResBlock options (None -> {}); its "act" also
                     reaches the TRF MLPs, so head and trunk share
                     one activation family.
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
    assert hasattr(geom, "bin_sizes"), (
      "TemplateResTRF needs geom.bin_sizes -- run "
      "build_shear_angle_map(geom) first (EmulatorExperiment does "
      "this for models with the needs_bins flag)")
    self.n_keep      = output_dim
    self.n_templates = n_templates
    # n_in = real input width: drop the n_amps amplitude columns.
    self.n_in = input_dim - n_amps

    # trunk: the TemplateMLP layer stack, emitting all templates in
    # the full-whitened basis (well conditioned).
    layers = [nn.Linear(in_features=self.n_in, out_features=int_dim_res)]
    for _ in range(n_blocks):
      layers.append(ResBlock(int_dim_res, **block_opts))
    layers.append(nn.Linear(in_features=int_dim_res,
                            out_features=n_templates * output_dim))
    layers.append(Affine())
    self.model = nn.Sequential(*layers)

    # the bin split: per-bin kept counts, contiguous in theta order.
    sizes = []
    for s in geom.bin_sizes:
      sizes.append(int(s))
    self.n_bins  = len(sizes)
    self.max_bin = max(sizes)
    # pad_idx maps each kept theta-order position to its slot in the
    # padded (n_bins, max_bin) layout (bin g's j-th entry at
    # g*max_bin + j); one fixed buffer scatters to pad and gathers
    # to unpad, per template.
    pos = []
    for g in range(self.n_bins):
      for j in range(sizes[g]):
        pos.append(g * self.max_bin + j)
    self.register_buffer(
      "pad_idx", torch.tensor(pos, dtype=torch.long))

    # the head: n_blocks_trf transformer blocks straight on the
    # (template, bin) tokens at their natural width max_bin -- no
    # embedding, no output projection (the same pairs-as-tokens move
    # as the conv head's pairs-as-channels). Every block is the
    # identity at init, so blocks(h) - h = 0 exactly. The trunk's
    # activation reaches the TRF MLPs too.
    trf_act = block_opts.get("act", activation_fcn)
    trf = []
    for _ in range(n_blocks_trf):
      trf.append(TRFBlock(self.max_bin,
                          n_tokens=n_templates * self.n_bins,
                          n_heads=n_heads,
                          n_mlp_blocks=n_mlp_blocks,
                          act=trf_act, shared_mlp=shared_mlp))
    self.trf = nn.ModuleList(trf)

    # FiLM (film=True): one identity-initialized generator per TRF
    # block predicts a per-(template, bin)-token (gamma, beta) from
    # the NON-amplitude parameters (n_cond = n_in: the amplitudes
    # never enter, so the head stays amplitude-blind and the
    # closed-form amplitude exactness survives). At init gamma = 1
    # / beta = 0, so blocks(t0) == t0 and corr = 0 still hold
    # exactly. None (default) = the fixed, parameter-blind head.
    self.film_gens = None
    if film:
      gens = []
      for _ in range(n_blocks_trf):
        gens.append(FiLMGenerator(
          n_cond=self.n_in,
          n_channels=n_templates * self.n_bins))
      self.film_gens = nn.ModuleList(gens)

    # one learnable gate per template, (n_templates, 1) so it
    # broadcasts over (B, n_templates, n_keep).
    self.gate = nn.Parameter(
      torch.full((n_templates, 1), float(gate_init)))

    # Frozen basis-change buffers, exactly ResCNN's: x @ W_fd maps
    # full-whitened -> theta order (/sigma), x @ W_df maps back.
    evecs   = geom.evecs.detach()
    sqrt_ev = geom.sqrt_ev.detach()
    sigma   = torch.sqrt(((evecs * sqrt_ev) ** 2).sum(1))
    self.register_buffer(
      "W_fd", (sqrt_ev[:, None] * evecs.t()) / sigma[None, :])
    self.register_buffer(
      "W_df", (sigma[:, None] * evecs) / sqrt_ev[None, :])

    # training phase, set by set_train_phase (see TemplateResCNN):
    # "joint" (default), "trunk" (head frozen AND bypassed), "head"
    # (trunk frozen and run under no_grad).
    self._phase = "joint"

  def set_train_phase(self, phase):
    """Switch the two-phase training mode (run_emulator calls this).

    Identical contract to TemplateResCNN.set_train_phase: "joint"
    trains everything; "trunk" freezes AND bypasses the head (pure
    TemplateMLP cost; numerically a no-op thanks to the zero-init
    identity); "head" freezes the trunk and runs it under no_grad,
    so backward touches only the TRF head + gates.

    Arguments:
      phase = "joint" | "trunk" | "head".
    """
    if phase not in ("joint", "trunk", "head"):
      raise ValueError(f"unknown train phase {phase!r}; "
                       "use 'joint', 'trunk', or 'head'")
    self._phase = phase
    trunk_on = phase in ("joint", "trunk")
    head_on  = phase in ("joint", "head")
    for p in self.model.parameters():
      p.requires_grad_(trunk_on)
    for p in self.trf.parameters():
      p.requires_grad_(head_on)
    if self.film_gens is not None:
      # the FiLM generators are head parameters: frozen with the
      # head in the trunk phase, trained with it in the head phase.
      for p in self.film_gens.parameters():
        p.requires_grad_(head_on)
    self.gate.requires_grad_(head_on)

  def forward(self, x):
    """Map the non-amplitude params to TRF-corrected templates.

    Arguments:
      x = (B, input_dim) encoded parameters; the last n_amps
          columns are the amplitudes (ignored here, read by the
          loss), [:, :-n_amps] the whitened features.

    Returns:
      (B, n_templates, n_keep): the corrected whitened templates,
      in coeff_fn order.
    """
    B = x.shape[0]
    # trunk templates in the full-whitened basis. In the "head"
    # phase the trunk is frozen, so skip building its autograd
    # graph: no trunk activations stored, no trunk backward.
    if self._phase == "head":
      with torch.no_grad():
        y = self.model(x[:, :self.n_in]).view(
          B, self.n_templates, self.n_keep)  # (B, T, n_keep)
    else:
      y = self.model(x[:, :self.n_in]).view(
        B, self.n_templates, self.n_keep)    # (B, T, n_keep)
    # "trunk" phase: the head is frozen at its zero-init identity,
    # so its output is known to be y -- skip the compute entirely.
    if self._phase == "trunk":
      return y

    # theta order per template (the matmul broadcasts over (B, T)),
    # then scatter into the padded per-bin layout. (reminder: W_fd =
    # f -> d, full-whitened -> diagonal theta order; W_df = d -> f,
    # its inverse -- subscripts read in multiply order.)
    h = y @ self.W_fd                        # (B, T, n_keep)
    padded = h.new_zeros(B, self.n_templates,
                         self.n_bins * self.max_bin)
    padded[..., self.pad_idx] = h
    # (B, T, G*max_bin) -> (B, T*G, max_bin): one token per
    # (template, bin) pair at its natural width, template-major --
    # the same order as the conv head's channels. view is free
    # (padded is contiguous).
    t0 = padded.view(B, self.n_templates * self.n_bins,
                     self.max_bin)
    t = t0
    n = len(self.trf)
    for i in range(n):
      t = self.trf[i](t)            # cross-bin + cross-template
      if self.film_gens is not None:
        # FiLM re-injection: a per-(template, bin)-token affine
        # whose coefficients depend on the NON-amplitude
        # parameters (identity at init; amplitude-blind by
        # construction). unsqueeze broadcasts (B, T*n_bins) over
        # the token width.
        gamma, beta = self.film_gens[i](x[:, :self.n_in])
        t = gamma.unsqueeze(-1) * t + beta.unsqueeze(-1)
    # the correction is what the blocks added (t - t0 = 0 at init:
    # every block starts as the identity). Unpack the tokens back to
    # (B, T, G*max_bin), gather the real entries out of the padding,
    # return to the full-whitened basis (reminder: @ W_df goes
    # d -> f), add through the per-template gate.
    corr = (t - t0).view(B, self.n_templates,
                         self.n_bins * self.max_bin)
    corr = corr[..., self.pad_idx]           # (B, T, n_keep)
    return y + self.gate * (corr @ self.W_df)
