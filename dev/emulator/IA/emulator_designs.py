"""Factored intrinsic-alignment template models."""

import torch
import torch.nn as nn

from ..activations import activation_fcn
from ..emulator_designs_building_blocks import Affine, ResBlock, CNNBlock


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


class TemplateMixCNNBlock(nn.Module):
  """
  Template-mixing 1D-conv correction block: the n_templates axis is
  the conv's channel axis, so one kernel reads all templates at each
  theta and writes a correction for each -- (B, T, K) -> (B, T, K).

  Why it exists: the shared per-template CNNBlock folds the templates
  into the batch axis, so every intermediate is (T*B, channels, K) --
  at T=3, bs 768, K=705, channels 16 that is ~100 MB per tensor, and
  the head becomes memory-bandwidth-bound (the GPU spends its time
  moving those tensors, not computing). Making the templates input
  CHANNELS keeps the intermediates at (B, channels, K), a 3x traffic
  cut at identical FLOPs. The trade: the correction is no longer one
  kernel shared identically across templates -- it mixes them. That
  is legal for the factored design (the output is still n_templates
  amplitude-blind templates; exactness in A1 is a property of the
  combine, not the head) and physically reasonable (GG, GI, II are
  correlated correlation-function shapes along the same theta axis).

  The mid activation is unconditional here: conv (T->channels) and
  collapse (channels->T) are both linear convs, so without it they
  fold into a single T->T kernel and the channels add nothing.

  Arguments:
    dim         = template length K (the activations' feature width;
                  their per-element parameters broadcast over the
                  channel axis, as in CNNBlock's act_mid).
    n_templates = conv input/output channels (the template count).
    kernel_size = kernel width; odd, so same-padding keeps K.
    channels    = internal conv filters.
    act         = activation factory act(dim) -> module, shared with
                  the trunk (defaults to activation_fcn, the paper's
                  H).
  """
  def __init__(self, dim, n_templates, kernel_size=11,
               channels=16, act=activation_fcn):
    super().__init__()
    assert kernel_size % 2 == 1, (
      "kernel_size must be odd so same-padding keeps the length")
    pad = (kernel_size - 1) // 2
    # templates in -> `channels` filters; length preserved.
    self.conv     = nn.Conv1d(in_channels=n_templates,
                              out_channels=channels,
                              kernel_size=kernel_size,
                              padding=pad)
    self.act_mid  = act(dim)
    # filters back to one correction per template (1x1 conv = a
    # per-position weighted sum over channels).
    self.collapse = nn.Conv1d(in_channels=channels,
                              out_channels=n_templates,
                              kernel_size=1)
    self.act      = act(dim)

  def forward(self, x):
    """(B, n_templates, K) templates -> (B, n_templates, K) corrections."""
    h = self.conv(x)              # (B, channels, K)
    h = self.act_mid(h)           # keeps the two convs from folding
    h = self.collapse(h)          # (B, n_templates, K)
    return self.act(h)


class TemplateResCNN(nn.Module):
  """
  Factored IA emulator with a 1D-CNN correction head: the
  TemplateMLP trunk emits n_templates whitened templates, then one
  shared gated conv stack corrects each template's theta-local
  structure before the loss combines them. The amplitude polynomial
  is untouched (the loss still forms xi = sum_t c_t * template_t
  from the appended raw amplitudes), so the correction inherits the
  factored design's exactness in the amplitudes.

  Why correct per template, not the combined xi: correcting after
  the combine would need the amplitudes in the network, surrendering
  the exact generalization the factoring buys. Each template is
  itself a correlation-function shape along theta (GG, GI, II share
  the same angular axis and bin layout), so one conv kernel serves
  all of them -- the templates fold into the batch axis and the
  shared head is the sample-efficiency move (one theta-kernel
  learned from 3x the examples).

  template_mix=True swaps that head for TemplateMixCNNBlock: the
  templates become the conv's channel axis instead of folding into
  the batch. Same FLOPs, one third the memory traffic (the folded
  head's (3B, channels, K) intermediates make it bandwidth-bound on
  a consumer GPU), and cross-template features -- at the cost of the
  one-shared-kernel inductive bias. Either head leaves the A1
  exactness untouched: that lives in the loss's combine, and both
  heads emit amplitude-blind templates.

  The basis handling is ResCNN's: templates live in the full
  (cov-eigenbasis) whitening, which scrambles theta, so fixed
  buffers map each template to the diagonal view (theta order,
  per-element /sigma) for the conv and back (W_fd / W_df, see
  ResCNN). Buffers, not live geometry calls in forward, so
  torch.compile CUDA graphs stay safe. The gate is per template
  (n_templates scalars, not one): the templates carry very
  different whitened magnitudes (GG holds the center, II is a
  small quadratic piece), so each learns its own correction scale;
  at gate = 0 the model is exactly the TemplateMLP trunk.

  Input layout is TemplateMLP's (last n_amps columns are the raw
  amplitudes, dropped from the trunk input); output is
  (B, n_templates, n_keep), what TemplateFactoredChi2 consumes --
  so swapping nla -> rescnn_nla changes only the model.

  factored / conv_head are capability flags EmulatorExperiment
  reads: factored picks the AmplitudeFactorGeometry input encoding
  and the template-combining loss; conv_head injects geom (for the
  basis buffers) and defaults compile_mode to "default"
  (reduce-overhead's CUDA-graph capture trips on the gated
  skip-add).
  """
  factored  = True
  conv_head = True

  def __init__(self, input_dim, output_dim, n_amps,
               n_templates, int_dim_res, geom, kernel_size=11,
               channels=16, n_blocks=4, n_blocks_cnn=1,
               gate_init=0.1, template_mix=False, block_opts=None):
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
      geom         = full-whitening DataVectorGeometry; its evecs /
                     sqrt_ev define the basis-change buffers.
      kernel_size  = conv kernel width (odd); forwarded to CNNBlock.
      channels     = conv filter count; forwarded to CNNBlock.
      n_blocks     = residual blocks in the trunk.
      n_blocks_cnn = stacked CNN correction blocks (default 1).
      gate_init    = initial per-template correction scale. Small
                     (default 0.1) to start near the pure trunk;
                     not 0 -- a 0 gate strands the CNN with no
                     gradient, so it never learns.
      template_mix = False (default): one CNNBlock stack shared
                     identically across templates (fold into batch).
                     True: TemplateMixCNNBlock -- templates are the
                     conv channels; 3x less memory traffic, cross-
                     template features, no shared-kernel constraint.
      block_opts   = ResBlock options (None -> {}); its "act" is
                     also handed to the CNN head, so head and trunk
                     share one activation family (falls back to
                     activation_fcn, the paper's H).
    """
    super().__init__()
    if block_opts is None:
      block_opts = {}
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

    # conv head, in the chosen mode. Both take the trunk's
    # activation so head and trunk share one family.
    #   fold mode (default): one CNNBlock stack applied identically
    #     to every template (templates fold into the batch axis).
    #   mix mode: TemplateMixCNNBlock, templates as conv channels.
    self.template_mix = bool(template_mix)
    cnn_act = block_opts.get("act", activation_fcn)
    cnn = []
    for _ in range(n_blocks_cnn):
      if self.template_mix:
        cnn.append(TemplateMixCNNBlock(output_dim,
                                       n_templates=n_templates,
                                       kernel_size=kernel_size,
                                       channels=channels,
                                       act=cnn_act))
      else:
        cnn.append(CNNBlock(output_dim, kernel_size=kernel_size,
                            channels=channels, act=cnn_act))
    self.cnn = nn.ModuleList(cnn)

    # one learnable gate per template, (n_templates, 1) so it
    # broadcasts over (B, n_templates, n_keep).
    self.gate = nn.Parameter(
      torch.full((n_templates, 1), float(gate_init)))

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
    # trunk templates in the full-whitened basis.
    y = self.model(x[:, :self.n_in]).view(
      B, self.n_templates, self.n_keep)      # (B, T, n_keep)
    if self.template_mix:
      # templates as conv channels: the basis change broadcasts the
      # matmul over the leading (B, T) axes, so no fold is needed
      # and every head intermediate stays (B, channels, n_keep).
      h = y @ self.W_fd                       # (B, T, n_keep) theta
      for blk in self.cnn:
        h = blk(h)                            # cross-template conv
      return y + self.gate * (h @ self.W_df)
    # fold templates into the batch axis so the one shared conv
    # stack corrects all of them: view is free (the trunk output is
    # contiguous), and CNNBlock sees its usual (rows, n_keep).
    h = y.view(B * self.n_templates, self.n_keep) @ self.W_fd
    for blk in self.cnn:
      h = blk(h)                              # theta-aware correction
    # back to full-whitened, unfold, per-template gate, add.
    corr = (h @ self.W_df).view(B, self.n_templates, self.n_keep)
    return y + self.gate * corr
