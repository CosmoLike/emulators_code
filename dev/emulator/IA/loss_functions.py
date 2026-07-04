"""Factored intrinsic-alignment losses and amplitude coefficients."""

import torch

from ..loss_functions import CosmolikeChi2


def nla_coeffs(amps):
  """NLA amplitude polynomial coefficients.

  The NLA intrinsic-alignment field is linear in the one amplitude
  A1, so xi is exactly a quadratic in it: GG (no IA) + GI (linear
  in the field) + II (quadratic). The 3 templates, in this order
  (the model and loss must agree), are [GG, GI, II]; template 0
  (GG, coefficient 1) also carries the training center.

  Arguments:
    amps = (B, 1) physical amplitude [A1] per sample.

  Returns:
    (B, 3): the coefficients [1, A1, A1^2].
  """
  a1 = amps[:, 0:1]
  return torch.cat([
    torch.ones_like(a1),                      # GG
    a1,                                       # GI
    a1 * a1,                                  # II
  ], dim=1)                                   # (B, 3)


def tatt_coeffs(amps):
  """TATT amplitude polynomial coefficients.

  The TATT intrinsic-alignment field is a1*O1 + a2*O2 +
  a1*b_TA*O1d, so xi is exactly a polynomial in (a1, a2, b_TA):
  GG (no IA) + GI (linear in the field) + II (quadratic). The 10
  templates, in this order (the model and loss must agree), are
    [GG, GI1, GI2, GI1d, II11, II22, II1d1d, II12, II11d, II21d].

  Arguments:
    amps = (B, 3) physical amplitudes [a1, a2, b_TA] per sample,
           in this order.

  Returns:
    (B, 10): the coefficients
      GG    -> 1
      GI1   -> a1         GI2 -> a2       GI1d  -> a1*b_TA
      II11  -> a1^2       II22 -> a2^2    II1d1d-> (a1*b_TA)^2
      II12  -> a1*a2      II11d-> a1^2*b_TA
      II21d -> a1*a2*b_TA
  """
  a1  = amps[:, 0:1]
  a2  = amps[:, 1:2]
  bta = amps[:, 2:3]
  a1b = a1 * bta                              # a1 * b_TA
  return torch.cat([
    torch.ones_like(a1),                      # GG
    a1, a2, a1b,                              # GI
    a1 * a1, a2 * a2, a1b * a1b,              # II diagonal
    a1 * a2, a1 * a1b, a2 * a1b,              # II cross
  ], dim=1)                                   # (B, 10)


class NLAAmpFactoredChi2(CosmolikeChi2):
  """
  Factored NLA loss. The model outputs three whitened templates
  [GG, GI, II]; this loss reads each sample's IA amplitude A1_1
  (the appended last column of the encoded params) and combines
  them in closed form, xi = GG + A1_1*GI + A1_1^2*II, then scores
  the standard chi2 on the combined xi. The A1_1 dependence is
  imposed, not learned, and A1_1 never enters the network -- so it
  generalizes perfectly, free at inference, prior-width-
  independent.

  The combination is linear, so it commutes with the whitening
  (combining whitened templates == whitening the combined xi), and
  the center is absorbed into the GG template automatically (the
  net learns whatever matches geom.encode(xi)). A1_1 is physical
  (A1_1 = 0 is the no-IA limit). Trained on the existing
  (cosmo, A1_2, A1_1) -> xi samples -- A1_1 is read per sample, not
  extracted.

  needs_params = True (the loss needs A1_1).
  """
  needs_params = True
  _params = None

  def encode(self, dv, params_whitened):
    """Raw dv -> whitened xi target the combination must match.

    Arguments:
      dv              = (B, total_size) raw full data vectors, the
                        xi observed at each sample's own A1_1.
      params_whitened = (B, n_param) encoded params; not used to
                        build the target (the standard whitened
                        xi), accepted only to match the param-aware
                        encode signature.

    Returns:
      (B, n_keep): the whitened xi target.
    """
    return self.geom.encode(dv)

  def _combine(self, pred, params_whitened):
    """Apply the exact NLA amplitude polynomial to the templates.

    Arguments:
      pred            = (B, 3, n_keep) whitened templates
                        [GG, GI, II].
      params_whitened = (B, n_param) encoded params, last column
                        the physical A1_1.

    Returns:
      (B, n_keep): whitened xi = GG + A1_1*GI + A1_1^2*II per
      sample.
    """
    a1 = params_whitened[:, -1:]                # (B, 1) physical
    GG, GI, II = pred[:, 0], pred[:, 1], pred[:, 2]
    return GG + a1 * GI + (a1 * a1) * II         # (B, n_keep)

  def decode(self, pred, params_whitened):
    """Templates -> physical xi (for the per-element diagnostics).

    Arguments:
      pred            = (B, 3, n_keep) whitened templates.
      params_whitened = (B, n_param) encoded params (A1_1 last
                        column).

    Returns:
      (B, n_keep): the physical squeezed xi (combine the
      templates, un-whiten, add the center back).
    """
    return self.geom.decode(
      self._combine(pred=pred, params_whitened=params_whitened))

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    """Per-sample chi2 of the combined xi against the target.

    Arguments:
      pred            = (B, 3, n_keep) whitened templates.
      target          = (B, n_keep) whitened xi (from encode).
      params_whitened = (B, n_param) encoded params (A1_1 last
                        column), or None to use the loss() stash.
      full            = if True, the full-Cinv reference path;
                        else the fast masked-block path.

    Returns:
      (B,): per-sample chi2, combined xi vs target.
    """
    if params_whitened is None:
      params_whitened = self._params
    w = self._combine(pred=pred, params_whitened=params_whitened)
    return CosmolikeChi2.chi2(self, pred=w, target=target, full=full)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    """Scalar training loss from the combined-xi chi2.

    Stashes params so the inherited reduction (which calls
    self.chi2 without params) can recover A1_1.

    Arguments:
      pred            = (B, 3, n_keep) whitened templates.
      target          = (B, n_keep) whitened xi.
      params_whitened = (B, n_param) encoded params (A1_1 last
                        column).
      *args, **kwargs = reduction knobs forwarded to the base
                        loss (mode, trim, focus, focus_scale).

    Returns:
      the scalar training loss.
    """
    self._params = params_whitened
    return CosmolikeChi2.loss(self, pred, target,
                              *args, **kwargs)


class TemplateFactoredChi2(CosmolikeChi2):
  """
  Factored IA loss. The model outputs n_templates whitened
  templates; this loss reads each sample's IA amplitudes (the
  appended last n_amps columns), builds the coefficients via
  coeff_fn, combines the templates in closed form
  (xi = sum_t c_t * template_t), and scores the standard chi2 on
  the combined xi.

  The amplitude dependence is imposed, not learned; the amplitudes
  never enter the network, so the emulator generalizes perfectly in
  them, free at inference, and the amplitude prior costs zero
  training coverage -- the win that grows from NLA's 1 amplitude to
  TATT's coupled, wide-prior 3. The combine is linear, so it
  commutes with the whitening; the center is absorbed into the GG
  (constant-coefficient) template. Amplitudes are physical (all
  zero = the no-IA limit); coeff_fn defines the template order.

    pred (B, T, n_keep)          whitened templates (the model)
       │
       │    A (B, n_amps)        raw amplitudes, sliced off the
       │       │                 END of the encoded params
       │       │  coeff_fn       the closed-form polynomial
       │       ▼
       │    c (B, T)             per-template coefficients
       │  ─────┘
       │  combine: xi = sum_t c_t * template_t   (einsum over t)
       ▼
    xi (B, n_keep)               whitened combined prediction
       │  CosmolikeChi2.chi2 / .loss against the whitened target
       ▼
    per-sample chi2 (B,)  /  scalar training loss

  (legend: B = batch rows; T = n_templates (3 nla / 10 tatt);
  n_keep = kept dv length, one template's width; n_amps = appended
  amplitude columns (1 nla / 3 tatt); coeff_fn = nla_coeffs /
  tatt_coeffs above.)

  needs_params = True (the loss needs the amplitudes).
  """
  needs_params = True
  _params = None

  def __init__(self, geom, coeff_fn, n_amps):
    """Hold the geometry, the amplitude polynomial, and n_amps.

    Arguments:
      geom     = DataVectorGeometry for the combined xi (its
                 whitening / Cinv score the chi2).
      coeff_fn = callable (B, n_amps) physical amplitudes ->
                 (B, n_templates) coefficients (nla_coeffs,
                 tatt_coeffs).
      n_amps   = number of appended amplitude columns to read from
                 the end of the encoded params.
    """
    super().__init__(geom)
    self.coeff_fn = coeff_fn
    self.n_amps   = n_amps

  def encode(self, dv, params_whitened):
    """Raw dv -> whitened xi target the combination must match.

    Arguments:
      dv              = (B, total_size) raw full data vectors, the
                        xi observed at each sample's own
                        amplitudes.
      params_whitened = (B, n_param) encoded params; not used to
                        build the target, accepted to match the
                        param-aware encode signature.

    Returns:
      (B, n_keep): the whitened xi target.
    """
    return self.geom.encode(dv)

  def _combine(self, pred, params_whitened):
    """Apply the amplitude polynomial to the templates.

    Arguments:
      pred            = (B, n_templates, n_keep) whitened
                        templates.
      params_whitened = (B, n_param) encoded params, last n_amps
                        columns the physical amplitudes.

    Returns:
      (B, n_keep): the whitened xi = sum_t c_t * template_t.
    """
    amps = params_whitened[:, -self.n_amps:]   # (B, n_amps)
    c    = self.coeff_fn(amps)                 # (B, n_templates)
    # operands in subscript order: c (b,t) = amplitude
    # coefficients, pred (b,t,k) = the t templates; sums over t
    # to the combined xi (b,k).
    return torch.einsum("bt,btk->bk", c, pred)

  def decode(self, pred, params_whitened):
    """Templates -> physical xi (for the per-element diagnostics).

    Arguments:
      pred            = (B, n_templates, n_keep) whitened
                        templates.
      params_whitened = (B, n_param) encoded params (amplitudes
                        last n_amps columns).

    Returns:
      (B, n_keep): the physical squeezed xi.
    """
    return self.geom.decode(
      self._combine(pred=pred, params_whitened=params_whitened))

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    """Per-sample chi2 of the combined xi against the target.

    Arguments:
      pred            = (B, n_templates, n_keep) whitened
                        templates.
      target          = (B, n_keep) whitened xi (from encode).
      params_whitened = (B, n_param) encoded params (amplitudes
                        last n_amps columns), or None to use the
                        loss() stash.
      full            = if True, the full-Cinv reference path;
                        else the fast masked-block path.

    Returns:
      (B,): per-sample chi2, combined xi vs target.
    """
    if params_whitened is None:
      params_whitened = self._params
    w = self._combine(pred=pred, params_whitened=params_whitened)
    return CosmolikeChi2.chi2(self, pred=w, target=target, full=full)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    """Scalar training loss from the combined-xi chi2.

    Stashes params so the inherited reduction (which calls
    self.chi2 without params) can recover the amplitudes.

    Arguments:
      pred            = (B, n_templates, n_keep) whitened
                        templates.
      target          = (B, n_keep) whitened xi.
      params_whitened = (B, n_param) encoded params (amplitudes
                        last n_amps columns).
      *args, **kwargs = reduction knobs (mode, trim, focus,
                        focus_scale).

    Returns:
      the scalar training loss.
    """
    self._params = params_whitened
    return CosmolikeChi2.loss(self, pred, target,
                              *args, **kwargs)
