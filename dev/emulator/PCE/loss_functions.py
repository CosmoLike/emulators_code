"""NPCE losses: a frozen PCE base plus a refiner."""

import torch

from ..loss_functions import CosmolikeChi2


class PCEResidualChi2(CosmolikeChi2):
  """
  NPCE integration: the refiner model learns the residual of the
  full whitened dv after a frozen PCE base, the chi2 staying
  plain. Mirrors ResidualBaseChi2, with the PCE base in place of
  the analytic center/R baseline:
    target = geom.encode(dv) - PCE(theta)       (encode)
    full   = PCE(theta) + model_output          (decode)
    chi2   = plain CosmolikeChi2 on the residual.

  The refiner is any model spec (ResMLP, ResCNN, ...) trained by
  run_emulator with the robust chi2 loss. It outputs the full dv
  correction, so it is not confined to the PCE's K-mode subspace --
  a too-small K only costs a smaller head start, never caps
  accuracy (the conservative high-T property). For a ResCNN
  refiner, pass a DiagonalGeometry geom (theta order), as in the
  standalone ResCNN run.

  needs_params = True: encode/decode take the whitened params
  (the model inputs) to evaluate the frozen PCE base.
  """
  needs_params = True

  def __init__(self, geom, pce):
    super().__init__(geom)
    self.pce = pce          # frozen PCE base (whitened dv)

  def _base(self, params_whitened):
    # frozen base -> no grad flows into the PCE.
    with torch.no_grad():
      return self.pce(params_whitened)

  def encode(self, dv, params_whitened):
    # whitened truth minus PCE base = the residual target.
    return self.geom.encode(dv) - self._base(params_whitened)

  def decode(self, y, params_whitened):
    # add the base back (whitened), then geometry decode.
    return self.geom.decode(y + self._base(params_whitened))

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    # plain: base baked into target, so pred - target ==
    # full_pred - truth. params accepted but unused.
    return CosmolikeChi2.chi2(self, pred=pred, target=target, full=full)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    # needs_params signature; the plain chi2 needs no params.
    return CosmolikeChi2.loss(
      self, pred, target, *args, **kwargs)


class PCERatioChi2(CosmolikeChi2):
  """
  Multiplicative ("1 + delta") NPCE: pred = b * (1 + delta) in
  physical (squeezed) dv space, b = geom.decode(PCE(theta)) the
  frozen base, delta the model output (fractional correction).
  Division-free; the chi2 is on (pred - truth) directly.

  Speed: the frozen base is precomputed once at load time and
  packed with the truth into the encoded target (encode returns
  [b ; xi], width 2*n_keep), so chi2 never re-runs the PCE in the
  training loop -- it just unpacks and forms b*(1+delta). (The
  loader stages a 2*n_keep-wide target via target_dim.)

  Trade-offs vs additive: target is not whitened; where b ~ 0
  (xi+/- zero crossings) the refiner has little leverage. Use a
  smooth, low-order PCE base.

  needs_params = True (encode/decode evaluate the base).
  """
  needs_params = True

  def __init__(self, geom, pce):
    super().__init__(geom)
    self.pce = pce

  @property
  def target_dim(self):
    # encode packs [base ; truth], so the loader stages a target
    # twice the kept-vector width.
    return 2 * self.geom.dest_idx.numel()

  def _base_phys(self, params_whitened):
    with torch.no_grad():
      return self.geom.decode(self.pce(params_whitened))

  def encode(self, dv, params_whitened):
    # precompute the frozen base (once per row at load) and pack it
    # with the physical truth (see Speed in the class doc).
    b  = self._base_phys(params_whitened)
    xi = self.geom.squeeze(dv).float()
    return torch.cat([b, xi], dim=1)         # (B, 2*n_keep)

  def decode(self, pred, params_whitened):
    # only used by the per-element diagnostics (not hot), so
    # recomputing the base here is fine.
    b = self._base_phys(params_whitened)
    return b * (1.0 + pred)

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    # unpack the cached base and truth -- no PCE recompute.
    nk  = self.geom.dest_idx.numel()
    b   = target[:, :nk]
    xi  = target[:, nk:]
    geo = self.geom
    r = b * (1.0 + pred) - xi
    # masked Mahalanobis (as in the base chi2) on the residual
    # r = b*(1+pred)-xi; unsqueeze to full length when full.
    if full:
      rf = geo.unsqueeze(r)
      return torch.einsum("bi,ij,bj->b", rf, geo.Cinv, rf)
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    # base is already in `target`; the plain chi2 needs no params.
    return CosmolikeChi2.loss(self, pred, target,
                              *args, **kwargs)
