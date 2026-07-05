"""NPCE losses: a frozen PCE base plus a neural refiner.

Both classes wrap a fitted, frozen PCEEmulator as the base prediction
under a trainable refiner network, differing in how base and refiner
combine: PCEResidualChi2 is additive in the whitened basis (the
refiner learns truth minus base), PCERatioChi2 multiplicative in the
physical basis (the refiner learns a fractional correction). Either
way the chi2 stays the plain masked Mahalanobis distance -- the base
moves only the target's zero point or scale, never the metric.

Both declare needs_params = True (encode / decode evaluate the base
from the whitened parameters), the capability flag the loaders
(batching.py), the training loop's compiled forward+loss and eval
twins (training.py), and the diagnostics already branch on; the
trim / focus / focus_scale reduction is inherited from CosmolikeChi2
(_reduce), so these losses ride every loss-side update for free.

PS: frozen = evaluated under no_grad, never trained; base = the
closed-form PCE prediction; refiner = the SGD-trained network (any
model spec) correcting it; Mahalanobis distance = r^T Cinv r, the
covariance-weighted squared residual.
"""

import torch

from ..loss_functions import CosmolikeChi2


class PCEResidualChi2(CosmolikeChi2):
  """
  NPCE integration: the refiner model learns the residual of the
  full whitened dv after a frozen PCE base, the chi2 staying
  plain. Mirrors ResidualBaseChi2, with the PCE base in place of
  the analytic center/R baseline.

  Encode (target construction, at load time):

      dv  (B, total_size)        raw data vector
         │  geom.encode          squeeze -> center -> whiten
         ▼
      t  (B, n_keep)             whitened truth
         │  - PCE(theta)         the frozen base (no_grad)
         ▼
      target (B, n_keep)         the refiner's residual target

  Loss: plain chi2 on (pred - target) -- the base cancels in the
  residual, (base + pred) - truth == pred - target, so the metric
  is exact. Decode inverts: geom.decode(y + PCE(theta)).

      (legend: B = batch rows; total_size = full 3x2pt dv length;
       n_keep = kept entries the model emulates; theta = the
       whitened parameters, params_whitened; PCE = the frozen
       PCEEmulator, evaluated in the whitened basis.)

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

  Speed design: the frozen base is precomputed once at load time
  and packed with the truth into the encoded target, so the chi2
  never re-runs the PCE in the training loop -- it just unpacks
  and forms b * (1 + delta). The loader stages the wider target
  via the target_dim attribute (batching.py reads it).

  Encode (at load, once per row):

      dv  (B, total_size)
         │  geom.squeeze         physical kept entries
         ▼
      xi  (B, n_keep)        b  (B, n_keep) = geom.decode(
         │                      │              PCE(theta))
         │                      │  the frozen base, physical
         └───────── cat ────────┘
         ▼
      target = [b ; xi]  (B, 2*n_keep)

  chi2 (hot path, per batch):

      target (B, 2*n_keep)
         │  unpack              b = target[:, :n_keep]
         │                      xi = target[:, n_keep:]
         ▼
      r = b*(1 + pred) - xi     pred = the fractional correction
         │  einsum r^T Cinv_sq r
         ▼
      c  (B,)                   per-sample chi2, no PCE recompute

      (legend: B = batch rows; total_size = full 3x2pt dv length;
       n_keep = kept entries; theta = the whitened parameters,
       params_whitened; Cinv_sq = the kept x kept sub-block of the
       inverse covariance.)

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
