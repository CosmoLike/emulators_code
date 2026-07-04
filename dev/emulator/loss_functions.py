"""Chi2 losses and the robustness annealing schedule.

The loss layer: each class holds a DataVectorGeometry (composition, not
inheritance) and adds the chi2 (the masked Mahalanobis distance
r^T Cinv r per sample) and the training loss on it. CosmolikeChi2 is the
plain loss (trimming, a focal hardness weight, a sqrt / pseudo-Huber
transform). RescaledChi2 and ResidualBaseChi2 are the two analytic-R
variants (R divides the net output, versus R moves only the baseline).
ElementWeightedChi2 up-weights the worst-fit dv elements. anneal_value is
the per-epoch trim / focus schedule; make_chi2 builds the right loss from
a geometry and a rescale mode.

PS: the Mahalanobis distance r^T Cinv r is a squared residual r weighted
by the inverse covariance Cinv (this is the chi2); "masked" means only
the unmasked data-vector entries the analysis keeps enter the sum.
"""

import numpy as np
import torch

from .analytics import _analytic_R


def anneal_value(epoch, opts):
  """Value of an annealed robustness knob at a given epoch.

  Holds opts["start"] for the first hold_epochs, ramps toward
  opts["end"] over the next anneal_epochs, then stays at end.
  shape picks the schedule:
    "const"  -- fixed at start forever (no annealing); the
                fixed-trim baseline. end / hold_epochs /
                anneal_epochs are ignored.
    "linear" -- straight ramp start -> end.
    "cosine" -- smooth ease, zero slope at both ends, avoiding
                the abrupt loss jumps a discrete schedule causes
                (those can mislead a reactive ReduceLROnPlateau).
    "step"   -- the linear ramp floored to a 0.01 grid, the
                literal 5% -> 4% -> 3% drop.

  Arguments:
    epoch = current epoch (1-based, as in the loop).
    opts  = dict with start, end, hold_epochs, anneal_epochs,
            shape (end / hold_epochs / anneal_epochs unused when
            shape is "const").

  Returns:
    the knob value at this epoch (float).
  """
  shape = opts["shape"]
  start = opts["start"]
  # constant schedule: hold `start` every epoch (baseline).
  if shape == "const":
    return float(start)

  end  = opts["end"]
  hold = opts["hold_epochs"]
  span = max(1, opts["anneal_epochs"])
  # before the ramp begins: hold the start value.
  if epoch <= hold:
    return float(start)

  # fraction along the ramp, clamped to [0, 1].
  t = min(1.0, (epoch - hold) / span)
  if shape == "cosine":
    # cosine ease runs 1 -> 0, taking value start -> end smoothly.
    ease = 0.5 * (1.0 + np.cos(np.pi * t))
    return float(end + (start - end) * ease)

  # linear value (also the base for the stepped grid).
  val = start + (end - start) * t
  if shape == "step":
    # floor to a 0.01 grid (5% -> 4% -> ...); `end` is the floor
    # it never drops below.
    val = max(end, np.floor(val * 100.0) / 100.0)
  return float(val)


class CosmolikeChi2:
  """
  Adds the chi2 and the training loss to a geometry.

  Composition, not inheritance: a CosmolikeChi2 holds a
  DataVectorGeometry (self.geom) rather than being one, so one
  geometry (built once -- one cosmolike read, one
  eigendecomposition) is shared by several loss variants (plain,
  rescaled, element-weighted) without rebuilding. self.geom owns
  the masked-dv geometry and transforms; this class adds the chi2
  (the per-sample Mahalanobis distance r^T Cinv r) and the loss on
  it. The loss trims the worst `trim` fraction of the batch before
  averaging, so a few contaminated data vectors cannot dominate
  the gradient; any per-sample transform (e.g. sqrt) is applied
  after the trim.

    pred, target  (B, out_dim)   both in the whitened eigenbasis
       │  r = unwhiten(pred - target)     whitened -> physical
       ▼
    r  (B, out_dim)              physical residual, kept entries
       │  einsum r^T Cinv_sq r   (full=True: unsqueeze r to the
       │                         full dv, contract the full Cinv
       │                         -- the slow reference path)
       ▼
    c  (B,)                      per-sample chi2
       │  loss() only: drop the worst `trim` fraction (topk) ->
       │  mode transform (chi2 | sqrt | sqrt_dchi2) -> focal
       │  weights -> normalized weighted mean
       ▼
    scalar training loss

  (legend: B = batch rows; out_dim = kept dv length, the unmasked
  entries the model emulates; Cinv_sq = the kept x kept sub-block
  of the inverse covariance, Cinv the full one; unwhiten = the
  geometry's eigenbasis -> physical map, applicable to the residual
  directly because whitening is linear.)
  """

  def __init__(self, geom):
    """Hold the geometry the chi2 contracts against.

    Arguments:
      geom = DataVectorGeometry for this probe; owns the
             whitening basis, Cinv / Cinv_sq, dest_idx,
             total_size, center, and every dv transform.
    """
    self.geom = geom

  # --- thin delegation to the geometry ---
  # The pipeline (loaders, run_emulator) reads a few geometry
  # quantities off the chi2 object. Forward them to self.geom
  # so those call sites are unchanged: only how chi2fn is built
  # differs, not how it is used.
  @property
  def dest_idx(self):
    return self.geom.dest_idx

  @property
  def total_size(self):
    return self.geom.total_size

  def encode(self, dv):
    return self.geom.encode(dv)

  def decode(self, whitened_sq):
    return self.geom.decode(whitened_sq)

  def chi2(self, pred, target, full=False):
    """
    Per-sample chi2 = r^T Cinv r, two equal ways (the chi2 sanity
    test proves they match; pass 0/1 or False/True):
      full=False (0, default) -- contract the squeezed residual
        with the masked sub-block Cinv_sq (out_dim x out_dim).
        Fast: no unsqueeze, small einsum. Needs geom.Cinv_sq.
      full=True  (1) -- unsqueeze the residual to the full vector
        and contract with the full Cinv (total_size). The slower
        reference; masked entries contribute 0.
    """
    # geo = self.geom (unwhiten / unsqueeze / Cinv live there).
    geo = self.geom
    if full:
      # reference path: full-length residual + full Cinv.
      r = geo.unsqueeze(geo.unwhiten(pred - target))
      # einsum operands are positional, the subscripts "bi,ij,bj->b"
      # naming them in order: r (b,i) and r (b,j) are the residual,
      # geo.Cinv (i,j) the full precision; contracts to chi2 (b,).
      return torch.einsum("bi,ij,bj->b", r, geo.Cinv, r)
    # fast path: squeezed residual + masked sub-block Cinv_sq.
    r = geo.unwhiten(pred - target)
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, mode="sqrt", trim=0.05,
           focus=0.0, focus_scale=1.0):
    """
    Scalar training loss from the per-sample chi2.

    Trims the worst `trim` fraction of the batch (a hard reject:
    robust to contamination, but it hides genuinely hard regions,
    so evaluation never trims), then averages a per-sample
    transform chosen by `mode`.

    Arguments:
      pred   = network outputs, whitened space (B, out_dim).
      target = whitened targets, same shape.
      mode   = per-sample transform before the mean:
               "chi2"       -> c
               "sqrt"       -> sqrt(c)
               "sqrt_dchi2" -> sqrt(1+2c)-1 (pseudo-Huber)
      trim   = fraction of the worst (largest-chi2) samples to
               drop before averaging; 0 disables trimming.
      focus  = focal weight exponent gamma. <= 0 -> plain mean;
               > 0 -> weight each sample by
               (c/(c+focus_scale))**focus (detached), up-weighting
               hard points so the optimizer keeps chasing the tail.
               Annealed. Float or 0-dim tensor (see _reduce).
      focus_scale = chi2 scale where the focal weight turns on;
                    hardness h = c/(c+focus_scale) crosses 0.5 at
                    c = focus_scale.
    Returns:
      a scalar loss tensor (the trimmed, focal-weighted mean).
    """
    c = self.chi2(pred=pred, target=target)   # per-sample chi2, (B,)
    return self._reduce(c=c, mode=mode, trim=trim, focus=focus,
                        focus_scale=focus_scale)

  def _reduce(self, c, mode, trim, focus, focus_scale):
    """
    Per-sample chi2 -> scalar loss: trim, transform, focal mean.
    Shared by every loss variant (the subclasses change how c is
    built, never how it is reduced).

      c  (B,)                per-sample chi2
         │  sort ascending          (static shape; see below)
         ▼
         │  keep = 1 for the first k entries, 0 after
         │                          k = round((1-trim)*B), >= 1
         ▼
         │  v = mode transform      (chi2 | sqrt | sqrt_dchi2)
         │  w = keep * (c/(c+focus_scale))**max(focus,0)  detached
         ▼
      loss = (w*v).sum() / (w.sum() + eps)

    Why sort + a zero-weight prefix mask instead of topk(c, k):
    the value is identical -- the kept set is the same k smallest
    samples, and a weighted mean does not care about order -- but
    topk's output shape depends on k, and k anneals every epoch.
    Under torch.compile that would recompile (and re-capture CUDA
    graphs) once per epoch; the sorted-mask form keeps every shape
    fixed at B, so one compiled graph serves a whole annealed run.
    For the same reason trim and focus accept 0-dim tensors: a
    Python float is guarded by value (new value = recompile), a
    tensor updates in place. Gradients match topk's exactly: the
    dropped samples' terms are multiplied by 0, and sort is a
    permutation (gradients pass straight through it).

    Arguments:
      c           = (B,) per-sample chi2.
      mode        = "chi2" | "sqrt" | "sqrt_dchi2" (a static
                    string: one specialization per mode).
      trim        = worst fraction dropped (0 = none); float or
                    0-dim tensor.
      focus       = focal exponent gamma; float or 0-dim tensor
                    (<= 0 -> plain mean, since h**0 = 1).
      focus_scale = focal turn-on scale, fixed per run; float or
                    0-dim tensor (the training loop passes a
                    device tensor -- a closure float is
                    torch-version-dependent under compile and can
                    surface as a CPU input that breaks CUDA-graph
                    replay).

    Returns:
      a scalar loss tensor.
    """
    B = c.numel()
    # ascending sort: the kept (smallest-chi2) samples come first,
    # so the keep mask is a prefix. Sorting when trim = 0 changes
    # nothing -- the weighted mean is permutation-invariant.
    c = torch.sort(c).values
    # keep-count k = round((1-trim)*B), floored at 1 so a tiny
    # batch never keeps zero samples. Computed in float64 so a
    # tensor trim rounds exactly as Python's round() did in the
    # old topk path (both round half to even).
    trim_t = torch.as_tensor(trim, dtype=torch.float64,
                             device=c.device)
    k = torch.clamp(torch.round((1.0 - trim_t) * B), min=1.0)
    # prefix mask: position i is kept iff i < k. arange is int64,
    # the comparison against the 0-dim float k broadcasts; cast to
    # the chi2 dtype for the weighted mean.
    keep = (torch.arange(B, device=c.device) < k).to(c.dtype)

    # per-sample transformed loss (not yet averaged)
    if mode == "chi2":
      v = c
    elif mode == "sqrt":
      v = torch.sqrt(c)
    elif mode == "sqrt_dchi2":
      v = torch.sqrt(1.0 + 2.0 * c) - 1.0
    else:
      raise ValueError(f"unknown loss mode: {mode}")

    # focal hardness weight: h = c/(c+focus_scale) in [0,1) is a
    # soft "is this point hard?" (0 for c<<scale, ->1 for c>>scale);
    # h**gamma sharpens it. detach() freezes the weight as a
    # priority, so the optimizer cannot lower the loss by shrinking
    # a point's weight instead of fitting it. gamma = max(focus, 0):
    # a negative focus (the "off" sentinel) clamps to 0 and h**0 = 1
    # everywhere, collapsing the weighted mean to the plain mean --
    # no fragile "focus == 0" test or special case.
    gamma = torch.clamp(
      torch.as_tensor(focus, dtype=c.dtype, device=c.device),
      min=0.0)
    h = (c / (c + focus_scale)).detach()
    # the trim enters as a weight: dropped samples get w = 0, so
    # both sums below run over the kept prefix only -- the same
    # numerator and normalizer as the old topk form.
    w = keep * h ** gamma
    # normalized weighted mean (stable scale as w anneals).
    return (w * v).sum() / (w.sum() + 1e-12)


class RescaledChi2(CosmolikeChi2):
  """
  CosmolikeChi2 with an analytic per-element rescaling R of the
  target: the network emulates the reshaped dv (dv*R, flatter
  across cosmologies), but the chi2 stays on the original
  physical dv -- R is divided back out of the residual, leaving
  the covariance and reported chi2 unchanged. R = 1 recovers the
  base class exactly.

  R is never stored: a deterministic function of the cosmological
  params, recomputed on-device from the whitened model-input params
  this class is handed (decoded back to physical via the param
  geometry). Two consumers, one source -- encode builds the target
  with R, chi2 undoes R, both calling _R on the same params, so
  they share a bit-identical R and no (N_rows, n_keep) array ever
  exists.

  A subclass, so the base (no-reshape) path is untouched and the
  two are A/B-swappable. Build by wrapping a geometry --
  RescaledChi2(geom); then call build_shear_angle_map(geom) and
  configure_rescaling to attach the rescale state before training.
  """

  # per-batch stash so the inherited loss reduction (which calls
  # self.chi2(pred=pred, target=target) without params) finds them.
  _params  = None
  # kept-element geometry tensors, built lazily in _R.
  _theta_t = None
  _zeff_t  = None
  # capability flag: this loss's encode/decode/chi2/loss take the
  # whitened params (to build R). The pipeline branches on
  # getattr(chi2fn, "needs_params", False) instead of isinstance,
  # so a future param-aware loss only has to set this True -- it
  # need not subclass RescaledChi2.
  needs_params = True
    
  def configure_rescaling(self, param_geometry, cosmo_mid,
                          names, include_amp=True,
                          u_star=0.5):
    """
    Attach the analytic-rescaling state (call once, after
    wrapping the geom and build_shear_angle_map).

    Arguments:
      param_geometry = ParamGeometry whose decode maps the
                       whitened model inputs back to physical
                       params (what R reads); the same object
                       passed to run_emulator.
      cosmo_mid      = (n_param,) reference cosmology where R = 1,
                       typically the training-cloud mean
                       train_set["C"][train_set["idx"]].mean(0).
      names          = parameter column names (pgeom.names).
      include_amp    = pass the (Om h^2)^ns/h amplitude factor
                       to _analytic_R (standard run: True).
      u_star         = kernel-peak lens position (~0.5).
    """
    # build_shear_angle_map(geom) must run first -- _R reads
    # these off geom. Fail loudly if the order is wrong.
    for a in ("theta_kept", "zsrc_i", "zsrc_j"):
      assert hasattr(self.geom, a), (
        "call build_shear_angle_map(geom) before "
        f"configure_rescaling (missing {a})")
        
    self.param_geometry = param_geometry
    self.cosmo_mid   = cosmo_mid
    self.names       = list(names)
    self.include_amp = include_amp
    self.u_star      = u_star
    # drop any stale geometry cache (rebuilt on next _R).
    self._theta_t = None
    self._zeff_t  = None
    return self

  def _R(self, params_whitened):
    """
    Per-(row, element) rescaling R for whitened model inputs.

    Decodes the whitened params to physical (the form _analytic_R
    reads), then evaluates the analytic R on the kept-element
    geometry (theta_kept and the cross-pair z_eff = min(z_i, z_j))
    on the params' device. The geometry tensors are built once and
    cached.

    Arguments:
      params_whitened = (B, n_param) whitened model inputs (the
                        same tensor the model consumes).
    Returns:
      R = (B, n_keep) float tensor on the params' device.
    """
    geo  = self.geom
    phys = self.param_geometry.decode(params_whitened)
    dev  = phys.device
    # build the device geometry tensors once (build_shear_angle_
    # map must have run to set theta_kept / zsrc_*).
    if (self._theta_t is None
        or self._theta_t.device != dev):
      zeff = np.minimum(geo.zsrc_i, geo.zsrc_j)
      self._theta_t = torch.as_tensor(
        geo.theta_kept, dtype=torch.float32, device=dev)
      self._zeff_t = torch.as_tensor(
        zeff, dtype=torch.float32, device=dev)
    return _analytic_R(theta_arcmin=self._theta_t,
                       z_eff=self._zeff_t,
                       cosmo=phys,
                       cosmo_mid=self.cosmo_mid,
                       names=self.names,
                       u_star=self.u_star,
                       include_amp=self.include_amp)

  def encode(self, dv, params_whitened):
    # squeeze -> apply the analytic rescaling -> center +
    # whiten. params_whitened gives R per row and element.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.whiten(geo.squeeze(dv) * R - geo.center)

  def decode(self, y, params_whitened):
    # network output -> reshaped dv -> physical dv (/ R).
    geo = self.geom
    R = self._R(params_whitened)
    return (geo.unwhiten(y).float() + geo.center) / R

  def chi2(self, pred, target, params_whitened=None,
               full=False):
    # residual in reshaped-whitened space, divided by R back to a
    # physical squeezed residual, then the usual masked Mahalanobis
    # (the center cancels in pred - target). params_whitened=None
    # -> use the stash set by loss().
    if params_whitened is None:
      params_whitened = self._params
    if params_whitened is None:
      raise RuntimeError(
        "RescaledChi2.chi2 needs the whitened params: pass "
        "them, or call via loss() which stashes them")

    geo = self.geom
    R = self._R(params_whitened)
    r = geo.unwhiten(pred - target) / R

    if full:
      r = geo.unsqueeze(r)
      # einsum as in the base chi2; r here is the residual / R.
      return torch.einsum("bi,ij,bj->b", r, geo.Cinv, r)
    return torch.einsum("bi,ij,bj->b", r, geo.Cinv_sq, r)

  def loss(self, pred, target, params_whitened,
           *args, **kwargs):
    # stash params for the inherited reduction, which calls
    # self.chi2(pred=pred, target=target) -> picks up self._params.
    # No copy of the base loss body.
    self._params = params_whitened
    return super().loss(pred, target, *args, **kwargs)


class ResidualBaseChi2(RescaledChi2):
  """
  Analytic baseline as a residual base (the "B" form), to test
  the conditioning question against RescaledChi2 (the "A" form)
  with everything else held fixed.

  Both use the same analytic R. The difference is where R enters
  the network's reconstruction d_pred (u = unwhiten of the net
  output, c = center):
    A (RescaledChi2):  d_pred = (u + c) / R  -- R divides the
        net output, so the chi2 gradient carries diag(1/R), a
        per-cosmology conditioning factor.
    B (this class):    d_pred =  u + c / R   -- R moves only the
        constant baseline c -> c/R; the net output enters at unit
        gain, so the chi2 is plain (no /R, no conditioning
        factor).
  So B puts R in the target, never in the loss: it overrides
  encode (c -> c/R) and decode but not chi2 -- inheriting the plain
  CosmolikeChi2 chi2.

  Reuses RescaledChi2's R machinery (_R, configure_rescaling, the
  _params stash, loss). Build and configure exactly like
  RescaledChi2: wrap a geom, build_shear_angle_map(geom), then
  configure_rescaling(...).
  """

  def encode(self, dv, params_whitened):
    # target = whiten(squeeze(dv) - center/R): the plain encode with
    # the constant baseline center swapped for the analytic-moved
    # center/R. R is baked into the target, so chi2 needs no R.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.whiten(geo.squeeze(dv) - geo.center / R)

  def decode(self, y, params_whitened):
    # physical dv = unwhiten(y) + center/R (baseline added back).
    # The net output enters at unit gain -- no /R.
    geo = self.geom
    R = self._R(params_whitened)
    return geo.unwhiten(y).float() + geo.center / R

  def chi2(self, pred, target, params_whitened=None,
           full=False):
    # plain chi2: R is already in the target, so do not divide it
    # out of the residual -- that absent /R is the whole point vs
    # the A form. params_whitened is accepted (the loader and eval
    # pass it, since this is a RescaledChi2 subclass) but ignored.
    return CosmolikeChi2.chi2(self, pred=pred, target=target, full=full)


class ElementWeightedChi2(CosmolikeChi2):
  """
  CosmolikeChi2 with a per-element focal weight in the training
  loss (no rescaling -- isolates the per-element weight from the
  analytic R, to test one thing at a time).

  Each dv element's residual is scaled by a detached factor >= 1
  before the chi2 sums over elements, so the network spends
  accuracy on the elements it currently fits worst in error-bar
  units -- the tight-covariance, most-constraining block. Mirrors
  the per-sample focal but over elements:
    hardness e_i = batch-mean marginal chi2 of element i,
    scale_i      = sqrt(1 + beta * (e/(e+kappa))**gamma).
  Easy elements keep scale 1 (never zeroed -- they sit near
  budget too). The inherited chi2 is unchanged, so eval reports
  the true (unweighted) chi2; only the training loss is shaped.
  """

  _elem_kappa  = 0.01
  _elem_gamma  = 1.0
  _elem_beta   = 4.0
  _sigma_cache = None

  def set_elem_weight(self, kappa=0.01, gamma=1.0, beta=4.0):
    """
    Set the per-element focal knobs (call once before training).

    Arguments:
      kappa = marginal-chi2 scale where an element counts as
              hard; e/(e+kappa) crosses 0.5 at e = kappa. e is
              in (residual/sigma)**2 units, so kappa ~ 0.01 is
              an element off by ~0.1 sigma.
      gamma = hardness sharpness (the focal exponent).
      beta  = boost strength; the hardest elements get a chi2
              weight up to 1 + beta.
    """
    self._elem_kappa = kappa
    self._elem_gamma = gamma
    self._elem_beta  = beta
    return self

  def _elem_sigma(self):
    # per-element marginal error bar sqrt(diag(cov)), cached.
    # cov = U diag(ev) U^T -> diag_i = sum_k (U_ik sqrt(ev_k))^2.
    if self._sigma_cache is None:
      self._sigma_cache = torch.sqrt(
        ((self.geom.evecs * self.geom.sqrt_ev) ** 2).sum(1))
    return self._sigma_cache

  def loss(self, pred, target, mode="sqrt", trim=0.05,
           focus=0.0, focus_scale=1.0):
    """
    Training loss with a per-element focal weight on the chi2.

    Same shape as CosmolikeChi2.loss (trim, mode transform,
    per-sample focal), but the per-sample chi2 is built from a
    per-element-weighted residual (hard elements scaled up). Eval
    calls the inherited self.chi2, so the reported metric is the
    true unweighted chi2.

    Arguments:
      pred, target = whitened outputs / targets (B, out_dim).
      mode   = "chi2" / "sqrt" / "sqrt_dchi2".
      trim   = fraction of worst samples dropped; 0 off.
      focus  = per-sample focal exponent (<=0 -> plain mean).
      focus_scale = per-sample focal turn-on scale.
    Returns:
      a scalar loss tensor.
    """
    # per-element focal (see class doc): scale each element's
    # residual by a detached factor >= 1 from its batch-mean
    # marginal chi2. No rescaling -- residual = unwhiten(pred-target).
    r = self.geom.unwhiten(pred - target)       # (B, n_keep)
    z = r / self._elem_sigma()                  # marginal resid
    e = (z * z).mean(0).detach()                # element hardness
    hard  = e / (e + self._elem_kappa)          # in [0,1)
    scale = torch.sqrt(
      1.0 + self._elem_beta * hard ** self._elem_gamma)
    rs = r * scale
    # masked Mahalanobis (as in the base chi2) on the element-
    # weighted residual rs; contracts to per-sample chi2 (b,).
    c = torch.einsum("bi,ij,bj->b", rs, self.geom.Cinv_sq, rs)
    # the reduction (trim / mode transform / focal mean) is the
    # base class's, in its static-shape form -- see _reduce.
    return self._reduce(c=c, mode=mode, trim=trim, focus=focus,
                        focus_scale=focus_scale)


def make_chi2(geom, rescale="none", param_geometry=None,
              cosmo_mid=None, data_dir="lsst_y1",
              dataset="lsst_y1_M1_GGL0.05.dataset",
              include_amp=True):
  """
  Build the chi2fn (loss + geometry wrapper), optionally rescaled.

  The analytic rescaling divides out a fast linear reference R
  (E&H zero-baryon, single-plane Limber) so the network emulates a
  flatter target; the chi2 is always reported on the original
  physical dv. The two variants share R and differ only in where R
  enters d_pred, hence whether R lands in the loss gradient (full
  derivation in the class docstrings):

    rescale = "none"     -> plain CosmolikeChi2 (no R).
              "rescaled" -> RescaledChi2 (v1, "A" form): R divides
                            the net output, so the chi2 gradient
                            carries a per-cosmology diag(1/R)
                            conditioning factor.
              "residual" -> ResidualBaseChi2 (v2, "B" form): R
                            moves only the baseline; the net enters
                            at unit gain and the chi2 is plain
                            (no /R) -- clean prior isolation.

  Both variants need the per-element angle/tomography map on the
  geometry (build_shear_angle_map, imported lazily so a plain build
  does not pull in the cosmolike-importing geometry module) and the
  analytic config (configure_rescaling).

  Arguments:
    geom           = DataVectorGeometry for the probe (e.g. xi).
    rescale        = "none" / "rescaled" / "residual" (see above).
    param_geometry = ParamGeometry whose decode maps the whitened
                     model inputs back to physical params (what R
                     reads); required when rescale != "none".
    cosmo_mid      = (n_param,) reference cosmology where R = 1,
                     typically the training-cloud mean; required
                     when rescale != "none".
    data_dir       = cosmolike data folder for the angle map.
    dataset        = .dataset ini for the angle map.
    include_amp    = pass the (Om h^2)^ns/h amplitude factor to the
                     analytic R (standard run: True).

  Returns:
    a CosmolikeChi2 (or RescaledChi2 / ResidualBaseChi2) to pass to
    run_emulator as chi2fn.
  """
  if rescale == "none":
    return CosmolikeChi2(geom=geom)
  # lazy import: build_shear_angle_map lives in the cosmolike-
  # importing geometry module, only needed for the rescaled path.
  from .geometries_output import build_shear_angle_map
  build_shear_angle_map(geom=geom, data_dir=data_dir,
                        dataset=dataset)
  cls = RescaledChi2 if rescale == "rescaled" else ResidualBaseChi2
  chi2fn = cls(geom=geom)
  chi2fn.configure_rescaling(param_geometry=param_geometry,
                             cosmo_mid=cosmo_mid,
                             names=list(param_geometry.names),
                             include_amp=include_amp)
  return chi2fn
