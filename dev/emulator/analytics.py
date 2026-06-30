"""Analytic cosmic-shear rescaling R (the As / shape preprocessor).

Computes a fast, closed-form reference xi (Eisenstein-Hu zero-baryon
transfer, linear, single-plane Limber) to divide out the broadband
cosmology dependence so the network emulates a flatter target. _analytic_R
holds the formula (numpy or torch, picked by input type).
analytic_shape_ratio wraps it over the masked data vector (the emulator
path); rescale_xi wraps it over the (theta, xip, xim) matrix layout (for
plotting and visual checks). The RescaledChi2 and ResidualBaseChi2 losses
call _analytic_R on-device.
"""

import numpy as np
import torch


def _analytic_R(theta_arcmin, z_eff, cosmo, cosmo_mid,
                names, u_star=0.5, include_amp=False):
  """
  Core analytic cosmic-shear rescaling R -- the one place the
  formula lives.

    R = (As_mid/As) * q_mid^ns_mid T(q_mid)^2 / (q^ns T(q)^2),
    q = K / (theta_rad * z_eff * Omega_m h),
    K = 100 Theta^2 / (c[km/s] * u*),
  with T the Eisenstein-Hu zero-baryon transfer function. Each
  cosmology uses its own ns.

  One formula, two array libraries: numpy (the analysis /
  plotting wrappers) or torch (the training loop, on-device from
  the resident params). The body is shared arithmetic; only log
  and the coercion differ. The numpy path is bit-identical to
  the original.

  theta_arcmin and z_eff broadcast to the element shape S (both
  (n_keep,) for the masked dv; (ntheta,1,1) and (1,nt,nt) for
  the full xi matrix).

  Arguments:
    theta_arcmin = per-element angular scale(s) [arcmin].
    z_eff        = per-element effective source redshift(s), e.g.
                   min(z_i, z_j) for a tomographic pair.
    cosmo        = (N, n_param) rows to rescale, numpy array or
                   torch tensor; a tensor's dtype/device sets the
                   output's.
    cosmo_mid    = (n_param,) reference ("mid") cosmology; R=1
                   for a row equal to it.
    names        = parameter column names (pgeom.names order);
                   locate As_1e9 / ns / H0 / omegam.
    u_star       = lens position (kernel peak) in the theta -> q
                   map; ~0.5.
    include_amp  = if True, also multiply R by the surviving
                   geometric-amplitude factor N ~ (Omega_m
                   h^2)^ns / h (a second amplitude direction
                   beyond A_s). The standard run sets it True;
                   off by default.

  Returns:
    R = (N, *S), same library/dtype as cosmo.
  """
  # pick the array library once: torch when cosmo is a tensor
  # (on-device path), numpy otherwise. log is the only math call
  # that differs; coerce casts the geometry arrays into cosmo's
  # library/dtype/device to broadcast.
  is_torch = torch.is_tensor(cosmo)
  if is_torch:
    log    = torch.log
    coerce = lambda a: torch.as_tensor(
      a, dtype=cosmo.dtype, device=cosmo.device)
    # a lone 1D row -> (1, n_param): the tensor np.atleast_2d, so
    # the [:, col] indexing below works.
    if cosmo.ndim == 1:
      cosmo = cosmo[None, :]
  else:
    log    = np.log
    coerce = lambda a: np.asarray(a, dtype="float64")
    cosmo  = np.atleast_2d(
      np.asarray(cosmo, dtype="float64"))
  mid = coerce(cosmo_mid)

  iA = names.index("As_1e9")
  iN = names.index("ns")
  iH = names.index("H0")
  iO = names.index("omegam")
  As   = cosmo[:, iA]
  ns   = cosmo[:, iN]
  Gam  = cosmo[:, iO] * (cosmo[:, iH] / 100.0)
  As_m, ns_m = mid[iA], mid[iN]
  Gam_m = mid[iO] * (mid[iH] / 100.0)

  Theta2 = (2.725 / 2.7) ** 2
  C_KMS  = 2.99792458e5
  K      = 100.0 * Theta2 / (C_KMS * u_star)
  th_rad = coerce(theta_arcmin) * (np.pi / (180.0 * 60.0))
  base   = K / (th_rad * coerce(z_eff))    # element shape S

  # flatten the elements to one axis so cosmo (N) broadcasts
  # cleanly, then restore the element shape at the end. S is a
  # plain tuple so (N,) + S works for both libraries.
  S     = tuple(base.shape)
  flat  = base.reshape(-1)                   # (n_elem,)
  # flat[None, :] -> (1, n_elem); Gam[:, None] -> (N, 1). Dividing
  # broadcasts to the full (N, n_elem) grid -- every cosmology's
  # Gamma against every element's base wavenumber. ([None, :] and
  # [:, None] are the numpy/torch spelling of unsqueeze.)
  q     = flat[None, :] / Gam[:, None]       # (N, n_elem)
  q_mid = flat[None, :] / Gam_m              # (1, n_elem)

  def T(qq):
    # Eisenstein-Hu zero-baryon transfer function (1998).
    L = log(2.0 * np.e + 1.8 * qq)
    C = 14.2 + 731.0 / (1.0 + 62.5 * qq)
    return L / (L + C * qq * qq)

  shape     = q ** ns[:, None] * T(q) ** 2
  shape_mid = q_mid ** ns_m * T(q_mid) ** 2
  R = (As_m / As)[:, None] * shape_mid / shape  # (N, n_elem)

  if include_amp:
    # surviving geometric amplitude N ~ (Om h^2)^ns / h: the z_s
    # and theta parts cancelled in the ratio, leaving a
    # per-cosmology scalar (second amplitude direction).
    wm   = cosmo[:, iO] * (cosmo[:, iH] / 100.0) ** 2
    wm_m = mid[iO] * (mid[iH] / 100.0) ** 2
    h    = cosmo[:, iH] / 100.0
    h_m  = mid[iH] / 100.0
    amp_mid = wm_m ** ns_m / h_m
    amp     = wm ** ns / h
    R = R * (amp_mid / amp)[:, None]

  return R.reshape((cosmo.shape[0],) + S)        # (N, *S)


def analytic_shape_ratio(
  cosmo, cosmo_mid, names, theta_kept, zsrc_i, zsrc_j,
  u_star=0.5, include_amp=False):
  """
  Cosmic-shear rescaling R over the masked (kept) data vector,
  for the emulator pipeline: column k aligns with kept element
  dest_idx[k]. Multiply the squeezed dv by R to preprocess;
  divide it back out before the chi2. Calls _analytic_R with the
  kept-element geometry.

  Arguments:
    cosmo       = (N, n_param) rows to rescale.
    cosmo_mid   = (n_param,) reference cosmology.
    names       = parameter names (pgeom.names).
    theta_kept  = (n_keep,) angular scale per kept element
                  [arcmin] (geom.theta_kept).
    zsrc_i      = (n_keep,) first source redshift per kept pair
                  (geom.zsrc_i).
    zsrc_j      = (n_keep,) second source redshift (geom.zsrc_j);
                  pair = min(zsrc_i, zsrc_j).
    u_star      = kernel peak ~0.5.
    include_amp = forwarded to _analytic_R (the (Omega_m
                  h^2)^ns / h amplitude factor).
  Returns:
    R = (N, n_keep) float64.
  """
  z_eff = np.minimum(zsrc_i, zsrc_j)
  return _analytic_R(theta_arcmin=theta_kept,
                     z_eff=z_eff,
                     cosmo=cosmo,
                     cosmo_mid=cosmo_mid,
                     names=names,
                     u_star=u_star,
                     include_amp=include_amp)


def rescale_xi(
  xi, cosmo, cosmo_mid, names, z_src,
  u_star=0.5, include_amp=True):
  """
  Rescale a list of xi curves by R, in the (theta, xip, xim)
  matrix layout, for plotting/visual checks. Calls _analytic_R
  with the full-block matrix geometry (all tomographic pairs and
  theta, unmasked). R > 0, so xi- keeps its sign; R = 1 for a
  curve equal to cosmo_mid. z_eff = min(z_i, z_j) for the cross
  pairs.

  Arguments:
    xi          = list of (theta, xip, xim); xip/xim are
                  (ntheta, ntomo, ntomo), theta [arcmin].
    cosmo       = (len(xi), n_param) params, one row per curve.
    cosmo_mid   = (n_param,) reference cosmology.
    names       = parameter names (pgeom.names).
    z_src       = (ntomo,) source-bin peak redshifts (geom.z_src).
    u_star      = kernel peak ~0.5.
    include_amp = forwarded to _analytic_R (the (Omega_m
                  h^2)^ns / h amplitude factor).
  Returns:
    a new list of (theta, xip*R, xim*R).
  """
  z     = np.asarray(z_src)
  z_eff = np.minimum(z[:, None], z[None, :])     # (nt, nt)
  theta = np.asarray(xi[0][0])                   # (ntheta,)
  # R[k] for curve k: (ntheta, ntomo, ntomo).
  R = _analytic_R(theta_arcmin=theta[:, None, None],
                  z_eff=z_eff[None, :, :],
                  cosmo=cosmo,
                  cosmo_mid=cosmo_mid,
                  names=names,
                  u_star=u_star,
                  include_amp=include_amp)
  out = []
  for k, (th, xip, xim) in enumerate(xi):
    out.append((th, xip * R[k], xim * R[k]))
  return out
