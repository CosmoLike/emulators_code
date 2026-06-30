"""Model diagnostics over a validation set.

Three post-training analyses that say why the metric sits where it does
(each returns a dict the plotting reads). coverage_diagnostic asks whether
the failing val points sit in sparse regions of the training set (a
kNN-distance vs delta-chi2 correlation, i.e. data coverage).
local_linear_floor compares the model to a local-linear interpolation of
the training targets (the data-only floor; plain chi2 only).
hard_direction_regression fits log10 delta-chi2 against the (log)
parameters to find which combination predicts the per-point hardness.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from .training import eval_source_chi2


def coverage_diagnostic(model, param_geometry, chi2fn, train_set,
                        val_set, device, k_nn=8, bs=256):
  """
  Do the failing val points sit in sparse regions of training?

  For each validation cosmology, measure its mean distance to the
  k nearest training cosmologies in whitened (param_geometry)
  parameter space -- Euclidean distance there weights each direction
  by its prior spread, so no single wide param dominates -- and
  relate that local sparsity to the per-point delta-chi2. A positive
  rank correlation (sparser neighbourhoods at the failures) means
  the floor is data coverage, not the model. Model-agnostic: any
  trained model works.

  Arguments:
    model          = the trained network (eval_source_chi2 sets
                     eval mode).
    param_geometry = ParamGeometry; .encode whitens the raw params
                     to the decorrelated, unit-variance metric the
                     kNN distance uses.
    chi2fn         = the loss/geometry wrapper (plain or rescaled);
                     scores the val dchi2.
    train_set      = training source dict; its used rows are the
                     interpolation anchors (the training cloud).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    k_nn           = neighbours averaged for the local-density
                     estimate (default 8).
    bs             = forward batch size for the dchi2 scoring.

  Returns:
    a dict with:
      knn_dist  = (Nval,) mean distance to the k nearest train pts.
      dchi2     = (Nval,) per-val delta-chi2 (same row order).
      k_nn      = the k used (for axis labels downstream).
      spearman  = rank correlation of knn_dist with log10 dchi2.
      median_good / median_bad = median knn_dist of the
                  dchi2<=0.2 / dchi2>0.2 populations.
      frac_dense / frac_sparse = frac>0.2 in the densest / sparsest
                  knn_dist decile.
      coverage_limited = bool verdict (failures in sparse regions:
                  median_bad > median_good, spearman > 0.1).
  """
  # per-val delta-chi2 from the model (sorted-idx order).
  _, dchi2 = eval_source_chi2(model=model,
                              param_geometry=param_geometry,
                              chi2fn=chi2fn,
                              source=val_set,
                              device=device,
                              bs=bs)

  # whitened params: the training cloud (anchors) and the val
  # points. encode decorrelates + unit-scales, so Euclidean
  # distance weights every direction by its prior spread.
  tr_rows = np.sort(np.unique(train_set["idx"]))
  va_rows = np.sort(val_set["idx"])
  with torch.no_grad():
    Xtr = param_geometry.encode(torch.from_numpy(
      np.asarray(train_set["C"][tr_rows], dtype="float64")
    ).float().to(device)).cpu().numpy()
    Xva = param_geometry.encode(torch.from_numpy(
      np.asarray(val_set["C"][va_rows], dtype="float64")
    ).float().to(device)).cpu().numpy()

  # mean distance from each val point to its k nearest training
  # points -- a local sparsity measure (large = under-covered).
  # cKDTree.query returns (distances, indices); keep distances.
  tree = cKDTree(Xtr)
  dists, _ = tree.query(Xva, k=k_nn)     # (Nval, k_nn)
  knn_dist = dists.mean(1)               # (Nval,)

  # quantify. log10 dchi2 (floored so a near-zero stays finite)
  # tames the heavy tail; spearman is the rank correlation.
  y = np.log10(np.maximum(dchi2, 1e-4))
  rho, _ = spearmanr(knn_dist, y)
  bad = dchi2 > 0.2
  q10, q90 = np.quantile(knn_dist, [0.1, 0.9])
  median_good = float(np.median(knn_dist[~bad]))
  median_bad  = float(np.median(knn_dist[bad]))
  frac_dense  = float(np.mean(dchi2[knn_dist <= q10] > 0.2))
  frac_sparse = float(np.mean(dchi2[knn_dist >= q90] > 0.2))
  cov = (median_bad > median_good) and (rho > 0.1)

  return {"knn_dist": knn_dist, "dchi2": dchi2, "k_nn": k_nn,
          "spearman": float(rho),
          "median_good": median_good, "median_bad": median_bad,
          "frac_dense": frac_dense, "frac_sparse": frac_sparse,
          "coverage_limited": bool(cov)}


def local_linear_floor(model, param_geometry, chi2fn, train_set,
                       val_set, device, k_nn=40, bs=256):
  """
  The data-only floor: a local linear map vs the trained model.

  For each val point, fit a local linear map params -> whitened
  target over its k nearest training points and predict the val
  target; that prediction's chi2 is the best a smooth local method
  extracts from the data. A linear fit is exact for a locally-linear
  map, so its error is the local nonlinearity (hardness) plus
  residual coverage. Comparing the fractions:
    f_model ~ f_floor  -> data / representation-limited (the net is
                          at what the data supports; lever = prior /
                          features / more N).
    f_model >> f_floor -> the net has headroom (arch / training).
  f_floor in the best-covered (densest) decile = pure hardness.

  Valid only for a plain CosmolikeChi2 (needs_params == False): the
  fit lives in the whitened target space chi2fn.encode(dv) builds,
  and a rescaled encode/chi2 would need each point's own R.

  Arguments:
    model          = the trained network (for the model dchi2).
    param_geometry = ParamGeometry; .encode whitens the params
                     (the kNN space) and chi2fn.encode the targets.
    chi2fn         = a plain CosmolikeChi2 (raises otherwise).
    train_set      = training source dict (the fit anchors).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    k_nn           = neighbours for the local linear fit (default
                     40; must exceed n_param + 1).
    bs             = forward batch size for the model dchi2.

  Returns:
    a dict with dchi2_floor, dchi2_model (both (Nval,)) and the
    scalars f_floor, f_model, f_hard, median_floor, median_model.
  """
  if getattr(chi2fn, "needs_params", False):
    raise ValueError(
      "local_linear_floor needs a plain CosmolikeChi2 "
      "(this chi2fn has needs_params == True)")

  tr_rows = np.sort(np.unique(train_set["idx"]))
  va_rows = np.sort(val_set["idx"])
  with torch.no_grad():
    Xtr = param_geometry.encode(torch.from_numpy(np.asarray(
      train_set["C"][tr_rows], "float64")).float().to(device))
    Xva = param_geometry.encode(torch.from_numpy(np.asarray(
      val_set["C"][va_rows], "float64")).float().to(device))
    Ttr = chi2fn.encode(torch.from_numpy(
      np.asarray(train_set["dv"][tr_rows])).float().to(device))
    Tva = chi2fn.encode(torch.from_numpy(
      np.asarray(val_set["dv"][va_rows])).float().to(device))

  # k nearest training neighbours of each val point (param space).
  tree = cKDTree(Xtr.cpu().numpy())
  knn_d, nbr = tree.query(Xva.cpu().numpy(), k=k_nn)
  knn_dist = knn_d.mean(1)                        # coverage scalar
  nbr = torch.from_numpy(nbr).to(device)

  # local linear fit: target ~ b + A (x - x_val) over the
  # neighbours, with intercept b = the prediction at x_val.
  # Solve on CPU (batched lstsq is not on MPS) -- one-time.
  Xn = Xtr[nbr]                                   # (Nval, k, n_param)
  Yn = Ttr[nbr]                                   # (Nval, k, out_dim)
  dX = (Xn - Xva[:, None, :]).cpu()
  ones = torch.ones(dX.shape[0], dX.shape[1], 1)
  # design = [1, (x - x_val)], so column 0's coefficient is b.
  design = torch.cat([ones, dX], dim=-1)          # (Nval, k, n_p+1)
  coef = torch.linalg.lstsq(design, Yn.cpu()).solution
  Tlin = coef[:, 0, :].to(device)                 # intercept = pred

  dchi2_floor = chi2fn.chi2(pred=Tlin,
                            target=Tva).double().cpu().numpy()
  _, dchi2_model = eval_source_chi2(model=model,
                                    param_geometry=param_geometry,
                                    chi2fn=chi2fn, source=val_set,
                                    device=device, bs=bs)
  # pure hardness: the floor in the densest (best-covered) decile.
  dense = knn_dist <= np.quantile(knn_dist, 0.1)
  return {"dchi2_floor": dchi2_floor, "dchi2_model": dchi2_model,
          "f_floor": float(np.mean(dchi2_floor > 0.2)),
          "f_model": float(np.mean(dchi2_model > 0.2)),
          "f_hard": float(np.mean(dchi2_floor[dense] > 0.2)),
          "median_floor": float(np.median(dchi2_floor)),
          "median_model": float(np.median(dchi2_model))}


def hard_direction_regression(model, param_geometry, chi2fn,
                              val_set, device, bs=256, log_set=None):
  """
  Which log-param combination predicts the per-point hardness?

  Fits log10 dchi2 ~ c0 + sum_i c_i z_i, with z_i = standardized
  ln(param / median) for the positive multiplicative cosmological
  params and standardized centered-linear for the additive
  nuisances (photo-z DZ, IA A1 -- which can be <= 0). Reports each
  feature's univariate correlation (a collinearity-robust ranking),
  the joint OLS coefficients (the alpha, beta, ... combination) and
  joint R^2 (how much of the difficulty is a clean log-linear
  direction), and the ln(omega_b h^2)-alone R^2 (does it collapse
  to that single physical-baryon direction?). Works for any chi2fn:
  the dchi2 comes from eval_source_chi2's param-aware path.

  Arguments:
    model          = the trained network.
    param_geometry = ParamGeometry; .names gives the column order.
    chi2fn         = the loss/geometry wrapper (plain or rescaled).
    val_set        = validation source dict (the points scored).
    device         = device the model is on.
    bs             = forward batch size for the dchi2 scoring.
    log_set        = parameter names ln-transformed before
                     standardizing (default the positive
                     cosmological params As_1e9 / ns / H0 / omegam
                     / omegab).

  Returns:
    a dict with labels (per feature), univariate (the per-feature
    correlations), joint_coef (the joint coefficients, no
    intercept), r2 (joint), and r2_omega (ln(omega_b h^2) alone).
  """
  if log_set is None:
    log_set = {"As_1e9", "ns", "H0", "omegam", "omegab"}
  params, dchi2 = eval_source_chi2(model=model,
                                   param_geometry=param_geometry,
                                   chi2fn=chi2fn, source=val_set,
                                   device=device, bs=bs)
  names = list(param_geometry.names)
  y = np.log10(np.maximum(dchi2, 1e-4))

  # ln(param/median) for the positive multiplicative params;
  # centered-linear for the additive nuisances. Standardize so the
  # coefficients are comparable.
  feat, lab = [], []
  for j, nm in enumerate(names):
    x = params[:, j].astype("float64")
    f = np.log(x / np.median(x)) if nm in log_set else x - np.mean(x)
    feat.append((f - f.mean()) / (f.std() + 1e-30))
    lab.append(("ln " if nm in log_set else "") + nm)
  feat = np.column_stack(feat)

  # univariate (collinearity-robust): each feature's own
  # correlation with log10 dchi2.
  uni_vals = []
  for j in range(feat.shape[1]):
    uni_vals.append(np.corrcoef(feat[:, j], y)[0, 1])
  uni = np.array(uni_vals)
  # joint OLS (a column of 1s is the intercept) and its R^2.
  Z = np.column_stack([np.ones_like(y), feat])
  coef, *_ = np.linalg.lstsq(Z, y, rcond=None)
  r2 = 1.0 - np.var(y - Z @ coef) / np.var(y)

  # does it collapse to a single direction, ln(omega_b h^2)?
  ob = params[:, names.index("omegab")].astype("float64")
  h  = params[:, names.index("H0")].astype("float64") / 100.0
  g  = np.log(ob * h ** 2 / np.median(ob * h ** 2))
  g  = (g - g.mean()) / g.std()
  Zo = np.column_stack([np.ones_like(y), g])
  co, *_ = np.linalg.lstsq(Zo, y, rcond=None)
  r2o = 1.0 - np.var(y - Zo @ co) / np.var(y)

  return {"labels": lab, "univariate": uni, "joint_coef": coef[1:],
          "r2": float(r2), "r2_omega": float(r2o)}
