"""Sparse-Legendre PCE machinery and the PCEEmulator (the NPCE base).

This subpackage holds the polynomial-chaos side of the NPCE (Neural
PCE) experiment: PCEEmulator, a closed-form sparse-Legendre expansion
mapping the cosmological parameters to the whitened data vector with
no network, plus its three fit helpers -- pce_multi_index (the sparse
candidate basis), pce_design (the normalized-Legendre design matrix),
and select_lars_loo (greedy term selection with a leave-one-out
stop). The companion loss_functions.py wraps a frozen PCEEmulator as
the base under a neural refiner.

Verdict for cosmic-shear xi (2026-06-26, recorded in
notes/npce-and-ia-template-factoring.md): a PCE base only adds
capacity, it cannot lower a data-coverage floor, so the NPCE was
deprioritized. The machinery is kept for reuse and for the build
lessons baked into the docstrings below: keep the degree low (a
high-degree Legendre fit Runge-oscillates), gate each mode on its
leave-one-out error (a wiggly base poisons the refiner), and let the
refiner backstop everything the gate drops.

PS: PCE = polynomial chaos expansion, a sum of orthogonal
polynomials of the inputs; LOO = leave-one-out error, the fit's
generalization error estimated without refitting (the PRESS shortcut
in select_lars_loo); whitened = transformed to unit-variance
components (the chi2 metric basis); mode = one SVD direction of the
centered training targets.
"""

import itertools
import numpy as np
import torch
import torch.nn as nn


def _pce_deg_tuples(m, pq, q):
  """Degree tuples (len m, each >=1) with sum a_i^q <= pq.

  Recursive enumeration with q-norm pruning: extend the tuple one
  dim at a time, abandoning a branch once its running sum of a_i^q
  exceeds the budget pq.
  """
  def rec(prefix, used):
    if len(prefix) == m:
      yield tuple(prefix)
      return
    d = 1
    while used + d ** q <= pq:
      yield from rec(prefix=prefix + [d], used=used + d ** q)
      d += 1
  yield from rec(prefix=[], used=0.0)


def pce_multi_index(n_dim, p_max=12, r_max=3, q=0.6):
  """
  Sparse candidate multi-index set A_cand (eq 11).

  Every multi-index alpha = (a_1, ..., a_n_dim) of per-variable
  degrees a_i >= 0 passes both rules of the hybrid truncation:
    - hyperbolic q-norm:  (sum_i a_i^q)^(1/q) <= p_max
    - max interaction:    #(a_i != 0)          <= r_max
  The all-zero index (the constant) is row 0.

  The q-norm scores a degree spread over many variables against one
  concentrated in a single variable. Worked example with p_max = 4,
  three terms of total degree 4:
    term          q = 1        q = 0.5
    x1^4          4  -> keep    4  -> keep
    x1^2 x2^2     4  -> keep    8  -> drop
    x1 x2 x3 x4   4  -> keep    16 -> drop
  At q = 1 the norm is the plain total degree, so a 4-way
  interaction counts as one degree-4 variable. At q < 1 spreading a
  degree over k variables costs k^(1/q) instead of k (the two- and
  four-variable terms score 8 and 16, past p_max = 4), dropping
  high-interaction cross-terms first -- the sparsity-of-effects
  prior. Smaller q = sparser basis.

  Arguments:
    n_dim = number of input parameters (here 12).
    p_max = maximum total degree (the q-norm bound); the smoothness
            knob (low = smooth, high = Runge risk).
    r_max = maximum interaction order = most variables allowed
            together in one term (#(a_i != 0) <= r_max).
    q     = hyperbolic-norm exponent in (0, 1]; the sparsity knob
            (q=1 = plain total degree; smaller drops
            high-interaction terms -> sparser; see above).
  Returns:
    multi_index = (n_terms, n_dim) int array, row 0 constant.
  """
  pq   = p_max ** q
  rows = [np.zeros(n_dim, dtype=int)]      # constant term
  # every active subset of size 1..r_max, with all degree tuples
  # (>=1 on those dims) under the q-norm budget.
  for m in range(1, r_max + 1):
    for dims in itertools.combinations(range(n_dim), m):
      for degs in _pce_deg_tuples(m=m, pq=pq, q=q):
        a = np.zeros(n_dim, dtype=int)
        for d, deg in zip(dims, degs):
          a[d] = deg
        rows.append(a)
  return np.array(rows, dtype=int)


def pce_design(Xm, multi_index):
  """
  Normalized-Legendre PCE design matrix Psi (eq 10).

    Psi[n,t] = prod_l sqrt(2 a_{t,l}+1) * P_{a_{t,l}}(Xm[n,l])

  P = Legendre polynomial (orthogonal on [-1,1]); the sqrt(2a+1)
  factor makes each 1-D factor orthonormal. Legendre values from the
  three-term recurrence
    (n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}, P_0=1, P_1=x.
  Pure torch, so one implementation runs on CPU (fit) and GPU
  (predict).

  Arguments:
    Xm          = (N, n_dim) inputs mapped to [-1, 1].
    multi_index = (n_terms, n_dim) long tensor of degrees.
  Returns:
    Psi = (N, n_terms) on Xm's device/dtype.
  """
  N, d = Xm.shape
  T    = multi_index.shape[0]
  maxd = int(multi_index.max())
  Psi  = torch.ones(N, T, dtype=Xm.dtype, device=Xm.device)
  for l in range(d):
    x = Xm[:, l]
    # Legendre table P_0..P_maxd for this dim: (N, maxd+1).
    cols = [torch.ones_like(x)]
    if maxd >= 1:
      cols.append(x)
    for n in range(1, maxd):
      cols.append(((2 * n + 1) * x * cols[n]
                   - n * cols[n - 1]) / (n + 1))
    tab  = torch.stack(cols, dim=-1)          # (N, maxd+1)
    a    = multi_index[:, l]                  # (T,)
    norm = torch.sqrt(2.0 * a.to(Xm.dtype) + 1.0)
    Psi  = Psi * (norm * tab[:, a])           # gather + scale
  return Psi


def select_lars_loo(Psi, y, max_terms=150, patience=10):
  """
  Greedy least-angle / OMP selection with a leave-one-out (LOO)
  stop -- the self-contained stand-in for UQLab's LARS+LOO sparse-
  PCE selection.

  Returns:
    support = int array of selected column indices.
    coef    = OLS coefficients aligned with `support`.
    loo     = relative leave-one-out MSE at the chosen model
              = mean((y - y_pred)^2 leave-one-out) / var(y)
              = 1 - R^2_LOO. 0 = perfect, 1 = no better than the
              mean; sqrt(loo) = typical error as a fraction of y's
              spread.
  """
  cn = np.sqrt((Psi ** 2).sum(0)) + 1e-30    # column norms
  vy = np.var(y) + 1e-30                      # target variance
  active    = [0]                             # constant term
  best_loo  = np.inf
  best_supp = [0]
  best_beta = None
  since     = 0

  for _ in range(max_terms):
    A    = Psi[:, active]
    G    = A.T @ A + 1e-10 * np.eye(len(active))
    Ginv = np.linalg.inv(G)
    beta = Ginv @ (A.T @ y)
    resid = y - A @ beta                      # in-sample resid

    # hat = leverage h_nn = diag of A (A^T A)^-1 A^T: how much
    # point n pulls its own fit; near 1 = high influence.
    hat = np.einsum("ni,ij,nj->n", A, Ginv, A)
    hat = np.minimum(hat, 1.0 - 1e-6)
    # LOO (generalization) error without refitting:
    #   resid / (1 - hat) = point n's residual when n is dropped
    #     from the fit (the PRESS shortcut; 1/(1 - h_nn) inflates the
    #     in-sample residual to its leave-one-out value).
    #   / vy normalizes by the target variance, so loo is
    #     dimensionless and comparable across modes:
    #       loo = mean(LOO residual^2) / var(y) = 1 - R^2_LOO.
    #   The absolute chi2 a mode adds is loo * var(mode), so a
    #   high-variance mode must reach a very small loo to help.
    loo = np.mean((resid / (1.0 - hat)) ** 2) / vy
    if loo < best_loo - 1e-6:
      best_loo  = loo
      best_supp = list(active)
      best_beta = beta.copy()
      since = 0
    else:
      since += 1
    if since >= patience or len(active) >= max_terms:
      break

    # next term: candidate column most correlated with the residual
    # (scaled by its norm); never re-pick an active one.
    score = np.abs(Psi.T @ resid) / cn
    score[active] = -1.0
    active.append(int(np.argmax(score)))

  return np.array(best_supp, dtype=int), best_beta, best_loo


class PCEEmulator(nn.Module):
  """
  Sparse-Legendre Polynomial Chaos Expansion (PCE) emulator -- the
  analytic "base" of the NPCE (Neural PCE). It maps the cosmological
  parameters to the whitened data vector with no network: every
  coefficient is a closed-form least-squares fit, built in one pass
  over the training set.

  A polynomial chaos expansion writes a quantity as a sum of
  orthogonal polynomials of the inputs. Here (eqs 9-11) each
  compressed dv coefficient lambda_i is expanded in normalized
  Legendre polynomials of the 12 parameters mapped to [-1, 1]:
    lambda_i(theta) ~ sum_alpha eta_{i,alpha} Psi_alpha(x)
  Psi_alpha is a product of 1-D Legendre polynomials (eq 10) and
  alpha runs over a sparse multi-index set from a hyperbolic q-norm
  + max-interaction truncation, pruned by LARS with a leave-one-out
  criterion (eq 11). Sparse = only a handful of terms survive
  (sparsity-of-effects), making the fit data-efficient and
  overfit-resistant.

  What the lambda_i are: the dv targets are covariance-whitened
  (geom.encode), ensemble-centered, and SVD-compressed to K leading
  modes; those K amplitudes are the lambda_i (eq-9
  principal-component amplitudes). Compressing in the whitened basis
  is deliberate -- that basis is the chi2 metric (chi2 == ||.||^2)
  -- so (a) the least-squares PCE on each mode directly minimizes
  the expected chi2, and (b) dropping a mode costs its
  singular-value^2 / N in mean chi2, bounding the truncation error
  in the reported metric.

  Two design rules, learned the hard way (see the NPCE notes):
    - Keep only well-predicted modes. Mode 0 (the overall amplitude,
      ~A_s/S_8 scaling) is smooth and cleanly polynomial-
      predictable; the higher "shape" modes often are not. A mode
      kept with a poor fit injects more error than it removes, so
      only modes with relative LOO < loo_max enter the base; the
      rest go to the NPCE refiner, which corrects the full dv and
      backstops everything dropped.
    - Keep the degree low. A high-degree Legendre fit Runge-
      oscillates, and subtracting a wiggly base makes the refiner's
      residual harder. Degree (p_max) is the smoothness knob; term
      count (max_terms) only adds richness within that degree -- so
      max_terms is generous and the LOO, not the cap, decides each
      mode's term count.

  Drop-in model(X) -> whitened dv: X is the pgeom-whitened parameter
  batch (the input the SGD models see). Build with the from_training
  classmethod and wrap as the base of an NPCE loss (PCEResidualChi2 =
  additive, PCERatioChi2 = multiplicative).

  Forward shape flow:

      X  (B, n_dim)              pgeom-whitened parameters
         │  box map + clamp      2 (X - lo) / (hi - lo) - 1
         ▼
      Xm (B, n_dim)              Legendre domain [-1, 1]
         │  pce_design           products of 1-D Legendre factors
         ▼
      Psi (B, n_terms)           the sparse polynomial basis
         │  @ C                  fitted coefficients, mode by mode
         ▼
      Z  (B, K)                  the K mode amplitudes (lambda_i)
         │  @ Vk^T  + Ybar       SVD reconstruction + ensemble mean
         ▼
      out (B, n_keep)            whitened data vector

      (legend: B = batch rows; n_dim = number of cosmological
       parameters (LSST-Y1 example: 12); n_terms = multi-indices in
       the sparse candidate basis; K = SVD modes kept by the LOO
       gate; n_keep = kept data-vector length the network emulates;
       lo / hi / C / Vk / Ybar = the frozen buffers listed below.)

  Buffers (frozen; move with .to(device), never trained):
    lo, hi      = per-parameter [-1, 1] box-map bounds.
    multi_index = (n_terms, n_dim) Legendre degree exponents.
    C           = (n_terms, K) sparse coefficient matrix (zero off
                  each mode's support).
    Vk          = (n_keep, K) leading SVD modes the amplitudes
                  reconstruct against.
    Ybar        = (n_keep,) training-ensemble mean of the whitened dv.
  """

  def __init__(self, lo, hi, multi_index, C, Vk, Ybar):
    super().__init__()
    self.register_buffer("lo", lo)
    self.register_buffer("hi", hi)
    self.register_buffer("multi_index", multi_index)
    self.register_buffer("C", C)
    self.register_buffer("Vk", Vk)
    self.register_buffer("Ybar", Ybar)

  @classmethod
  def from_training(cls, device, X_white, Y_white,
                    p_max=4, r_max=2, q=0.5,
                    k_max=40, loo_max=0.05,
                    max_terms=30, max_fail=4, silent=False):
    """
    Fit from whitened training inputs/targets.

    Fit pipeline (one pass, no gradient descent):

        X_white (N, n_dim)      Y_white (N, n_keep)
           │                       │  center on the mean Ybar
           │  box map to [-1,1]    ▼
           ▼                    Yc (N, n_keep)
        Psi (N, n_terms)           │  SVD -> modes Vt, variances S^2
           │                       ▼
           │                    z_k = Yc @ Vt[k]  mode amplitudes
           │                       │
           └───────────┬───────────┘
                       │  per mode k: select_lars_loo(Psi, z_k)
                       ▼
        keep mode k iff loo < loo_max
        (stop after max_fail consecutive misses)
                       │
                       ▼
        frozen buffers: lo, hi, multi_index, C, Vk, Ybar

        (legend: N = training rows; n_dim / n_terms / n_keep as in
         the class docstring; z_k = the k-th SVD amplitude over the
         training set; loo = relative leave-one-out MSE from
         select_lars_loo; Vt = the SVD's right singular vectors,
         row per mode.)

    Arguments:
      device   = device the buffers live on.
      X_white  = (N, n_dim) pgeom-whitened training params.
      Y_white  = (N, n_keep) covariance-whitened targets.
      p_max    = max total degree (smoothness knob); low (3-6).
      r_max    = max interaction order (vars per term).
      q        = sparsity exponent in (0,1] (q=1 = total degree;
                 smaller = sparser; example in pce_multi_index).
      k_max    = max leading SVD modes to try.
      loo_max  = keep a mode only if its relative LOO < this.
      max_terms = per-mode active-set cap.
      max_fail = stop after this many consecutive gate failures
                 (leading modes are the predictable ones, so a run
                 of misses means the rest miss too -- avoids fitting
                 modes that will only be dropped).
      silent   = suppress the fit report.
    Returns:
      a fitted PCEEmulator on `device`.
    """
    Xn = np.asarray(X_white.detach().cpu(), dtype="float64")
    Yn = np.asarray(Y_white.detach().cpu(), dtype="float64")
    N, n_dim = Xn.shape

    mid  = 0.5 * (Xn.min(0) + Xn.max(0))
    half = 0.5 * (Xn.max(0) - Xn.min(0)) * 1.05 + 1e-12
    lo, hi = mid - half, mid + half
    Xm = 2.0 * (Xn - lo) / (hi - lo) - 1.0

    mi   = pce_multi_index(n_dim=n_dim, p_max=p_max, r_max=r_max, q=q)
    mi_t = torch.as_tensor(mi, dtype=torch.long)
    Psi  = pce_design(
      Xm=torch.as_tensor(Xm, dtype=torch.float64),
      multi_index=mi_t).numpy()

    Ybar = Yn.mean(0)
    Yc   = Yn - Ybar
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    var = S ** 2

    # fit leading modes; keep the well-predicted ones (loo <
    # loo_max). Stop after max_fail consecutive misses.
    kfit = min(k_max, len(S))
    cols, kept, loos, all_loo = [], [], [], []
    fails = 0
    for k in range(kfit):
      zk = Yc @ Vt[k]
      supp, beta, loo = select_lars_loo(Psi=Psi, y=zk, max_terms=max_terms)
      all_loo.append(loo)
      if loo < loo_max:
        col = np.zeros(Psi.shape[1])
        col[supp] = beta
        cols.append(col)
        kept.append(k)
        loos.append(loo)
        fails = 0
      else:
        fails += 1
        if fails >= max_fail:
          break
    if not cols:                       # always keep mode 0
      zk = Yc @ Vt[0]
      supp, beta, loo = select_lars_loo(Psi=Psi, y=zk, max_terms=max_terms)
      col = np.zeros(Psi.shape[1])
      col[supp] = beta
      cols, kept, loos = [col], [0], [loo]

    C  = np.stack(cols, axis=1)
    Vk = Vt[kept].T
    K  = len(kept)
    drop_chi2 = float((var.sum() - var[kept].sum()) / N)

    if not silent:
      act   = (C != 0).sum(0)
      tried_vals = []
      for l in all_loo[:8]:
        tried_vals.append(f"{l:.2e}")
      tried = ", ".join(tried_vals)
      print(f"PCE fit: N {N}  n_dim {n_dim}  "
            f"candidates {Psi.shape[1]}  fit {len(all_loo)}")
      print(f"  KEPT {K} (loo<{loo_max})  "
            f"mean dropped chi2 {drop_chi2:.4f}")
      print(f"  active/mode: median {int(np.median(act))}"
            f"  max {int(act.max())}")
      print(f"  tried-mode LOO[:8]: [{tried}]")

    f = torch.float32
    return cls(lo=torch.tensor(lo, dtype=f, device=device),
               hi=torch.tensor(hi, dtype=f, device=device),
               multi_index=mi_t.to(device),
               C=torch.tensor(C, dtype=f, device=device),
               Vk=torch.tensor(Vk, dtype=f, device=device),
               Ybar=torch.tensor(Ybar, dtype=f, device=device))

  def forward(self, X):
    Xm  = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
    Xm  = Xm.clamp(-1.0, 1.0)
    Psi = pce_design(Xm=Xm, multi_index=self.multi_index)
    Z   = Psi @ self.C
    return self.Ybar + Z @ self.Vk.t()
