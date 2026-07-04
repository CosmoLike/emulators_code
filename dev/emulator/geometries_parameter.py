"""Input (parameter) whitening geometries.

The input side of the network: maps raw cosmological parameters to the
decorrelated, unit-variance vector the model consumes (encode) and back
(decode). ParamGeometry is the base: center, rotate into the covmat
eigenbasis, unit-scale. LogParamGeometry whitens in log space for the
multiplicative parameters; NLAInputGeometry and AmplitudeFactorGeometry
whiten every parameter except the intrinsic-alignment amplitude(s), which
they append raw so the loss can apply them in closed form (the factored-IA
emulator).

PS: to whiten is to rotate into the covariance eigenbasis and scale each
direction to unit variance, so correlated quantities become decorrelated
and equally scaled. encode = center the raw input, then whiten it; decode
is its exact inverse.

    theta (B, n_param)       raw cosmological parameters
       │  - center           subtract the parameter means
       ▼
       │  rotate             into the parameter-covmat eigenbasis
       ▼
       │  unit-scale         divide by each direction's spread
       ▼
    x  (B, encoded_dim)      whitened model inputs

(legend: B = batch rows; n_param = number of sampled parameters;
encoded_dim = the model-input width -- equal to n_param for the plain
geometries, larger for the factored ones, which whiten everything
except the IA amplitude(s) and append those raw at the end for the
loss's closed-form combine. decode runs the arrows bottom-up, exactly
inverting each step.)
"""

import numpy as np
import torch


class ParamGeometry:
  """
  Whitening transform for the cosmological parameters.

  Whitening (see module PS) is applied to the input
  parameters, so the emulator sees decorrelated, unit-variance
  inputs instead of strongly-correlated physical parameters.
  Build from a covmat file at training time (from_covmat) or
  from saved tensors at inference time (from_state); the
  transform travels with the weights. center is the training
  mean of the parameters, subtracted before whitening.
  """

  def __init__(self, 
               device,
               names, 
               center,
               evecs, 
               sqrt_ev):
    """
    Place the transform tensors on the device.

    Plain constructor: stores fields only; the two
    classmethods below build them. as_tensor accepts numpy
    (from a covmat) or cpu tensors (from a saved state).

    Arguments:
      device  = device the tensors live on.
      names   = parameter column order (the covmat header
                names), kept for the record and to check
                alignment against C0's columns.
      center  = training mean of the parameters, the
                zero-point subtracted before whitening.
      evecs   = eigenvectors of the parameter covariance
                (the rotation; columns orthonormal).
      sqrt_ev = square roots of the covariance eigenvalues
                (the per-direction whitening scale).
    """
    self.names   = list(names)

    self.center  = torch.as_tensor(center, 
                                   dtype=torch.float32, 
                                   device=device)
    self.evecs   = torch.as_tensor(evecs, 
                                   dtype=torch.float32, 
                                   device=device)
    self.sqrt_ev = torch.as_tensor(sqrt_ev, 
                                   dtype=torch.float32, 
                                   device=device)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    state's keys match __init__, so cls(device, **state)
    reconstructs the transform with no covmat reread.

    cls is this class, ParamGeometry: the standard first
    argument of a classmethod (the class itself, as self is
    the instance in a normal method), so cls(...) runs
    __init__ and returns a new instance.
    """
    return cls(device, **state)

  @classmethod
  def from_covmat(cls, device, center, covmat_path):
    """
    Build the transform from a covmat file (training).

    The covmat columns are the emulated parameters in C0's
    order (sibling files from one run). Reads the header
    names for the record, then eigendecomposes the symmetric
    covariance, cov = V diag(lam) V^T, with V orthonormal and
    eigenvalues lam > 0 -- V is the rotation, sqrt(lam) the
    whitening scale. cls(...) is ParamGeometry(...): runs
    __init__ and returns a new instance.

    Arguments:
      device      = device for the built tensors.
      center      = training mean of the parameters.
      covmat_path = path to the covmat file; its first line
                    is a "#"-prefixed list of column names.
    """
    with open(covmat_path) as f:
      names = f.readline().lstrip("#").split()
    cov = np.loadtxt(covmat_path)
    lam, V = np.linalg.eigh(cov)
    return cls(device=device, names=names, center=center, evecs=V, sqrt_ev=np.sqrt(lam))

  def state(self):
    """Tensors to save; keys match __init__."""
    return {"names": self.names,
            "center":  self.center.cpu(),
            "evecs":   self.evecs.cpu(),
            "sqrt_ev": self.sqrt_ev.cpu()}

  def whiten(self, x):
    """
    Rotate into the eigenbasis; scale to unit variance.

    x @ evecs rotates into the covariance eigenbasis (shapes
    (B, n) @ (n, n) -> (B, n)); dividing by sqrt_ev scales
    each direction to unit variance, a decorrelated vector.
    """
    return (x @ self.evecs) / self.sqrt_ev

  def unwhiten(self, a):
    """
    Exact inverse of whiten.

    Multiply by sqrt_ev and rotate back (@ evecs.T); evecs is
    orthonormal, so this inverts whiten exactly.
    """
    return (a * self.sqrt_ev) @ self.evecs.T

  def encode(self, theta):
    """Raw params -> network input: center, then whiten."""
    return self.whiten(theta - self.center)

  def decode(self, a):
    """Network input -> raw params: unwhiten, add center."""
    return self.unwhiten(a) + self.center


class LogParamGeometry(ParamGeometry):
  """
  ParamGeometry that whitens in log space for the positive,
  multiplicatively-acting parameters (linear for additive
  nuisances). The dv depends on those through products and powers
  (A_s, (Om h^2)^ns, 1/h, ...), linear in log -- so log inputs
  hand the network a flatter, lower-effective-DOF map (the
  hardness lever), aimed at the A_s / Om h^2 structure direction
  the hardness regression flagged.

  log_mask[i] = True -> ln(param) before centering+whitening (exp
  on the way back). Defaults log A_s, H0, Omega_m, Omega_b. n_s
  stays linear on purpose: the dv depends on k^ns, so n_s is the
  exponent, not a multiplicative factor -- logging it would be
  wrong. DZ / A1 stay linear too (they can be <= 0). center +
  basis are computed in the transformed space, hence from_samples
  (no precomputed log covmat).
  """

  def __init__(self, device, names, center, evecs, sqrt_ev,
               log_mask):
    super().__init__(device, names, center, evecs, sqrt_ev)
    self.log_mask = torch.as_tensor(
      log_mask, dtype=torch.bool, device=device)

  @classmethod
  def from_samples(cls, device, samples, names,
                   log_names=("As_1e9", "H0", "omegam",
                              "omegab")):
    """
    Build from raw training parameter samples.

    Arguments:
      device    = device for the tensors.
      samples   = (N, n_param) raw physical training params.
      names     = parameter column names (covmat order).
      log_names = which params to ln-transform (positive,
                  multiplicative in the dv). Empty () gives a
                  plain linear geometry built from samples.
    Returns:
      a LogParamGeometry; center / whitening basis live in the
      mixed log/linear space.
    """
    names = list(names)
    # log_mask[i] = True if names[i] is a log-transformed param.
    log_vals = []
    for n in names:
      log_vals.append(n in log_names)
    log_mask = np.array(log_vals)
    X = np.asarray(samples, dtype="float64")
    assert (X[:, log_mask] > 0).all(), \
      "logged params must be strictly positive"

    # transform the logged columns, then center in that space.
    Xt = X.copy()
    Xt[:, log_mask] = np.log(Xt[:, log_mask])
    center = Xt.mean(0)

    lam, V = np.linalg.eigh(np.cov(Xt, rowvar=False))
    return cls(device=device,
               names=names,
               center=center,
               evecs=V,
               sqrt_ev=np.sqrt(lam),
               log_mask=log_mask)

  def state(self):
    s = super().state()
    s["log_mask"] = self.log_mask.cpu()
    return s

  def _to_t(self, theta):
    # raw params -> transformed (ln on the logged columns).
    t = theta.clone()
    t[:, self.log_mask] = torch.log(theta[:, self.log_mask])
    return t

  def _from_t(self, t):
    # transformed -> raw (exp on the logged columns).
    out = t.clone()
    out[:, self.log_mask] = torch.exp(t[:, self.log_mask])
    return out

  def encode(self, theta):
    return self.whiten(self._to_t(theta) - self.center)

  def decode(self, a):
    return self._from_t(self.unwhiten(a) + self.center)


class NLAInputGeometry:
  """
  Input whitening for the factored NLA emulator. Whitens the 11
  parameters except the IA amplitude A1_1 (which factors out
  exactly) and appends the raw A1_1 as the last column, so the
  loss can apply the A1_1 polynomial. The templates must not see
  A1_1 -- else the model could absorb A1_1 dependence into them
  and the exact-A1_1 generalization would be lost.

  encode(raw_12) -> (B, 12): [11 whitened non-A1_1 params ; raw
  A1_1]. The model reads [:, :-1]; the loss reads [:, -1].
  """
  def __init__(self, device, pg11, idx_a1, n_param):
    """Store the split fields (the classmethod builds them).

    Arguments:
      device  = device the index tensor lives on.
      pg11    = ParamGeometry that whitens the 11 non-A1_1
                parameters.
      idx_a1  = column index of A1_1 in the raw (n_param)-wide
                parameter vector.
      n_param = total number of raw parameters (here 12).
    """
    self.pg11    = pg11
    self.idx_a1  = idx_a1
    self.n_param = n_param
    # keep = column indices that are not A1_1 (the 11 the model
    # sees). A long tensor so it can index a tensor's columns.
    keep = []
    for j in range(n_param):
      if j != idx_a1:
        keep.append(j)
    self.keep = torch.tensor(keep, dtype=torch.long,
                             device=device)

  @classmethod
  def from_covmat(cls, device, center, covmat_path, a1_name):
    """Build the input geometry from the parameter covmat.

    Reads the covmat header for the column names, drops the
    A1_1 row/column, and eigendecomposes the remaining 11x11
    sub-covariance for the inner ParamGeometry that whitens the
    non-A1_1 parameters.

    Arguments:
      device      = device for the built tensors.
      center      = full (n_param,) training-mean parameters;
                    its 11 non-A1_1 entries center the inner
                    whitening.
      covmat_path = path to the covmat file; first line is a
                    "#"-prefixed list of column names.
      a1_name     = name of the A1_1 column to factor out (here
                    "LSST_A1_1").

    Returns:
      an NLAInputGeometry whose encode whitens the 11 non-A1_1
      params and appends raw A1_1.
    """
    with open(covmat_path) as f:
      names = f.readline().lstrip("#").split()
    cov    = np.loadtxt(covmat_path)
    idx_a1 = names.index(a1_name)

    # keep = the 11 non-A1_1 columns, in original order.
    keep = []
    for j in range(len(names)):
      if j != idx_a1:
        keep.append(j)
    # 11x11 sub-covariance and the 11 sub-means (A1_1 removed).
    cov11  = cov[np.ix_(keep, keep)]
    cen    = (center.detach().cpu().numpy()
              if torch.is_tensor(center)
              else np.asarray(center))[keep]

    # eigendecompose the sub-cov -> inner whitening basis.
    lam, V = np.linalg.eigh(cov11)
    kept_names = []
    for j in keep:
      kept_names.append(names[j])
    pg11 = ParamGeometry(device, kept_names,
                         cen, V, np.sqrt(lam))

    return cls(device=device, pg11=pg11, idx_a1=idx_a1, n_param=len(names))

  def encode(self, theta):
    """Raw parameters -> model input with A1_1 carried along.

    Arguments:
      theta = (B, n_param) raw physical parameters, one row per
              cosmology, columns in covmat order.

    Returns:
      (B, n_param): the 11 non-A1_1 parameters whitened, raw
      A1_1 appended as the last column (model reads [:, :-1],
      loss reads [:, -1]).
    """
    w11 = self.pg11.encode(theta[:, self.keep])   # (B, 11)
    a1  = theta[:, self.idx_a1:self.idx_a1 + 1]    # (B, 1) raw
    return torch.cat([w11, a1], dim=1)             # (B, n_param)

  def decode(self, enc):
    """Inverse of encode: model input + A1_1 -> raw parameters.

    Arguments:
      enc = (B, n_param) encoded vector from encode
            ([11 whitened ; raw A1_1]).

    Returns:
      (B, n_param) raw physical parameters in covmat order
      (un-whiten the 11, reinsert A1_1 at its column).
    """
    raw11 = self.pg11.decode(enc[:, :-1])          # (B, 11) raw
    a1    = enc[:, -1:]                             # (B, 1) raw
    out = torch.empty(enc.shape[0], self.n_param,
                      dtype=enc.dtype, device=enc.device)
    out[:, self.keep] = raw11
    out[:, self.idx_a1:self.idx_a1 + 1] = a1
    return out


class AmplitudeFactorGeometry:
  """
  Input whitening for a factored intrinsic-alignment emulator.
  Whitens every parameter except the IA amplitudes (which factor
  out of the data vector exactly, as a polynomial) and appends
  the raw amplitudes as the last columns, so the loss can apply
  that polynomial. The templates must not see the amplitudes --
  else the model could absorb amplitude dependence into them and
  the exact, prior-width-independent amplitude generalization
  would be lost.

  Generalizes the single-amplitude NLA case to any number of
  amplitudes: NLA factors out [A1_1] (1); TATT factors out
  [a1, a2, b_TA] (3). The redshift-evolution powers (eta; the
  NLA A1_2, the TATT eta1/eta2) stay in the whitened input --
  they sit inside the projection integral and do not factor.

  encode(raw) -> (B, n_param): [whitened non-amplitude params ;
  raw amplitudes]. The model reads [:, :-n_amps]; the loss reads
  [:, -n_amps:].
  """
  def __init__(self, device, pg_keep, amp_idx, n_param,
               names=None):
    """Store the split fields (the classmethod builds them).

    Arguments:
      device    = device the index tensors live on.
      pg_keep   = ParamGeometry that whitens the kept parameters.
      amp_idx   = list of amplitude column indices in the raw
                  parameter vector, in the order the coeff_fn
                  expects (e.g. [a1, a2, b_TA] for TATT).
      n_param   = total number of raw parameters.
      names     = full raw-order parameter names (the covmat
                  header); the diagnostics read them off the
                  geometry, like ParamGeometry.names.
    """
    self.pg_keep = pg_keep
    self.n_param = n_param
    self.names   = list(names) if names is not None else None
    # normalize to plain ints first: amp_idx arrives as a list from
    # the classmethods but as a saved tensor from from_state, and set
    # membership on tensor elements is identity-based (hash, not
    # value), which would silently keep every column below.
    idx_list = []
    for a in amp_idx:
      idx_list.append(int(a))
    self.n_amps  = len(idx_list)
    # amplitude columns, in coeff_fn order (appended as-is).
    self.amp_idx = torch.tensor(idx_list, dtype=torch.long,
                                device=device)
    # keep = every column that stays in the whitened input block
    # (the non-amplitudes).
    drop = set(idx_list)
    keep = []
    for j in range(n_param):
      if j not in drop:
        keep.append(j)
    self.keep = torch.tensor(keep, dtype=torch.long,
                             device=device)

  @property
  def encoded_dim(self):
    """Width of encode()'s output: the whitened block plus the
    appended raw amplitudes (== n_param here; the property is the
    geometry's own statement of its output width, which
    run_emulator reads instead of assuming the raw count)."""
    return int(self.keep.numel()) + self.n_amps

  @classmethod
  def from_covmat(cls, device, center, covmat_path, amp_names):
    """Build the input geometry from the parameter covmat.

    Reads the covmat header for the column names, drops the
    amplitude rows/columns, and eigendecomposes the remaining
    sub-covariance for the inner ParamGeometry that whitens the
    non-amplitude parameters.

    Arguments:
      device      = device for the built tensors.
      center      = full (n_param,) training-mean parameters;
                    its non-amplitude entries center the inner
                    whitening.
      covmat_path = path to the covmat file; first line is a
                    "#"-prefixed list of column names.
      amp_names   = list of amplitude column names to append for
                    the loss, in coeff_fn order (NLA:
                    ["LSST_A1_1"]; TATT: the a1/a2/b_TA names).

    Returns:
      an AmplitudeFactorGeometry whose encode whitens the
      non-amplitude params and appends the raw amplitudes.
    """
    with open(covmat_path) as f:
      names = f.readline().lstrip("#").split()
    cov     = np.loadtxt(covmat_path)
    amp_idx = []
    for a in amp_names:
      amp_idx.append(names.index(a))
    drop = set(amp_idx)

    keep = []
    for j in range(len(names)):
      if j not in drop:
        keep.append(j)
    cov_k   = cov[np.ix_(keep, keep)]
    cen     = (center.detach().cpu().numpy()
               if torch.is_tensor(center)
               else np.asarray(center))[keep]

    lam, V  = np.linalg.eigh(cov_k)
    kept_names = []
    for j in keep:
      kept_names.append(names[j])
    pg_keep = ParamGeometry(device, kept_names,
                            cen, V, np.sqrt(lam))

    return cls(device=device, pg_keep=pg_keep, amp_idx=amp_idx,
               n_param=len(names), names=names)

  @classmethod
  def from_state(cls, device, state):
    """Rebuild from a saved state dict (inference path).

    state's keys match __init__ (the nested "pg_keep" dict rebuilds
    through ParamGeometry.from_state), so no covmat reread.
    """
    return cls(device=device,
               pg_keep=ParamGeometry.from_state(device,
                                                state["pg_keep"]),
               amp_idx=state["amp_idx"],
               n_param=state["n_param"],
               names=state.get("names"))

  def state(self):
    """Tensors to save; keys match __init__ (pg_keep nests the
    kept-column ParamGeometry's own state)."""
    return {"pg_keep": self.pg_keep.state(),
            "amp_idx": self.amp_idx.cpu(),
            "n_param": self.n_param,
            "names": self.names}

  def encode(self, theta):
    """Raw parameters -> model input with amplitudes appended.

    Arguments:
      theta = (B, n_param) raw physical parameters, one row per
              cosmology, columns in covmat order.

    Returns:
      (B, n_param): the non-amplitude params whitened, raw
      amplitudes appended as the last n_amps columns (model
      reads [:, :-n_amps], loss reads [:, -n_amps:]).
    """
    w    = self.pg_keep.encode(theta[:, self.keep])
    amps = theta[:, self.amp_idx]              # (B, n_amps) raw
    return torch.cat([w, amps], dim=1)

  def decode(self, enc):
    """Inverse of encode: model input + amplitudes -> raw params.

    Arguments:
      enc = (B, n_param) encoded vector from encode
            ([whitened non-amplitude ; raw amplitudes]).

    Returns:
      (B, n_param) raw physical parameters in covmat order.
    """
    raw_keep = self.pg_keep.decode(enc[:, :-self.n_amps])
    amps     = enc[:, -self.n_amps:]
    out = torch.empty(enc.shape[0], self.n_param,
                      dtype=enc.dtype, device=enc.device)
    out[:, self.keep]    = raw_keep
    out[:, self.amp_idx] = amps
    return out
