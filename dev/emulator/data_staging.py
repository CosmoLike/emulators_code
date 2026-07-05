"""Raw data loading, streaming statistics, and the physical cut.

The bottom of the pipeline: turns the on-disk parameter (.txt) and
data-vector (.npy) dumps into the in-memory "source" dicts the rest of the
package consumes, never loading the (memmap-sized) dv file whole.
stream_chunks, stream_stats, and param_stats compute per-column
normalization stats over selected rows; stage_source materializes a row
subset in RAM if it fits (else keeps the memmap); phys_cut_idx applies the
physical cuts (omega_b h^2 below a bound, omegam^2 h^2 inside an optional
window); read_param_names reads parameter names off a covmat
header. load_source orchestrates: memmap, cut, size, stage one source into
a {C, dv, idx, (+means)} dict.

The staging pipeline, per source (load_source top to bottom):

    <params>.txt                    <dv>.npy
       │  np.loadtxt                   │  np.load(mmap_mode="r")
       ▼                               ▼
    C  (N, cols)                    dv (N, total_size), on disk
       │  phys_cut_idx: omegabh2 bound (+ optional lower bound)
       │  and the optional omegam2h2 window
       ▼
    pool = surviving row indices
       │  keep N // divisor rows (or n_keep), one seeded shuffle
       ▼
    idx = this run's rows
       │  stage_source: do the used rows fit ram_frac of free RAM?
       │    yes -> materialize C[idx] / dv[idx], reindex local
       │    no  -> keep the memmap, idx stays global
       ▼
    source dict {C, dv, idx, C_mean, dv_mean}

    (legend: N = rows in the dump; cols = the .txt columns
     (weight, lnp, params, chi2); total_size = full dv length;
     divisor / n_keep = the size knobs, exactly one given;
     C_mean / dv_mean = training-subset means the geometries
     center on; local reindex = idx becomes arange so the staged
     arrays and the loaders agree on row numbering.)

PS: a dump is the full on-disk array from the data-generation run, every
simulated cosmology stored as one row (the data-vector dump is the .npy
file, the parameter dump the .txt); a training run draws its N_train
subset of rows from it. a memmap (memory-mapped array) is a NumPy array
backed by the file on disk and read in slices, so an array larger than RAM
is never loaded whole.
"""

import os

import numpy as np
import psutil
import torch


def stream_chunks(idx, chunk):
  """
  Yield the row indices in sorted blocks of `chunk` rows.

  A generator (it `yield`s, so blocks are produced lazily). Each
  block is sorted so that indexing a memmap with it walks the
  file in increasing order -- sequential disk access, not random
  seeks.

  Arguments:
    idx   = 1D array of row indices (any order).
    chunk = number of indices per block.

  Yields:
    a sorted sub-array of up to `chunk` indices.
  """
  # step through idx in windows of `chunk` (the last may be
  # short); np.sort orders each window for sequential reads.
  for a in range(0, len(idx), chunk):
    yield np.sort(idx[a:a+chunk])


def stream_stats(mm, idx, method=1, CHUNK=10000):
  """
  Per-column normalization stats over a chosen subset of rows.

  Context: a run uses only the N_train subset of the dump. `mm`
  is a row-indexable view of the data vectors (the on-disk dump,
  a memmap larger than RAM, or its staged in-RAM subset), and
  `idx` the rows used. Stats accumulate over just those rows,
  streamed CHUNK at a time, so `mm` is never fully loaded.
  `method` picks the scheme:
    1 = z-score  -> returns (mean, std)
    2 = min-max  -> returns (min,  max - min)
  The caller then normalizes a row as (x - offset) / scale.

  Arguments:
    mm     = 2D array indexable by row (in-RAM or memmap);
             columns are the quantities to summarize.
    idx    = the rows of `mm` to include (the N_train subset).
    method = 1 for z-score, 2 for min-max.
    CHUNK  = rows read per streamed block.

  Returns:
    (offset, scale) as float32 torch tensors, one per column.
  """
  n = len(idx)               # total rows summarized
  ncols = mm.shape[1]        # one stat per column

  if method == 1:
    # one-pass mean/variance via running sums. float64
    # accumulators keep precision and avoid overflow over many
    # rows.
    s1 = np.zeros(ncols, dtype="float64")   # sum of x
    s2 = np.zeros(ncols, dtype="float64")   # sum of x^2
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      # read this block and upcast to float64.
      x = np.asarray(mm[rows], dtype="float64")
      s1 += x.sum(axis=0)            # accumulate sum
      s2 += (x * x).sum(axis=0)      # accumulate sum of sq

    mean = s1 / n
    # variance = (sum_sq - sum^2/n) / (n-1): the one-pass form of
    # the unbiased sample variance; sqrt gives the std.
    std  = np.sqrt((s2 - s1 * s1 / n) / (n - 1))
    offset, scale = mean, std
  elif method == 2:
    # running min/max, started at +inf / -inf so the first block
    # replaces them.
    mn = np.full(ncols,  np.inf, dtype="float64")
    mx = np.full(ncols, -np.inf, dtype="float64")
    for rows in stream_chunks(idx=idx, chunk=CHUNK):
      x = np.asarray(mm[rows], dtype="float64")
      mn = np.minimum(mn, x.min(axis=0))   # tighten the min
      mx = np.maximum(mx, x.max(axis=0))   # tighten the max
    offset, scale = mn, mx - mn

  # hand back float32 torch tensors (the model's dtype).
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def param_stats(arr, idx, method=1):
  """
  Per-column normalization stats for the cosmo params.

  The caller normalizes as (x - offset) / scale. Sums run in
  float64 for accurate totals; the returned tensors are float32
  (the model's dtype).

  Arguments:
    arr    = the parameter array (or memmap), row per sample.
    idx    = row indices to compute the stats over (the training
             subset; never the validation rows).
    method = 1 -> z-score: returns (mean, std);
             2 -> min-max: returns (min, max - min).

  Returns:
    (offset, scale) float32 torch tensors, one value per column.
  """
  a = np.asarray(arr[idx], dtype="float64")
  if method == 1:
    offset = a.mean(axis=0)
    scale  = a.std(axis=0, ddof=1)
  elif method == 2:
    offset = a.min(axis=0)
    scale  = a.max(axis=0) - offset
  off = torch.from_numpy(offset.astype("float32"))
  scl = torch.from_numpy(scale.astype("float32"))
  return off, scl


def stage_source(C, dv, idx, ram_frac=0.7):
  """
  Stage a source's used rows in RAM if they fit, else leave them
  on disk.

  A run uses only the N_train subset of the dump, named by `idx`.
  That subset is far smaller than the dump and usually fits in
  RAM even when the dump does not. If its bytes are below
  ram_frac of available RAM, materialize the compact subset (C
  and dv restricted to the used rows) and reindex locally;
  otherwise return the inputs unchanged so the loaders stream dv
  from the memmap by global index. Either way idx matches its own
  C/dv, so the rest of the pipeline is identical.

  Arguments:
    C        = full parameter dump, (N, Ncosmo).
    dv       = full dv dump, (N, Ndv); ndarray or np.memmap.
    idx      = the rows this run uses, as global row indices.
    ram_frac = fraction of available RAM the materialized subset
               may occupy (default 0.7).

  Returns:
    C_src, dv_src, idx_src = compact in-RAM subset with idx_src
      = arange(n_used) when it fits; otherwise (C, dv, idx)
      unchanged (dv still the memmap, idx still global).
  """
  rows   = np.sort(np.unique(idx))      # sorted -> sequential
  nbytes = rows.size * dv.shape[1] * dv.dtype.itemsize
  avail  = psutil.virtual_memory().available
  if nbytes < ram_frac * avail:
    # materialize into RAM, reindex locally.
    return (np.asarray(C[rows]),
            np.asarray(dv[rows]),
            np.arange(rows.size))
  # too big for RAM: keep full arrays + global index, stream dv
  # from disk.
  return C, dv, idx


def phys_cut_idx(C, idx, names, cut, omegabh2_lo=None,
                 omegam2h2_lo=None, omegam2h2_hi=None):
  """
  Keep only rows inside the physical-density cuts.

  Two cuts, each on a derived product a per-parameter scan misses:

    omega_b h^2  = Omega_b * (H0/100)^2   inside (omegabh2_lo, cut)
    omegam^2 h^2 = (Omega_m * H0/100)^2   inside (omegam2h2_lo,
                                                  omegam2h2_hi)

  The high-omega_b h^2 cosmologies (a sparse, ~2x Planck corner)
  fail catastrophically and no real posterior visits them.
  omegam^2 h^2 is Gamma^2, the transfer-shape parameter squared,
  and it is the coordinate along the sampling cloud's long axis:
  both of its tails are rarefied (few training neighbours) and
  carry the catastrophic failures, so the window keeps the
  well-covered core around Planck's Gamma^2 ~ 0.045. Either bound
  may be None (that side is not cut), so old configs without the
  window keys behave exactly as before.

  Arguments:
    C     = full parameter dump, (N, n_param), physical units,
            column order given by `names`.
    idx   = candidate row indices into C (e.g. a shuffle).
    names = parameter column names in C's column order; locate
            the omegab, omegam, and H0 columns by name.
    cut   = upper bound on omega_b h^2 (rows >= cut dropped).
    omegabh2_lo  = optional lower bound on omega_b h^2 (rows at or
                   below it dropped; None = no lower cut).
    omegam2h2_lo = optional lower bound on omegam^2 h^2 (rows at
                   or below it dropped; None = no lower cut).
    omegam2h2_hi = optional upper bound on omegam^2 h^2 (rows at
                   or above it dropped; None = no upper cut).

  Returns:
    the subset of idx passing every given cut, in idx's order.
  """
  i_ob = names.index("omegab")     # baryon density column
  i_h0 = names.index("H0")         # Hubble column (km/s/Mpc)
  obh2 = C[idx, i_ob] * (C[idx, i_h0] / 100.0) ** 2
  keep = obh2 < cut
  if omegabh2_lo is not None:
    keep &= obh2 > omegabh2_lo
  if omegam2h2_lo is not None or omegam2h2_hi is not None:
    i_om = names.index("omegam")   # matter density column
    #   omegam^2 h^2 = (Omega_m * H0/100)^2 = Gamma^2
    g2 = (C[idx, i_om] * C[idx, i_h0] / 100.0) ** 2
    if omegam2h2_lo is not None:
      keep &= g2 > omegam2h2_lo
    if omegam2h2_hi is not None:
      keep &= g2 < omegam2h2_hi
  return idx[keep]


def read_param_names(covmat_path, comment="#"):
  """
  Parameter column names from a covmat header line.

  Reads only the first line, strips the leading comment marker,
  splits on whitespace -- the column order the parameter arrays
  (and ParamGeometry) use.

  Arguments:
    covmat_path = path to the covmat file; its first line lists
                  the column names, prefixed by `comment`.
    comment     = the leading marker to strip (default "#").

  Returns:
    a list of parameter-name strings, in column order.
  """
  with open(covmat_path) as f:
    return f.readline().lstrip(comment).split()


def load_source(dv_path, params_path, names, cut, divisor=None,
                gen=None, ram_frac=0.7, with_means=False,
                param_cols=slice(2, -1), verbose=True, n_keep=None,
                omegabh2_lo=None,
                omegam2h2_lo=None, omegam2h2_hi=None):
  """
  Load, physically cut, and stage one dv/param source.

  Memmaps the dv dump (never reading it whole), keeps the
  modeled param columns, applies the physical cuts (omega_b h^2
  bound + the optional omegam^2 h^2 window), takes the
  first N // divisor cut rows of a fixed shuffle, stages that
  subset, and -- when with_means -- computes the centering means.
  Wraps phys_cut_idx / stage_source / stream_stats / param_stats.

  Arguments:
    dv_path     = .npy data-vector dump (memmapped).
    params_path = parameter text file; param_cols selects the
                  modeled columns.
    names       = parameter column names (covmat order, the kept
                  columns); phys_cut_idx finds the omegab / H0
                  columns by them.
    cut         = upper bound on omega_b h^2 (rows >= cut dropped).
    divisor     = keep N // divisor rows (10 -> ~1/10 for train);
                  pass this or n_keep (exactly one).
    gen         = torch.Generator seeding the cut+shuffle (required).
    ram_frac    = fraction of available RAM stage_source may fill
                  (default 0.7).
    with_means  = if True, also compute C_mean / dv_mean (train
                  needs them; val does not).
    param_cols  = column selector for the loaded params (default
                  slice(2, -1): drop the leading weight / lnp and
                  the trailing chi2 column).
    verbose     = if True (default), print a one-line summary
                  (shapes, rows, in-RAM).
    n_keep      = absolute rows to keep (overrides divisor; for a
                  learning-curve sweep at explicit sizes). Pass
                  this or divisor.
    omegabh2_lo  = optional lower bound on omega_b h^2 (None = no
                   lower cut; `cut` stays the upper bound).
    omegam2h2_lo = optional lower bound on omegam^2 h^2 (the
                   Gamma^2 window, see phys_cut_idx; None = no
                   lower cut).
    omegam2h2_hi = optional upper bound on omegam^2 h^2 (None =
                   no upper cut).

  Returns:
    a source dict {"C", "dv", "idx"} (plus "C_mean" / "dv_mean"
    when with_means), ready for build_loaders / run_emulator.
  """
  # require a generator and exactly one sizing rule.
  if gen is None:
    raise ValueError("load_source needs a torch.Generator (gen=)")
  if (divisor is None) == (n_keep is None):
    raise ValueError(
      "pass exactly one of divisor (keep N // divisor rows) or "
      "n_keep (an absolute row count)")
  dv = np.load(dv_path, mmap_mode="r", allow_pickle=False)
  # keep only the modeled parameter columns (see param_cols).
  C  = np.loadtxt(params_path, dtype="float32")[:, param_cols]
  if C.shape[0] != dv.shape[0]:
    raise ValueError(
      f"incompatible files: {params_path} has {C.shape[0]} rows, "
      f"{dv_path} has {dv.shape[0]}")

  n     = C.shape[0]
  order = torch.randperm(n, generator=gen).numpy()
  phys  = phys_cut_idx(C=C, idx=order, names=names, cut=cut,
                       omegabh2_lo=omegabh2_lo,
                       omegam2h2_lo=omegam2h2_lo,
                       omegam2h2_hi=omegam2h2_hi)
  # rows to keep: an absolute n_keep, or N // divisor.
  keep  = int(n_keep) if n_keep is not None else int(n // divisor)
  if len(phys) < keep:
    raise ValueError(
      f"physical pool too small: {len(phys)} < {keep}")
  idx = phys[:keep]

  # stage the cut rows in RAM if they fit, else keep the memmap.
  C_src, dv_src, idx_src = stage_source(
    C=C, dv=dv, idx=idx, ram_frac=ram_frac)
  src = {"C": C_src, "dv": dv_src, "idx": idx_src}
  if with_means:
    # the per-column std (2nd return) is unused: whitening comes
    # from the covmat, only the means center the targets.
    dv_mean, _ = stream_stats(mm=dv_src, idx=idx_src, method=1)
    c_mean,  _ = param_stats(arr=C_src, idx=idx_src, method=1)
    src["C_mean"]  = c_mean
    src["dv_mean"] = dv_mean

  if verbose:
    in_ram = not isinstance(dv_src, np.memmap)
    print(f"  {os.path.basename(dv_path)}: C {tuple(C_src.shape)} "
          f"dv {tuple(dv_src.shape)} used {idx_src.shape[0]} rows "
          f"| in RAM: {in_ram}")
  return src
