"""Plain-text I/O for sweep / learning-curve results.

save_learning_curves writes a whitespace-delimited table (row per N_train,
column per curve, "#"-comment header carrying the config) that np.loadtxt
reads back -- the format the sweep and bake-off drivers save, so several
runs can be overlaid later.
"""


def save_learning_curves(path, sizes, curves, meta=None):
  """
  Write learning curve(s) as a whitespace-delimited text table.

  A single config writes a one-entry `curves`; a bake-off writes all its
  curves to one file. Header lines are "#" comments np.loadtxt skips.
  Layout:

    # learning curve: f(delta-chi2 > threshold) vs N_train
    # model=ResMLP  rescale=none  threshold=0.2  pool=82000
    # columns: N_train, H, power, multigate, gated_power
    2000     0.401234  0.410512  0.395001  0.402310
    4203     ...

  Arguments:
    path   = output text-file path.
    sizes  = the N_train values, one per row (cast to int).
    curves = mapping label -> per-size fractions aligned with `sizes`
             (curves[label][i] is the value at sizes[i]). Labels become the
             data columns (dict order), documented on the "# columns:" line.
    meta   = optional mapping written as a "# key=val  key=val" line
             (model / rescale / threshold / pool); None to omit.
  """
  sizes  = list(sizes)
  labels = list(curves)
  lines  = ["# learning curve: f(delta-chi2 > threshold) vs N_train"]
  if meta:
    # one "# key=val  key=val ..." line (insertion order kept).
    pairs = []
    for k, v in meta.items():
      pairs.append(f"{k}={v}")
    lines.append("# " + "  ".join(pairs))
  # column header is a comment too (skipped on load); labels are
  # comma-separated to keep a label with spaces unambiguous.
  header = ["N_train"]
  for l in labels:
    header.append(str(l))
  lines.append("# columns: " + ", ".join(header))
  for i, n in enumerate(sizes):
    row = [f"{int(n):d}"]
    for l in labels:
      row.append(f"{curves[l][i]:.6f}")
    lines.append("  ".join(row))
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")
