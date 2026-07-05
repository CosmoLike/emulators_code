"""Run-output I/O: learning-curve tables and trained-emulator files.

save_learning_curves writes a whitespace-delimited table (row per N_train,
column per curve, "#"-comment header carrying the config) that np.loadtxt
reads back, the format the sweep and bake-off drivers save, so several
runs can be overlaid later. save_emulator persists a trained run as two
files: <root>.emul, the model weights (a torch state_dict, cpu tensors),
and <root>.h5, everything inference or a paper trail needs (both
whitening geometries, the training histories, and the full config).

PS: state_dict = torch's name -> tensor mapping of a model's learnable
parameters and buffers; whitening = the center/rotate/scale transform the
geometries apply to parameters (input) and data vectors (output).
"""

import time

import numpy as np
import torch
import yaml


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


def save_sweep_table(path, param, values, fracs, meta=None):
  """
  Write a one-hyperparameter sweep as a whitespace-delimited table.

  The generic twin of save_learning_curves for an arbitrary swept
  knob. Numeric values become the first data column; categorical
  values (strings, or booleans -- a film on/off sweep) become an
  integer index column with the label map on a "# values:" comment
  line, so the body stays np.loadtxt-loadable either way. Layouts:

      # sweep: f(delta-chi2 > threshold) vs lr.lr_base
      # model=rescnn  threshold=0.2  n_train=250000
      # columns: lr.lr_base, frac
      0.001  0.401234
      ...

      # sweep: f(delta-chi2 > threshold) vs model.activation
      # values: 0=H, 1=power, 2=multigate
      # columns: index, frac
      0  0.401234
      ...

  Arguments:
    path   = output text-file path.
    param  = the swept hyperparameter's dotted YAML path (names the
             x column).
    values = the swept values, one per row (numbers, strings, or
             booleans).
    fracs  = per-value fractions aligned with `values`.
    meta   = optional mapping written as a "# key=val" line; None
             to omit.
  """
  # booleans pass isinstance(v, int) in Python; check them first so
  # a True/False sweep is labeled, not silently cast to 1/0.
  numeric = True
  for v in values:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      numeric = False
  lines = [f"# sweep: f(delta-chi2 > threshold) vs {param}"]
  if meta:
    pairs = []
    for k, v in meta.items():
      pairs.append(f"{k}={v}")
    lines.append("# " + "  ".join(pairs))
  if numeric:
    lines.append(f"# columns: {param}, frac")
    for v, f in zip(values, fracs):
      lines.append(f"{float(v):.8g}  {f:.6f}")
  else:
    labels = []
    for i, v in enumerate(values):
      labels.append(f"{i}={v}")
    lines.append("# values: " + ", ".join(labels))
    lines.append("# columns: index, frac")
    for i, f in enumerate(fracs):
      lines.append(f"{i:d}  {f:.6f}")
  with open(path, "w") as f:
    f.write("\n".join(lines) + "\n")


def save_emulator(path_root,
                  model,
                  param_geometry,
                  geometry,
                  config,
                  histories,
                  train_args=None,
                  attrs=None):
  """
  Persist a trained emulator as <path_root>.emul + <path_root>.h5.

  The .emul holds only the model weights: torch.save of the
  state_dict with every tensor moved to cpu, so it loads on any
  machine (a cuda-saved state needs the saving GPU visible). A
  torch.compile'd model wraps the real one and prefixes every
  state_dict key with "_orig_mod."; the prefix is stripped so the
  saved keys always match the plain architecture.

  The .h5 holds everything else, grouped:
    param_geometry/  the input-whitening state, keys exactly
                     ParamGeometry.state() (names, center, evecs,
                     sqrt_ev), so ParamGeometry.from_state rebuilds
                     it with no covmat reread.
    dv_geometry/     the output-geometry state, keys exactly
                     DataVectorGeometry.state() (total_size,
                     dest_idx, evecs, sqrt_ev, Cinv, center, dtype),
                     so from_state rebuilds it with no cosmolike.
    history/         per-epoch training curves: train_losses,
                     val_medians, val_means, val_fracs (one row per
                     epoch, one column per threshold), thresholds.
    config_yaml      the driver's resolved config (data + train_args
                     blocks), as YAML text.
    train_args_yaml  the collapsed train_args actually used (search
                     ranges resolved to their defaults), as YAML.
  plus one root attribute per entry of `attrs` (run identity:
  model name, activation, rescale, N_train, best epoch, ...), a
  "created" timestamp, and the torch version.

  Arguments:
    path_root      = output path without extension; writes
                     <path_root>.emul and <path_root>.h5.
    model          = the trained network (best-epoch weights already
                     restored by the training loop).
    param_geometry = the input ParamGeometry (its .state() is saved).
    geometry       = the output DataVectorGeometry (its .state() is
                     saved). Pass chi2fn.geom, not the chi2fn.
    config         = the resolved config mapping (data + train_args),
                     stored verbatim as YAML text.
    histories      = mapping with the per-epoch lists the training
                     loop returned: "train_losses", "val_medians",
                     "val_means", "val_fracs" (list of per-threshold
                     tensors), "thresholds".
    train_args     = the collapsed train_args the run actually used
                     (search ranges resolved), or None to omit.
    attrs          = optional mapping of scalar run metadata, each
                     written as one h5 root attribute.

  Returns:
    (emul_path, h5_path), the two files written.
  """
  # h5py lives only here: the training machines (cocoa) ship it, the
  # plotting/train paths never need it.
  import h5py

  # --- <root>.emul: the weights, cpu, unprefixed ---
  sd = {}
  for k, v in model.state_dict().items():
    # a torch.compile wrapper (OptimizedModule) stores the real
    # model as ._orig_mod, so its keys arrive prefixed; strip it.
    sd[k.removeprefix("_orig_mod.")] = v.detach().cpu()
  emul_path = path_root + ".emul"
  torch.save(sd, emul_path)

  # --- <root>.h5: geometries + histories + config + identity ---
  h5_path = path_root + ".h5"
  str_dt  = h5py.string_dtype(encoding="utf-8")

  def write_state(group, state):
    # Write one geometry state() dict recursively, so ANY geometry
    # saves without per-class code: tensors -> datasets, name lists
    # -> string datasets, nested dicts (a composed geometry, e.g.
    # AmplitudeFactorGeometry's pg_keep) -> subgroups, scalars and
    # dtypes -> attributes. Keys stay exactly state()'s, so the
    # matching from_state rebuilds from a read-back dict.
    for k, v in state.items():
      if isinstance(v, dict):
        write_state(group.create_group(k), v)
      elif torch.is_tensor(v):
        group.create_dataset(k, data=v.cpu().numpy())
      elif isinstance(v, list):
        group.create_dataset(k,
                             data=np.asarray(v, dtype=object),
                             dtype=str_dt)
      elif isinstance(v, (int, float)):
        group.attrs[k] = v
      else:
        group.attrs[k] = str(v)   # torch.dtype and friends

  with h5py.File(h5_path, "w") as f:
    # input whitening, keys exactly param_geometry.state().
    write_state(f.create_group("param_geometry"),
                param_geometry.state())

    # output geometry, keys exactly geometry.state().
    write_state(f.create_group("dv_geometry"), geometry.state())

    # per-epoch histories; fracs stack to (nepochs, n_thresholds).
    hg = f.create_group("history")
    hg.create_dataset("train_losses",
                      data=np.asarray(histories["train_losses"]))
    hg.create_dataset("val_medians",
                      data=np.asarray(histories["val_medians"]))
    hg.create_dataset("val_means",
                      data=np.asarray(histories["val_means"]))
    rows = []
    for fr in histories["val_fracs"]:
      rows.append(np.asarray(fr.cpu() if torch.is_tensor(fr) else fr))
    hg.create_dataset("val_fracs", data=np.stack(rows))
    thr = histories["thresholds"]
    if torch.is_tensor(thr):
      thr = thr.cpu().numpy()
    hg.create_dataset("thresholds", data=np.asarray(thr))

    # the full configs, verbatim, as YAML text.
    f.create_dataset("config_yaml",
                     data=yaml.safe_dump(config, sort_keys=False),
                     dtype=str_dt)
    if train_args is not None:
      f.create_dataset("train_args_yaml",
                       data=yaml.safe_dump(train_args,
                                           sort_keys=False),
                       dtype=str_dt)

    # run identity + provenance as root attributes. str() guards the
    # str subclasses h5py rejects (torch.__version__ is one: numpy
    # coerces a str subclass to a fixed-width unicode dtype h5py has
    # no conversion for; a plain str stores as variable-length utf8).
    if attrs is not None:
      for k, v in attrs.items():
        f.attrs[k] = str(v) if isinstance(v, str) else v
    f.attrs["created"]       = time.strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["torch_version"] = str(torch.__version__)

  return emul_path, h5_path
