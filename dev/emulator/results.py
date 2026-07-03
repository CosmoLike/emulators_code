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
  with h5py.File(h5_path, "w") as f:
    # input whitening, keys exactly ParamGeometry.state().
    pg = f.create_group("param_geometry")
    ps = param_geometry.state()
    pg.create_dataset("names",
                      data=np.asarray(ps["names"], dtype=object),
                      dtype=str_dt)
    for key in ("center", "evecs", "sqrt_ev"):
      pg.create_dataset(key, data=ps[key].numpy())

    # output geometry, keys exactly DataVectorGeometry.state().
    dg = f.create_group("dv_geometry")
    ds = geometry.state()
    dg.attrs["total_size"] = int(ds["total_size"])
    dg.attrs["dtype"]      = str(ds["dtype"])
    for key in ("dest_idx", "evecs", "sqrt_ev", "Cinv", "center"):
      dg.create_dataset(key, data=ds[key].numpy())

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
