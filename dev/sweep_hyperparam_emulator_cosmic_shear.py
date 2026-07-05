#!/usr/bin/env python3
"""Sweep ONE YAML-chosen hyperparameter at fixed N_train."""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# Sweeps a single hyperparameter -- chosen in the YAML, one value per training
# run at a fixed N_train -- and records validation f(delta-chi2 > threshold)
# per value, with the same table + figure outputs as the N_train sweep.
# sweep_ntrain_emulator_cosmic_shear.py stays the driver for the N_train axis
# (its points have unequal cost and its staging differs per point); this one
# covers every other knob: batch size, learning rate, kernel size, film
# on/off, the activation family, ...
#
#     python external_modules/code/emulators/emultrf/dev/sweep_hyperparam_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --out lrsweep_rescnn
#
#- The swept knob lives in a `sweep` block of the same training YAML (one YAML
#  serves every driver). `parameter` is the dotted path of ONE train_args leaf;
#  `values` the list to try:
#
#      sweep:
#        parameter: lr.lr_base
#        values:
#          - 0.0010
#          - 0.0025
#          - 0.0063
#
#  Any train_args leaf works by its dotted path: top-level knobs (bs, clip,
#  trunk_epochs), sub-block knobs (scheduler.patience, trim.start,
#  model.cnn.kernel_size, model.cnn.film, model.mlp.width), and per-phase
#  overrides (head.lr_base -- the block is created if the YAML omits it; the
#  usual trunk_epochs > 0 guard still applies). Values may be numbers, strings,
#  or booleans (film: [false, true]).
#
#  Two special cases:
#  - model.activation (or model.activation.type) sweeps the activation FAMILY:
#    the driver sets the experiment's resolved activation per value (the
#    --activation flag must then be left unset); values are the family names
#    (H, power, multigate, gated_power). model.activation.n_gates sweeps as an
#    ordinary leaf.
#  - model.name / model.ia are refused: they change the model CLASS, so
#    comparing architectures is one sweep per architecture (or the bake-off
#    driver), not one sweep across them.
#
#- Per value: a fresh model is trained on the SAME staged data and geometry
#  (staged once per worker -- the data does not vary across values, unlike the
#  N_train sweep) and scored as f(delta-chi2 > --threshold) on the fixed val
#  set.
#
#- Multiple GPUs (one node): values are near-equal-cost jobs, so they split
#  round-robin, one process per GPU (spawn; each worker stages its own copy
#  and streams from the shared dump memmap). One GPU (or the Apple-MPS dev
#  machine) falls back to a serial loop. `--gpu-pack` (off by default)
#  additionally co-locates up to 4 trainings per GPU when each is estimated
#  at <= 20% of the card (<= 40% -> 2) -- worth it on an H200, not on an
#  RTX 3060; see sweep_ntrain's header for the full rationale and caveats.
#
#- `--root` / `--fileroot` / `--yaml`: the cocoa layout, as in the training
#  driver (data under --root/chains, YAML + outputs under --fileroot).
#- `--rescale` / `--activation`: fixed across the sweep, as in the training
#  driver (--activation conflicts with sweeping model.activation).
#- `--n-gpus` (default: all visible CUDA devices): GPUs to spread across.
#- `--gpu-pack`: co-locate small trainings on one GPU (see above; off by
#  default).
#- `--threshold` (default 0.2): delta-chi2 cutoff the fraction counts.
#- `--out` (default hyperparam_sweep): writes <out>.txt (save_sweep_table:
#  numeric values as a value/frac table, categorical as index/frac with a
#  label map line) and <out>.pdf (plot_sweep_curve), under --fileroot.
#- `--quiet`: suppress stdout (txt and pdf still written).
#
#- Trains one full model per value -- run it on the workstation, where
#  cosmolike lives.
#-------------------------------------------------------------------------------

import argparse
import copy
import os
import time

import numpy as np
import torch

# This script sits beside the emulator/ package (same .../emultrf/dev/ folder),
# so launching it by path makes its own directory sys.path[0] and
# `import emulator` resolves with no path manipulation. Run it from $ROOTDIR;
# emulator.cocoa reads $ROOTDIR to resolve the data paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.experiment import EmulatorExperiment
from emulator.results import save_sweep_table
from emulator.scheduling import (
  even_assign, run_gpu_pool, GPU_TOKENS,
  estimate_train_vram_fraction, vram_tokens)

# train_args keys a sweep may enter through (the dotted path's first
# segment). Guards against a typo'd path silently no-opping:
# exp.train reads train_args by .get, so an unknown top-level key
# would be ignored, and the sweep would train the same config N
# times. model.* keys are further validated by build_specs
# (MODEL_BLOCK_KEYS) at train time, loudly.
SWEEPABLE_TOP_KEYS = ("nepochs", "bs", "loss_mode", "trunk_epochs",
                      "clip", "rewind", "trunk", "head", "model",
                      "optimizer", "lr", "scheduler", "trim",
                      "focus")

# dotted paths that sweep the activation FAMILY: these are resolved
# by from_config into exp.activation (build_specs deliberately does
# NOT re-read the YAML block, so a train_args copy would be
# ignored); the job sets exp.activation per value instead.
ACTIVATION_PATHS = ("model.activation", "model.activation.type")


def set_by_path(train_args, path, value):
  """
  A deep copy of train_args with one dotted-path leaf replaced.

  Walks the nested mapping along `path` ("lr.lr_base" -> ["lr",
  "lr_base"]), creating intermediate mappings that do not exist yet
  (so `head.lr_base` sweeps even when the YAML has no head: block;
  run_emulator's trunk_epochs guard still applies), and sets the
  final key. The input is never mutated -- each sweep point gets its
  own copy.

  Arguments:
    train_args = the resolved train_args mapping to copy.
    path       = dotted path of the leaf to set.
    value      = the value this sweep point tries.

  Returns:
    the modified deep copy.
  """
  out  = copy.deepcopy(train_args)
  node = out
  keys = path.split(".")
  for k in keys[:-1]:
    nxt = node.get(k)
    if not isinstance(nxt, dict):
      nxt = {}
      node[k] = nxt
    node = nxt
  node[keys[-1]] = value
  return out


def read_sweep_block(cfg):
  """
  Validate and unpack the YAML `sweep` block.

  Arguments:
    cfg = the resolved config mapping (data + train_args + sweep).

  Returns:
    (param, values, act_mode): the dotted path, the value list, and
    whether this is the activation-family special case.
  """
  if "sweep" not in cfg:
    raise KeyError(
      "the YAML needs a `sweep` block:\n"
      "  sweep:\n"
      "    parameter: lr.lr_base\n"
      "    values:\n"
      "      - 0.001\n"
      "      - 0.0025")
  blk    = cfg["sweep"]
  param  = str(blk.get("parameter", "")).strip()
  values = blk.get("values")
  if not param or not isinstance(values, list) or len(values) == 0:
    raise ValueError(
      "sweep block needs `parameter` (a dotted train_args path) "
      "and a non-empty `values` list")
  if param in ("model.name", "model.ia"):
    raise ValueError(
      f"cannot sweep {param}: it changes the model class -- run "
      "one sweep per architecture (or the activation bake-off "
      "driver) and overlay the saved tables")
  act_mode = param in ACTIVATION_PATHS
  if not act_mode and param.split(".")[0] not in SWEEPABLE_TOP_KEYS:
    raise ValueError(
      f"sweep parameter {param!r} does not enter train_args "
      f"(first segment must be one of: "
      f"{' / '.join(SWEEPABLE_TOP_KEYS)})")
  return param, values, act_mode


def _hyper_setup(gpu_id, extra):
  """
  Per-worker setup for run_gpu_pool: experiment + data, staged once.

  Unlike the N_train sweep, the data and geometry are FIXED across
  the sweep points, so each worker stages train + val and builds the
  geometry a single time; jobs then only train.

  Arguments:
    gpu_id = CUDA device index this lane owns.
    extra  = the parent's payload dict: cfg, rescale, activation,
             param, act_mode, threshold.

  Returns:
    the fully staged EmulatorExperiment.
  """
  device = torch.device(f"cuda:{gpu_id}")
  exp = EmulatorExperiment.from_config(extra["cfg"],
                                       device=device,
                                       rescale=extra["rescale"],
                                       activation=extra["activation"],
                                       quiet=True)
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()
  return exp


def _hyper_job(gpu_id, exp, payload, extra):
  """
  One sweep point for run_gpu_pool: set the knob, train, score.

  Ordinary knobs go through set_by_path into a per-point train_args
  copy; the activation special case sets exp.activation instead (the
  train_args stay untouched -- build_specs reads the family off the
  experiment). Total by design: a failed point returns frac = nan.

  Arguments:
    gpu_id  = CUDA device index (bookkeeping; also used by the
              serial path with gpu_id 0).
    exp     = this lane's staged experiment from _hyper_setup.
    payload = (index, value): the point's position in the value list
              (results realign by it -- values need not be unique)
              and the value to try.
    extra   = the parent's payload dict (param, act_mode, threshold).

  Returns:
    (index, frac, gpu_id, seconds).
  """
  idx, value = payload
  t0 = time.time()
  try:
    if extra["act_mode"]:
      exp.activation = str(value)
      ta = None                     # train_args unchanged
    else:
      ta = set_by_path(exp.train_args, extra["param"], value)
    exp.train(train_args=ta, silent=True)
    f = float(exp.frac_above(threshold=extra["threshold"]))
  except Exception as err:          # keep the sweep alive
    f = float("nan")
    print(f"[gpu {gpu_id}] {extra['param']} = {value!r} "
          f"failed: {err}")
  # free this point's model before the next (possibly co-located)
  # point sizes its loaders; the staged data + geometry stay.
  exp.model = None
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  return (idx, f, gpu_id, time.time() - t0)


def main():
  parser = argparse.ArgumentParser(
    prog="sweep_hyperparam_emulator_cosmic_shear")
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML + sweep outputs under --fileroot). Same schema as the
  # training driver, plus the sweep: block.
  add_cocoa_path_args(parser)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across "
                           "the sweep: 'none' (default), 'rescaled' "
                           "(v1), or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep (leave unset when sweeping "
                           "model.activation): 'H', 'power', "
                           "'multigate', or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default=None)
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread the sweep "
                           "across (default: all visible CUDA "
                           "devices). 1, or no CUDA, is serial.",
                      type=int,
                      default=None)
  parser.add_argument("--gpu-pack",
                      dest="gpu_pack",
                      help="co-locate small trainings on one GPU "
                           "(<=20%% of the card -> up to 4 share, "
                           "<=40%% -> up to 2, else exclusive). Off "
                           "by default; meant for large cards "
                           "(H200), not a 12 GB RTX 3060",
                      action="store_true")
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2, the emulator goal)",
                      type=float,
                      default=0.2)
  parser.add_argument("--out",
                      dest="out",
                      help="output base path -> <out>.txt + "
                           "<out>.pdf (default hyperparam_sweep)",
                      type=str,
                      default="hyperparam_sweep")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still "
                           "written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # headless figure output, set before pyplot loads and before any
  # worker spawns (children inherit it).
  os.environ.setdefault("MPLBACKEND", "Agg")

  cfg, fileroot, _ = resolve_cocoa_config(args)
  param, values, act_mode = read_sweep_block(cfg)
  if act_mode and args.activation is not None:
    raise ValueError(
      "--activation conflicts with sweeping model.activation: "
      "drop the flag and let the sweep set the family per value")

  # the parent's experiment: banner + device + pool size; the serial
  # path reuses it.
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): "
      "this sweep trains one model per value and is not meant for "
      "CPU")
  log = exp.log
  # print_design (experiment.py): the startup banner; the sweep then
  # overrides ONE leaf of what it shows, per point.
  exp.print_design()
  log(f"sweep: {param}  ->  {values}"
      + ("  (activation family)" if act_mode else ""))

  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus,
                                                     n_cuda)
  n_workers = min(n_request, len(values))

  # payloads carry the position so results realign afterward.
  payloads = []
  for i, v in enumerate(values):
    payloads.append((i, v))

  # --gpu-pack engages the pool even on a SINGLE CUDA card (its whole
  # point there: up to 4 small trainings co-located on one big GPU,
  # e.g. a lone H200 allocation).
  use_pool = (n_workers > 1
              or (args.gpu_pack and n_cuda >= 1 and len(values) > 1))

  if not use_pool:
    # serial on this one device (single GPU or the MPS dev machine),
    # reusing the parent experiment; same setup-once, train-per-value
    # shape as the workers.
    log("serial (1 worker)")
    log("loading sources:")
    exp.stage_train()
    exp.stage_val()
    exp.build_geometry()
    extra = {"param":     param,
             "act_mode":  act_mode,
             "threshold": args.threshold}
    fracs = [None] * len(values)
    for payload in payloads:
      idx, f, gpu, secs = _hyper_job(0, exp, payload, extra)
      fracs[idx] = f
      log(f"  {param} = {values[idx]!r:>12}  "
          f"f(>{args.threshold:g}) {f:.4f}  ({secs:.0f}s)")
  else:
    # equal-cost jobs -> round-robin; ram_frac 0 keeps every worker
    # streaming from the one shared dump memmap.
    n_pool = max(1, n_workers)
    log(f"parallel sweep across {n_pool} GPU(s)"
        + (" (gpu-pack)" if args.gpu_pack else "") + ":")
    worker_cfg = dict(cfg)
    worker_cfg["data"] = dict(cfg["data"])
    worker_cfg["data"]["ram_frac"] = 0.0
    buckets = even_assign(payloads, n_pool)
    for k, b in enumerate(buckets):
      vals = []
      for i, v in b:
        vals.append(v)
      log(f"  gpu {k}: {len(b)} points  ->  {vals}")

    # --gpu-pack: every point stages the same N_train, so one token
    # count covers all jobs. The row estimate is the pre-cut
    # rows // divisor (an upper bound on the staged subset).
    lanes = 1
    job_tokens = None
    if args.gpu_pack:
      dv = np.load(cfg["data"]["train_dv"], mmap_mode="r")
      n_est = dv.shape[0] // int(cfg["data"].get("train_divisor", 1))
      total = torch.cuda.get_device_properties(0).total_memory
      frac  = estimate_train_vram_fraction(n_rows=n_est,
                                           dv_width=dv.shape[1],
                                           total_bytes=total)
      tokens = vram_tokens(frac)
      def job_tokens(payload):
        return tokens
      lanes = GPU_TOKENS
      log(f"  gpu-pack on: est. {frac:.2f} of a GPU per point "
          f"-> {tokens} token(s)/4")

    extra = {"cfg":        worker_cfg,
             "rescale":    args.rescale,
             "activation": args.activation,
             "param":      param,
             "act_mode":   act_mode,
             "threshold":  args.threshold}

    def on_result(r):
      idx, f, gpu, secs = r
      log(f"  {param} = {values[idx]!r:>12}  "
          f"f(>{args.threshold:g}) {f:.4f}  (gpu {gpu}, {secs:.0f}s)")

    results = run_gpu_pool(setup_fn=_hyper_setup,
                           job_fn=_hyper_job,
                           buckets=buckets,
                           extra=extra,
                           lanes_per_gpu=lanes,
                           job_tokens=job_tokens,
                           on_result=on_result)
    fracs = [None] * len(values)
    for idx, f, gpu, secs in results:
      fracs[idx] = f

  # table + figure, like the N_train sweep (overlay several <out>.txt
  # yourself to compare configs).
  out_txt = cocoa_output(fileroot, args.out + ".txt")
  out_pdf = cocoa_output(fileroot, args.out + ".pdf")
  save_sweep_table(
    path=out_txt,
    param=param,
    values=values,
    fracs=fracs,
    meta={"model": exp.model_name,
          "rescale": args.rescale,
          "activation": ("swept" if act_mode else args.activation),
          "threshold": args.threshold,
          "train_divisor": cfg["data"].get("train_divisor"),
          "n_gpus": n_workers})
  log(f"saved sweep table -> {out_txt}")

  from emulator.plotting import plot_sweep_curve
  plot_sweep_curve(param=param,
                   values=values,
                   fracs=fracs,
                   threshold=args.threshold,
                   savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
