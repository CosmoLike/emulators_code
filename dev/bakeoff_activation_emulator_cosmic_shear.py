#!/usr/bin/env python3
"""Activation bake-off: f(delta-chi2 > thr) vs N_train, one curve per act."""

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Bakes off the ResBlock activation head-to-head: per activation, measures
# validation f(delta-chi2 > threshold) over a grid of training-set sizes,
# overlaying the curves. A double loop (N_train x activation) whose verdict is
# the curve shape: a real inductive-bias win keeps descending (lower sample
# complexity) where others flatten, not a single-N offset.
#
#     python external_modules/code/emulators/emultrf/dev/bakeoff_activation_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --n-min 2000 --n-points 6 --out bakeoff_act
#
#- Per N_train (geometric grid [--n-min .. --n-max], --n-max defaults to the
#  full physically-cut pool), stages a nested subset and builds the geometry
#  once, then silently trains a fresh model per activation and scores
#  f(delta-chi2 > --threshold) on the fixed validation set. Data and geometry
#  are activation-independent: shared at that N; only the model rebuilds.
#
#- Multiple GPUs (one node): activations split across GPUs, one process per GPU,
#  each running the whole grid for its share. Activations cost about the same
#  and every GPU runs the full grid: equal work, no cost-aware balancing (unlike
#  the N_train sweep). At most len(activations) GPUs used; default 4
#  activations, 8 GPUs, 4 idle. With --n-gpus 4:
#
#     python external_modules/code/emulators/emultrf/dev/bakeoff_activation_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --n-points 8 --n-gpus 4 --out bakeoff_act
#
#  One GPU (or none, e.g. the Apple-MPS dev machine) runs the serial double
#  loop, so one script runs everywhere.
#
#- Reuses the training driver's YAML (data, model = ResMLP / ResCNN via
#  train_args.model.name, rest of train_args). The YAML activation and usual
#  --activation flag are ignored; this driver sweeps it.
#
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML and curve outputs (e.g.
#  emulators/training_scripts). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot (data + train_args),
#  training-driver schema. The `data` block lists bare filenames, resolved under
#  --root/chains.
#- `--activations` (default H,power,multigate,gated_power): comma-separated
#  subset to bake off.
#- `--rescale`: analytic-R mode, fixed across the bake-off (as in training).
#- `--n-gpus` (default: all visible CUDA devices, capped at len(activations)):
#  GPUs to split activations across. 1, or no CUDA, is serial.
#- `--n-min` (default 2000), `--n-max` (default = pool), `--n-points` (default
#  5): geometric N_train grid (clamped to the pool, deduplicated).
#- `--threshold` (default 0.2): delta-chi2 cutoff the fraction counts.
#- `--out` (default bakeoff_act): writes <out>.txt (every curve + config,
#  np.loadtxt-loadable, one column per activation) and <out>.pdf (figure), under
#  --fileroot.
#- `--quiet`: suppress stdout (txt and pdf still written).
#
#- One full model per (N_train, activation): len(grid) x len(activations)
#  trainings long -- run it on the workstation.
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

import argparse
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
from emulator.results import save_learning_curves


# activations this driver can build (make_activation names).
ACTS = ["H", "power", "multigate", "gated_power"]


def _bakeoff_worker(gpu_id, my_acts, sizes, cfg, rescale,
                    threshold, result_q):
  """
  One GPU's share of the bake-off, in its own process.

  Pins itself to GPU `gpu_id`, builds its own EmulatorExperiment there, and runs
  the full grid for `my_acts`: per N, stages the subset, builds the geometry
  once, then trains each activation on it, putting an
  (N, activation, frac, gpu_id, seconds) tuple onto result_q per training. Each
  process has its own cosmolike global state and experiment, so workers never
  interfere. A failed training emits frac = nan, so the parent gets one result
  per (N, activation) and never deadlocks.

  Arguments:
    gpu_id    = CUDA device index this worker owns.
    my_acts   = activations assigned to this GPU (its share).
    sizes     = full N_train grid (every worker runs it all).
    cfg       = YAML config (data + train_args), host ram_frac already split
                across workers by the parent.
    rescale   = analytic-R mode, forwarded to the experiment.
    threshold = delta-chi2 cutoff for frac_above.
    result_q  = process-safe queue the parent drains.
  """
  # claim this GPU so every default-device op (and run_emulator's mem_get_info,
  # which reads the current device when sizing resident data) hits this card,
  # not card 0.
  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")

  # this worker's own experiment on its GPU (quiet: the parent logs). Activation
  # is set per training below; my_acts[0] is just the first.
  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=my_acts[0],
                                       quiet=True)
  # validation set is fixed; stage it once per worker.
  exp.stage_val()

  for N in sizes:
    # stage subset and build geometry once; both activation-independent.
    exp.stage_train(n_train=int(N))
    exp.build_geometry()

    for act in my_acts:
      t0 = time.time()
      try:
        # build_specs reads exp.activation, so set it before training.
        exp.activation = act
        exp.train(silent=True)
        f = float(exp.frac_above(threshold=threshold))
      except Exception as err:                 # keep the bake-off alive
        f = float("nan")
        print(f"[gpu {gpu_id}] N_train {int(N)} act {act} failed: {err}")
      result_q.put((int(N), act, f, gpu_id, time.time() - t0))

      # free this activation's model and loaders, keep the shared geometry.
      # Returning the reserved VRAM lets the next training size its loaders to
      # true free memory.
      exp.model = None
      torch.cuda.empty_cache()

    # done with this N: free geometry and staged data first.
    exp.train_set = None
    exp.geom      = None
    exp.pgeom     = None
    exp.chi2fn    = None
    torch.cuda.empty_cache()


def _run_parallel_bakeoff(cfg, sizes, activations, n_workers, args, log):
  """
  Run the bake-off across n_workers GPUs, one process each.

  Splits activations evenly across the GPUs (plain round-robin: each costs about
  the same and every GPU runs the whole grid, so equal work without cost-aware
  balancing). Spawns one process per GPU (spawn start method: a forked child
  cannot reuse the parent's CUDA context) and collects per-(N, activation)
  fractions. The host-RAM staging budget (ram_frac) is split by n_workers so
  they do not collectively overflow host memory.

  Arguments:
    cfg         = YAML config (data + train_args).
    sizes       = N_train grid (ints).
    activations = activations to bake off.
    n_workers   = GPU processes to launch.
    args        = CLI namespace (rescale / threshold read).
    log         = print function (no-op under --quiet).

  Returns:
    curves = dict activation -> {N_train: frac}.
  """
  import torch.multiprocessing as mp

  # Workers make no private in-RAM copy of their subset: ram_frac 0 tells
  # stage_source to keep the shared dump memmap and stream the subset from it.
  # One memmap, shared across processes via the OS page cache, keeps per-worker
  # host RAM flat regardless of GPU count; a private copy would multiply it by
  # the GPU count for almost no gain (the subset is GPU-resident anyway). Copy
  # the data block first to leave the original cfg intact.
  worker_cfg = dict(cfg)
  worker_cfg["data"] = dict(cfg["data"])
  worker_cfg["data"]["ram_frac"] = 0.0

  # split activations round-robin: GPU g gets activation g, g + n_workers,
  # g + 2*n_workers, ... (a strided walk). Equal counts, so no cost-aware (LPT)
  # split.
  buckets = []
  for g in range(n_workers):
    bucket = []
    for i in range(g, len(activations), n_workers):
      bucket.append(activations[i])
    buckets.append(bucket)
  for k, b in enumerate(buckets):
    log(f"  gpu {k}: activations {b}")

  # spawn (not fork): CUDA state cannot be inherited across a fork. Each worker
  # gets its activation bucket plus the full grid.
  ctx = mp.get_context("spawn")
  result_q = ctx.Queue()
  sizes_i = []
  for N in sizes:
    sizes_i.append(int(N))
  procs = []
  for k in range(n_workers):
    p = ctx.Process(target=_bakeoff_worker,
                    args=(k,
                          buckets[k],
                          sizes_i,
                          worker_cfg,
                          args.rescale,
                          args.threshold,
                          result_q))
    p.start()
    procs.append(p)

  # drain one result per (N, activation) as workers finish; the parent does the
  # logging (workers are quiet).
  curves = {}
  for act in activations:
    curves[act] = {}
  total = len(sizes) * len(activations)
  for _ in range(total):
    N, act, f, gpu, secs = result_q.get()
    curves[act][int(N)] = f
    log(f"  N_train {N:8d}  {act:12s}  f(>{args.threshold:g}) {f:.4f}  "
        f"(gpu {gpu}, {secs:.0f}s)")

  for p in procs:
    p.join()

  return curves


def main():
  parser = argparse.ArgumentParser(
    prog="bakeoff_activation_emulator_cosmic_shear")
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML + curve outputs under --fileroot). Same schema as the
  # training driver.
  add_cocoa_path_args(parser)
  parser.add_argument("--activations",
                      dest="activations",
                      help="comma-separated activations to bake off, a "
                           "subset of H,power,multigate,gated_power "
                           "(default: all four)",
                      type=str,
                      default=",".join(ACTS))
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across the "
                           "bake-off: 'none' (default), 'rescaled' (v1), "
                           "or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to split the activations across "
                           "(default: all visible CUDA devices, capped at "
                           "the number of activations). 1, or no CUDA, is "
                           "serial.",
                      type=int,
                      default=None)
  parser.add_argument("--n-min",
                      dest="n_min",
                      help="smallest N_train in the grid (default 2000)",
                      type=int,
                      default=2000)
  parser.add_argument("--n-max",
                      dest="n_max",
                      help="largest N_train in the grid (default and "
                           "ceiling: the physically-cut training pool)",
                      type=int,
                      default=None)
  parser.add_argument("--n-points",
                      dest="n_points",
                      help="number of geometric grid points (default 5)",
                      type=int,
                      default=5)
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2, the emulator goal)",
                      type=float,
                      default=0.2)
  parser.add_argument("--out",
                      dest="out",
                      help="output base path -> <out>.txt + <out>.pdf "
                           "(default bakeoff_act)",
                      type=str,
                      default="bakeoff_act")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # validate activations up front (fail before training): parse --activations
  # into a clean name list.
  activations = []
  for a in args.activations.split(","):
    a = a.strip()
    if a:
      activations.append(a)
  # collect any unknown names.
  bad = []
  for a in activations:
    if a not in ACTS:
      bad.append(a)
  if bad:
    raise ValueError(
      f"unknown activation(s) {bad}; choose from {ACTS}")

  # headless figure output: pick a non-interactive matplotlib backend before
  # emulator.plotting imports pyplot (lazily, below) and any worker spawns, so
  # children inherit it.
  os.environ.setdefault("MPLBACKEND", "Agg")

  # resolve the cocoa layout and read the config once (data paths made
  # absolute under $ROOTDIR/<root>): parent uses cfg for the grid and hands
  # each GPU process a copy; absolute paths mean every worker reads the same
  # files.
  cfg, fileroot, _ = resolve_cocoa_config(args)

  # build the experiment on the compute device (CUDA, or Apple MPS on the dev
  # machine); pool size and model name read off it, serial path reuses it. A GPU
  # tool: refuse a pure-CPU box. Starting activation is a placeholder, set per
  # run.
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=activations[0],
                                       quiet=args.quiet)
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): this "
      "bake-off trains one model per (N, activation) and is not meant for CPU")
  log = exp.log
  model_name = exp.model_cls.__name__

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped to the
  # pool; unique() drops int-cast collisions.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))

  # GPUs to use: capped by what is visible, by --n-gpus, and by the activation
  # count (split is over activations, so extras sit idle).
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus, n_cuda)
  n_workers = min(n_request, len(activations))

  log(f"model: {model_name}  |  rescale: {args.rescale}  "
      f"|  activations: {activations}")
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # 1 worker (single GPU, or the MPS dev machine) -> serial, reusing the built
  # experiment; else one process per GPU, by activation.
  if n_workers <= 1:
    log(f"device: {exp.device}  |  serial (1 worker)")
    log("loading validation source:")
    exp.stage_val()
    curves = {}
    for act in activations:
      curves[act] = {}
    for N in sizes:
      exp.stage_train(n_train=int(N))
      exp.build_geometry()
      for act in activations:
        t0 = time.time()
        exp.activation = act
        exp.train(silent=True)
        f = exp.frac_above(threshold=args.threshold)
        curves[act][int(N)] = f
        log(f"  N_train {int(N):8d}  {act:12s}  "
            f"f(>{args.threshold:g}) {f:.4f}  ({time.time() - t0:.0f}s)")
  else:
    log(f"parallel bake-off across {n_workers} GPUs (split by activation):")
    curves = _run_parallel_bakeoff(cfg=cfg,
                                   sizes=sizes,
                                   activations=activations,
                                   n_workers=n_workers,
                                   args=args,
                                   log=log)

  # save every curve + config as a plain-text table, one column per activation
  # (np.loadtxt-loadable, # headers skipped). Outputs land under the emulator's
  # fileroot.
  out_txt = cocoa_output(fileroot, args.out + ".txt")
  out_pdf = cocoa_output(fileroot, args.out + ".pdf")
  # turn {activation: {N: frac}} into one column per activation, ordered to
  # match `sizes`, for the table writer.
  columns = {}
  for act in activations:
    col = []
    for n in sizes:
      col.append(curves[act][int(n)])
    columns[act] = col

  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves=columns,
    meta={"model": model_name,
          "rescale": args.rescale,
          "activation": "swept",
          "threshold": args.threshold,
          "pool": pool,
          "n_gpus": n_workers})
  log(f"saved curve data -> {out_txt}")

  # overlaid figure: one curve per activation.
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(curves=curves,
                       threshold=args.threshold,
                       savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
