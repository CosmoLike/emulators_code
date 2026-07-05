#!/usr/bin/env python3
"""N_train learning curve: f(delta-chi2 > thr) vs N_train for one config."""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# Sweeps the training-set size for one fixed config, recording validation
# f(delta-chi2 > threshold) at each size -- the learning curve telling whether
# the floor is data-limited (still falling at the largest N) or capacity /
# architecture-limited (a flat tail).
#
#     python external_modules/code/emulators/emultrf/dev/sweep_ntrain_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --n-min 2000 --n-points 6 --out ntrain_resmlp
#
#- Reuses the training driver's YAML (and its model/rescale/activation choices).
#  To compare architectures or chi2 modes, run once per config (vary
#  train_args.model.name or --rescale / --activation, with a different --out),
#  then overlay the saved <out>.txt curves.
#
#- Per N_train (geometric grid [--n-min .. --n-max], --n-max defaults to the
#  full physically-cut pool), stages a nested subset of that size, rebuilds the
#  geometry from it, trains a fresh model silently, and scores
#  f(delta-chi2 > --threshold) on the fixed validation set.
#
#- Multiple GPUs (one node): grid points are independent trainings, so they run
#  in parallel, one process per GPU, split by the Longest-Processing-Time rule
#  (largest N_train first to the least-loaded GPU) so each GPU gets about the
#  same total N_train and they finish together. On an 8-GPU node, add
#  `--n-points 12 --n-gpus 8` to the call above.
#
#  One GPU (or none, e.g. the Apple-MPS dev machine) falls back to a serial
#  loop, so the same script runs everywhere.
#
#- `--gpu-pack` (optional, off by default): co-locate several trainings on one
#  GPU when they are small. Each point's VRAM need is estimated conservatively
#  (2 * N * dv_width float32 + a 2 GiB fixed overhead, see
#  emulator/scheduling.py); a point at or below 20% of the card shares it up
#  to 4 ways, one at or below 40% up to 2 ways, anything bigger runs
#  exclusive. It engages even with a single visible GPU (a lone H200
#  allocation: up to 4 small points co-located on the one card). Worth it on
#  a large card (H200) where a small-N training is
#  launch-bound and leaves most of the GPU idle; leave it off on a small card
#  (an RTX 3060: the fixed overhead alone is ~17% of 12 GB, and co-located
#  contexts would crowd out the data). If a point outgrows its estimate the
#  loaders degrade to streaming against the real free VRAM rather than crash.
#  Keep it off for timing measurements -- co-located points contend and their
#  s/epoch is not comparable to exclusive runs.
#
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML and curve outputs (e.g.
#  emulators/training_scripts). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot (data + train_args),
#  training-driver schema; train_args.model.name picks the architecture
#  (resmlp | rescnn | restrf) and model.ia the factored IA design layered
#  on it (omit | nla | tatt), with the nested mlp / activation / cnn /
#  trf sub-blocks, the optional two-phase trunk_epochs + trunk / head
#  override blocks, and the clip / rewind guards -- see the training
#  driver's header for every key. The `data` block lists bare filenames,
#  resolved under --root/chains.
#- `--rescale` / `--activation`: as in the training driver, fixed across the
#  sweep (analytic-R mode and ResBlock activation).
#- `--n-gpus` (default: all visible CUDA devices): GPUs to spread across. 1, or
#  no CUDA, is serial.
#- `--n-min` (default 2000), `--n-max` (default = pool), `--n-points` (default
#  5): geometric N_train grid (clamped to the pool, deduplicated).
#- `--threshold` (default 0.2): delta-chi2 cutoff the fraction counts.
#- `--out` (default ntrain_sweep): writes <out>.txt (curve + config,
#  np.loadtxt-loadable) and <out>.pdf (single-curve figure), under --fileroot.
#- `--quiet`: suppress stdout (txt and pdf still written).
#
#- Trains one full model per grid point (--n-points trainings, divided across
#  the GPUs) -- run it on the workstation, where cosmolike lives.
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
from emulator.scheduling import (
  lpt_assign, run_gpu_pool, GPU_TOKENS,
  estimate_train_vram_fraction, vram_tokens)


def _sweep_setup(gpu_id, extra):
  """
  Per-worker setup for run_gpu_pool: one experiment on this GPU.

  Runs once in each spawned lane. Builds this worker's own
  EmulatorExperiment on its GPU (each process has its own cosmolike
  global state and cached experiment, so workers never interfere) and
  stages the validation set, which is fixed across the sweep.

  Arguments:
    gpu_id = CUDA device index this lane owns (already claimed by
             run_gpu_pool via torch.cuda.set_device).
    extra  = the parent's payload dict: cfg (host ram_frac already
             set for streaming), rescale, activation, threshold.

  Returns:
    the staged EmulatorExperiment (the pool hands it to _sweep_job).
  """
  device = torch.device(f"cuda:{gpu_id}")
  exp = EmulatorExperiment.from_config(extra["cfg"],
                                       device=device,
                                       rescale=extra["rescale"],
                                       activation=extra["activation"],
                                       quiet=True)
  exp.stage_val()
  return exp


def _sweep_job(gpu_id, exp, N, extra):
  """
  One sweep point for run_gpu_pool: train at N_train = N, score it.

  Stages the nested subset of size N, rebuilds the geometry from its
  means, trains a fresh model quietly, and scores it on the fixed val
  set. Total by design (any failure returns frac = nan instead of
  killing the lane), and it returns this point's GPU tensors before
  finishing so the next point (possibly a co-located lane's) sizes
  its loaders against the true free memory -- the caching allocator
  would otherwise keep the VRAM reserved and mem_get_info would
  under-report.

  Arguments:
    gpu_id = CUDA device index (result bookkeeping only).
    exp    = this lane's experiment from _sweep_setup.
    N      = the N_train value to train at.
    extra  = the parent's payload dict (reads threshold).

  Returns:
    (N, frac, gpu_id, seconds) -- one result row for the parent.
  """
  t0 = time.time()
  try:
    exp.stage_train(n_train=int(N))
    exp.build_geometry()
    exp.train(silent=True)
    f = float(exp.frac_above(threshold=extra["threshold"]))
  except Exception as err:                  # keep the sweep alive
    f = float("nan")
    print(f"[gpu {gpu_id}] N_train {int(N)} failed: {err}")
  exp.model     = None
  exp.train_set = None
  exp.geom      = None
  exp.pgeom     = None
  exp.chi2fn    = None
  torch.cuda.empty_cache()
  return (int(N), f, gpu_id, time.time() - t0)


def _run_parallel(cfg, sizes, n_workers, args, log):
  """
  Run the sweep across n_workers GPUs via run_gpu_pool, LPT-balanced.

  Splits `sizes` with lpt_assign so each GPU gets about the same total
  N_train, then hands the buckets to run_gpu_pool (scheduling.py): one
  spawned process per (GPU, lane), each building its own experiment
  once (_sweep_setup) and training its points (_sweep_job); the parent
  drains and logs one result per point. Host RAM stays flat because
  ram_frac 0 tells stage_source to keep the shared dump memmap (a
  per-worker private copy would multiply host RAM by the worker
  count).

  Under --gpu-pack the pool runs up to GPU_TOKENS lanes per GPU and
  gates each point on its VRAM tokens (vram_tokens of the
  estimate_train_vram_fraction of its N), so several small-N
  trainings share one large GPU while a big-N point runs exclusive.

  Arguments:
    cfg       = the parsed YAML config (data + train_args).
    sizes     = the N_train grid (a sequence of ints).
    n_workers = number of GPUs to spread across.
    args      = the parsed CLI namespace (rescale / activation /
                threshold / gpu_pack).
    log       = print function (no-op under --quiet).

  Returns:
    fracs = list of f(delta-chi2 > threshold), aligned with `sizes`.
  """
  # ram_frac 0: stream from the one shared dump memmap (OS page
  # cache); copy the data block first to leave the original cfg
  # untouched.
  worker_cfg = dict(cfg)
  worker_cfg["data"] = dict(cfg["data"])
  worker_cfg["data"]["ram_frac"] = 0.0

  # LPT split: largest N first, each to the least-loaded GPU. Bucket
  # order is big-first, which is also the right queue order under
  # packing (the exclusive points start first, the small ones fill
  # the remaining lanes).
  buckets = lpt_assign(sizes, n_workers)
  for k, b in enumerate(buckets):
    log(f"  gpu {k}: {len(b)} points, total N {sum(b)}  ->  {sorted(b)}")

  # --gpu-pack: estimate each point's VRAM share and convert it to
  # capacity tokens; the pool then co-locates points whose tokens fit
  # (<=20% of the card -> 4 per GPU, <=40% -> 2, else exclusive).
  # The estimate reads the dv dump's width (an upper bound on the
  # staged target width) and GPU 0's total memory (a homogeneous-GPU
  # assumption -- true on amypond's pair and on an H200 node).
  lanes = 1
  job_tokens = None
  if args.gpu_pack:
    dv_width = np.load(cfg["data"]["train_dv"], mmap_mode="r").shape[1]
    total    = torch.cuda.get_device_properties(0).total_memory
    def job_tokens(N):
      return vram_tokens(estimate_train_vram_fraction(
        n_rows=int(N), dv_width=dv_width, total_bytes=total))
    lanes = GPU_TOKENS
    toks = []
    for N in sizes:
      toks.append(f"{int(N)}:{job_tokens(N)}")
    log(f"  gpu-pack on: tokens/4 per point  ->  {', '.join(toks)}")

  extra = {"cfg":        worker_cfg,
           "rescale":    args.rescale,
           "activation": args.activation,
           "threshold":  args.threshold}

  # the parent logs each point as it lands (workers run quiet, so
  # multiple streams do not interleave).
  def on_result(r):
    N, f, gpu, secs = r
    log(f"  N_train {N:8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"(gpu {gpu}, {secs:.0f}s)")

  results = run_gpu_pool(setup_fn=_sweep_setup,
                         job_fn=_sweep_job,
                         buckets=buckets,
                         extra=extra,
                         lanes_per_gpu=lanes,
                         job_tokens=job_tokens,
                         on_result=on_result)

  # results arrived out of order; re-align to `sizes`.
  by_N = {}
  for N, f, gpu, secs in results:
    by_N[N] = f
  fracs = []
  for N in sizes:
    fracs.append(by_N[int(N)])
  return fracs


def main():
  parser = argparse.ArgumentParser(
    prog="sweep_ntrain_emulator_cosmic_shear")
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML + curve outputs under --fileroot). Same schema as the
  # training driver.
  add_cocoa_path_args(parser)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across the "
                           "sweep: 'none' (default), 'rescaled' (v1), "
                           "or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "sweep: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation; default: "
                           "the YAML's choice, else 'H'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default=None)
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread the sweep across "
                           "(default: all visible CUDA devices). 1, or no "
                           "CUDA, takes the serial path.",
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
  parser.add_argument("--gpu-pack",
                      dest="gpu_pack",
                      help="co-locate small trainings on one GPU "
                           "(<=20%% of the card -> up to 4 share, "
                           "<=40%% -> up to 2, else exclusive; "
                           "conservative VRAM estimate). Off by "
                           "default; meant for large cards (H200), "
                           "not a 12 GB RTX 3060",
                      action="store_true")
  parser.add_argument("--threshold",
                      dest="threshold",
                      help="delta-chi2 cutoff the fraction counts "
                           "(default 0.2, the emulator goal)",
                      type=float,
                      default=0.2)
  parser.add_argument("--out",
                      dest="out",
                      help="output base path -> <out>.txt + <out>.pdf "
                           "(default ntrain_sweep)",
                      type=str,
                      default="ntrain_sweep")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (txt / pdf still written)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # headless figure output: pick a non-interactive matplotlib backend before
  # emulator.plotting imports pyplot (lazily below) and before any worker spawns,
  # so the children inherit it.
  os.environ.setdefault("MPLBACKEND", "Agg")

  # resolve the cocoa layout and read the config once (data paths made
  # absolute under $ROOTDIR/<root>). The parent uses cfg for the grid and
  # hands a copy (host-RAM budget set for streaming) to each GPU process;
  # absolute paths mean every spawned worker reads the same files.
  cfg, fileroot, _ = resolve_cocoa_config(args)

  # build the experiment on the real compute device (CUDA, or Apple MPS on the
  # dev machine); pool size and model name are read off it, and the serial path
  # reuses it. Under spawn the parent may hold a GPU context and still launch
  # workers safely. Trains a model per grid point, so it is a GPU tool: refuse a
  # pure-CPU box.
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  if exp.device.type == "cpu":
    raise RuntimeError(
      "no GPU found (need CUDA, or Apple MPS on the dev machine): this "
      "sweep trains one model per grid point and is not meant for CPU")
  log = exp.log
  model_name = exp.model_cls.__name__

  # N_train grid: geometric from n_min to the pool (or --n-max), clamped to the
  # physically-cut pool so every size is loadable; unique() drops int-cast
  # collisions at the low end.
  pool  = exp.pool_size()
  n_max = pool if args.n_max is None else min(args.n_max, pool)
  if args.n_min >= n_max:
    raise ValueError(
      f"--n-min {args.n_min} must be below n_max {n_max} (pool {pool})")
  sizes = np.unique(
    np.geomspace(args.n_min, n_max, args.n_points).astype(int))

  # how many GPUs to use: capped by what is visible, by --n-gpus, and by the
  # point count (no idle workers).
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus, n_cuda)
  n_workers = min(n_request, len(sizes))

  # print_design (experiment.py): the startup banner -- the resolved
  # model block, run knobs, guards, every train_args sub-block, and the
  # physical cuts. A stale YAML here would waste a whole sweep, not one
  # training. Shared with the train / tune drivers.
  exp.print_design()
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # 1 worker (single GPU, or the MPS dev machine) -> serial on this one device,
  # reusing the experiment; otherwise one process per GPU, LPT-balanced.
  # Exception: --gpu-pack engages the pool even on a SINGLE CUDA card (its
  # whole point there: up to 4 small trainings co-located on one big GPU,
  # e.g. a lone H200 allocation).
  use_pool = (n_workers > 1
              or (args.gpu_pack and n_cuda >= 1 and len(sizes) > 1))
  if not use_pool:
    # the banner already named the device; this line says only the mode.
    log("serial (1 worker)")
    log("loading validation source:")
    exp.stage_val()
    fracs = []
    for N in sizes:
      t0 = time.time()
      exp.stage_train(n_train=int(N))
      exp.build_geometry()
      exp.train(silent=True)
      f = exp.frac_above(threshold=args.threshold)
      fracs.append(f)
      log(f"  N_train {int(N):8d}  f(>{args.threshold:g}) {f:.4f}  "
          f"({time.time() - t0:.0f}s)")
  else:
    n_pool = max(1, n_workers)
    log(f"parallel sweep across {n_pool} GPU(s) (LPT-balanced"
        + (", gpu-pack" if args.gpu_pack else "") + "):")
    fracs = _run_parallel(cfg=cfg,
                          sizes=sizes,
                          n_workers=n_pool,
                          args=args,
                          log=log)

  # save the curve + its config as a plain-text table, so several runs (one per
  # architecture / chi2 mode) overlay later (np.loadtxt-loadable; # headers
  # skipped). Outputs land under the emulator's fileroot.
  out_txt = cocoa_output(fileroot, args.out + ".txt")
  out_pdf = cocoa_output(fileroot, args.out + ".pdf")
  save_learning_curves(
    path=out_txt,
    sizes=sizes,
    curves={"frac": fracs},
    meta={"model": model_name,
          "rescale": args.rescale,
          "activation": args.activation,
          "threshold": args.threshold,
          "pool": pool,
          "n_gpus": n_workers})
  log(f"saved curve data -> {out_txt}")

  # one-curve figure (overlay several <out>.txt yourself to compare).
  from emulator.plotting import plot_learning_curves
  plot_learning_curves(
    curves={f"{model_name} ({args.rescale})": (sizes, fracs)},
    threshold=args.threshold,
    savepath=out_pdf)
  log(f"saved figure -> {out_pdf}")


if __name__ == "__main__":
  main()
