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
#     python driver/sweep_ntrain_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/nla_cosmic_shear/ \
#       --yaml test.yaml \
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
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML and curve outputs (e.g.
#  emulators/nla_cosmic_shear). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot (data + train_args),
#  training-driver schema; train_args.model.name picks ResMLP / ResCNN. The
#  `data` block lists bare filenames, resolved under --root/chains.
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
import sys
import time

import numpy as np
import torch

# The emulator package sits one directory up from driver/; put the repo root on
# sys.path so `import emulator` resolves from any working directory (see the
# training driver for the why).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.experiment import EmulatorExperiment
from emulator.results import save_learning_curves
from emulator.scheduling import lpt_assign


def _sweep_worker(gpu_id, my_sizes, cfg, rescale, activation,
                  threshold, result_q):
  """
  One GPU's share of the N_train sweep; runs in its own process.

  Pins itself to GPU `gpu_id`, builds its own EmulatorExperiment there, and
  trains every N_train in `my_sizes` (its LPT bucket), putting one
  (N, frac, gpu_id, seconds) tuple onto result_q as each finishes. Each process
  has its own cosmolike global state and cached experiment, so workers never
  interfere. A failure emits frac = nan, so the parent gets one result per N and
  never deadlocks waiting on a missing one.

  Arguments:
    gpu_id     = CUDA device index this worker owns.
    my_sizes   = N_train values assigned to this GPU (its LPT bucket).
    cfg        = parsed YAML config (data + train_args), host ram_frac already
                 divided across workers by the parent.
    rescale    = analytic-R mode, forwarded to the experiment.
    activation = ResBlock activation name, forwarded to the experiment.
    threshold  = delta-chi2 cutoff for frac_above.
    result_q   = multiprocessing queue the parent drains.
  """
  # claim this GPU so every default-device op (and run_emulator's
  # torch.cuda.mem_get_info, which reads the current device when sizing the
  # resident set) targets this card, not card 0.
  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")

  # this worker's own experiment on its GPU (quiet: the parent logs).
  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=activation,
                                       quiet=True)
  # validation set is fixed across the sweep; stage it once per worker.
  exp.stage_val()

  for N in my_sizes:
    t0 = time.time()
    try:
      # nested subset of size N, geometry rebuilt from its means, a fresh model
      # trained quietly, then scored on the fixed val set.
      exp.stage_train(n_train=int(N))
      exp.build_geometry()
      exp.train(silent=True)
      f = float(exp.frac_above(threshold=threshold))
    except Exception as err:                  # keep the sweep alive
      f = float("nan")
      print(f"[gpu {gpu_id}] N_train {int(N)} failed: {err}")
    result_q.put((int(N), f, gpu_id, time.time() - t0))

    # drop this point's GPU tensors and return the reserved VRAM, so the next N
    # sizes its loaders against the true free memory. Otherwise the caching
    # allocator keeps them reserved and run_emulator's mem_get_info under-reports
    # the free VRAM.
    exp.model     = None
    exp.train_set = None
    exp.geom      = None
    exp.pgeom     = None
    exp.chi2fn    = None
    torch.cuda.empty_cache()


def _run_parallel(cfg, sizes, n_workers, args, log):
  """
  Run the sweep across n_workers GPUs, one process each, LPT-balanced.

  Splits `sizes` with lpt_assign so each GPU gets about the same total N_train,
  spawns one process per GPU, and collects the per-point fractions. The host-RAM
  budget (ram_frac) is divided by n_workers so they do not collectively overflow
  host memory.

  Arguments:
    cfg       = the parsed YAML config (data + train_args).
    sizes     = the N_train grid (a sequence of ints).
    n_workers = number of GPU processes to launch.
    args      = the parsed CLI namespace (rescale / activation / threshold).
    log       = print function (no-op under --quiet).

  Returns:
    fracs = list of f(delta-chi2 > threshold), aligned with `sizes`.
  """
  import torch.multiprocessing as mp

  # The workers never copy their subset into private RAM: ram_frac 0 tells
  # stage_source to keep the shared dump memmap and stream the subset from it.
  # The dump is one memmap shared across processes (via the OS page cache), so
  # per-worker host RAM stays flat regardless of GPU count; a private copy would
  # multiply host RAM by the GPU count for almost no gain (the subset sits
  # resident on the GPU anyway). Copy the data block first to leave the original
  # cfg untouched.
  worker_cfg = dict(cfg)
  worker_cfg["data"] = dict(cfg["data"])
  worker_cfg["data"]["ram_frac"] = 0.0

  # LPT split: largest N first, each to the least-loaded GPU.
  buckets = lpt_assign(sizes, n_workers)
  for k, b in enumerate(buckets):
    log(f"  gpu {k}: {len(b)} points, total N {sum(b)}  ->  {sorted(b)}")

  # One child process per GPU via the "spawn" start method: spawn launches a
  # fresh Python interpreter per child, which re-imports this module to find
  # _sweep_worker. The alternative "fork" (Linux default) clones the parent's
  # memory, but a forked child inherits the parent's CUDA state and CUDA refuses
  # to run through an inherited context (it hangs or errors); spawn gives each
  # child a fresh interpreter and CUDA context, so every worker sets up its GPU
  # cleanly. (macOS defaults to spawn; Linux must ask for it.)
  ctx = mp.get_context("spawn")

  # A process-safe queue, from the same spawn context so children can reconstruct
  # it: the one-way channel workers send results on, each calling
  # result_q.put((N, frac, gpu, secs)) and the parent loop below result_q.get().
  # Process-safe means processes can put/get concurrently without corruption
  # (internally a pipe guarded by locks).
  result_q = ctx.Queue()

  # Keep a handle to every child in `procs` to join() them below (and so Python
  # does not garbage-collect them mid-run).
  procs = []
  for k in range(n_workers):
    # Build (not yet start) one child. `target` is the function it runs, `args`
    # the positional tuple. Under spawn both are pickled and shipped to the
    # child, so each must be picklable: _sweep_worker is module-level
    # (importable by name), and buckets[k] / worker_cfg / strings / float / queue
    # are plain picklable data.
    #
    # args fills _sweep_worker's parameters in order:
    #   k               -> gpu_id
    #   buckets[k]      -> my_sizes
    #   worker_cfg      -> cfg
    #   args.rescale    -> rescale
    #   args.activation -> activation
    #   args.threshold  -> threshold
    #   result_q        -> result_q
    p = ctx.Process(target=_sweep_worker,
                    args=(k,
                          buckets[k],
                          worker_cfg,
                          args.rescale,
                          args.activation,
                          args.threshold,
                          result_q))
    # start() launches the OS process running _sweep_worker(*args) and returns
    # immediately (training runs in the background), so the loop moves on to the
    # next GPU; the parent gathers results afterward.
    p.start()
    procs.append(p)

  # drain one result per point as the workers finish; the parent does all the
  # logging (workers run quiet, so 8 streams do not interleave).
  results = {}
  for _ in range(len(sizes)):
    N, f, gpu, secs = result_q.get()
    results[N] = f
    log(f"  N_train {N:8d}  f(>{args.threshold:g}) {f:.4f}  "
        f"(gpu {gpu}, {secs:.0f}s)")

  for p in procs:
    p.join()

  # results arrived out of order; re-align to `sizes`.
  fracs = []
  for N in sizes:
    fracs.append(results[int(N)])
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
                           "sweep: 'H' (default), 'power', 'multigate', "
                           "or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
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
  cfg, fileroot = resolve_cocoa_config(args)

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

  log(f"model: {model_name}  |  rescale: {args.rescale}  "
      f"|  activation: {args.activation}")
  log(f"pool {pool}  |  N_train grid: {sizes.tolist()}")

  # 1 worker (single GPU, or the MPS dev machine) -> serial on this one device,
  # reusing the experiment; otherwise one process per GPU, LPT-balanced.
  if n_workers <= 1:
    log(f"device: {exp.device}  |  serial (1 worker)")
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
    log(f"parallel sweep across {n_workers} GPUs (LPT-balanced):")
    fracs = _run_parallel(cfg=cfg,
                          sizes=sizes,
                          n_workers=n_workers,
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
