#!/usr/bin/env python3
"""Optuna search over single xi emulator hyperparams (any architecture)."""

#-------------------------------------------------------------------------------
# How to run this program
#-------------------------------------------------------------------------------
# Tuning twin of train_single_emulator_cosmic_shear.py: same single cosmic-shear
# (xi) emulator setup (resmlp | rescnn | restrf, optionally with a factored
# ia design, per the YAML), but runs an Optuna study minimizing validation
# f(delta-chi2 > 0.2) rather than one run.
#
#     python external_modules/code/emulators/emultrf/dev/tune_single_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --n-trials 50 --timeout 4200
#
#- The searched hyperparameters come from the YAML train_args block. Each leaf is
#  a fixed scalar or a range: a 4-item list [default, min, max, kind], kind int /
#  float / log (a whitespace string "default min max kind" also works). A range
#  may sit at any nesting depth -- inside the model sub-blocks (model.mlp.width,
#  model.cnn.kernel_size), the schedules (trim.start), or the per-phase
#  trunk / head override blocks (head.lr_base):
#
#      lr:
#        lr_base: [0.0025, 1.0e-5, 1.0e-1, log]   # searched, log scale
#        bs_base: 64.0                            # fixed
#      model:
#        mlp:
#          width:    [128, 64, 256, int]          # searched, integer
#          n_blocks: 4                            # fixed
#
#  The first value is the default: the training driver uses it and this search
#  warm-starts trial 0 from it -- one YAML serves both drivers.
#
#- Multiple GPUs (one node): trials are independent trainings, so the study
#  parallelizes -- one worker process per GPU (spawn), all sharing ONE study
#  through an Optuna journal-file storage under --fileroot. Each worker builds
#  its own experiment on its GPU, asks the shared study for suggestions
#  (sampler seeded per worker), and reports results back; --n-trials is the
#  TOTAL across workers. The journal file persists: rerunning with the same
#  --journal RESUMES the study (more trials on top of the recorded ones);
#  delete the file (or pass a fresh --journal name) to start over. One GPU
#  (or the Apple-MPS dev machine) keeps the original serial in-memory study,
#  no file written.
#
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML (e.g.
#  emulators/training_scripts). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot, `data` + `train_args`
#  blocks (training driver schema; train_args may now carry ranges). The `data`
#  block lists bare filenames, resolved under --root/chains.
#- `--n-trials` (default 50) and `--timeout` (seconds, optional) bound the study
#  (the timeout applies per worker in the parallel path).
#- `--n-gpus` (default: all visible CUDA devices): GPUs to spread trials
#  across. 1, or no CUDA, is the serial path.
#- `--journal` (default tune_journal.log): the shared study's journal file
#  under --fileroot (parallel path only). Persistent: same name resumes,
#  new name starts fresh.
#- `--rescale` / `--activation` set the analytic-R mode and ResBlock activation,
#  fixed across the study, not searched (see the training driver).
#- `--quiet` suppresses all stdout (per-trial lines and final summary).
#
#- The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
#  use_amp = False, report thresholds, the (name, ia) MODELS registry) are
#  EmulatorExperiment defaults (emulator/experiment.py), shared with the training
#  driver. The model is the YAML's (train_args.model.name = resmlp | rescnn |
#  restrf, plus model.ia = omit | nla | tatt), also fixed; only hyperparameters
#  vary.
#
#- Output: stdout -- a per-trial line (frac>0.2, running best, params) and a
#  final summary of the best frac>0.2 and params -- plus, in the parallel
#  path, the persistent journal file.
#-------------------------------------------------------------------------------

import argparse

import optuna
import torch

# This script sits beside the emulator/ package (same .../emultrf/dev/ folder),
# so launching it by path makes its own directory sys.path[0] and
# `import emulator` resolves with no path manipulation. Run it from $ROOTDIR;
# emulator.cocoa reads $ROOTDIR to resolve the data paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.training import suggest_train_args, search_defaults
from emulator.experiment import EmulatorExperiment

# one shared study name inside the journal file; the file path (not
# this name) is what separates studies.
STUDY_NAME = "tune_single"


def journal_storage(path):
  """
  Open (or create) an Optuna journal-file storage.

  The journal backend is Optuna's recommended storage for several
  processes sharing one study through the filesystem: every worker
  appends its trial records to the file under a file lock, so no
  database server is needed and a crashed run leaves a readable
  log. The import shim covers the backend's rename
  (JournalFileStorage in optuna 3.x -> JournalFileBackend in 4.x).

  Arguments:
    path = the journal file's path (created if absent; appending to
           an existing file RESUMES its studies).

  Returns:
    an optuna.storages.JournalStorage.
  """
  from optuna.storages import JournalStorage
  try:
    from optuna.storages.journal import JournalFileBackend
  except ImportError:                     # optuna < 4.0
    from optuna.storages import JournalFileStorage as JournalFileBackend
  return JournalStorage(JournalFileBackend(path))


def _tune_worker(gpu_id, n_trials, cfg, rescale, activation,
                 journal_path, timeout, quiet):
  """
  One GPU's share of the study; runs in its own spawned process.

  Pins its GPU, builds and stages its own EmulatorExperiment there
  (each process has its own cosmolike global state), loads the
  SHARED study from the journal file, and runs its share of trials:
  Optuna serializes suggestions/reports through the storage, so
  workers cooperate on one search history. The sampler is seeded
  per worker (seed = gpu_id) -- identical seeds would propose
  identical points in parallel.

  Arguments:
    gpu_id       = CUDA device index this worker owns.
    n_trials     = this worker's trial share (the total splits
                   across workers).
    cfg          = parsed YAML config; host ram_frac already divided
                   across workers by the parent.
    rescale      = analytic-R mode, forwarded to the experiment.
    activation   = ResBlock activation name, forwarded likewise.
    journal_path = the shared study's journal file.
    timeout      = per-worker optimize() timeout in seconds (None =
                   unbounded).
    quiet        = suppress this worker's per-trial lines.
  """
  torch.cuda.set_device(gpu_id)
  device = torch.device(f"cuda:{gpu_id}")

  exp = EmulatorExperiment.from_config(cfg,
                                       device=device,
                                       rescale=rescale,
                                       activation=activation,
                                       quiet=True)
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()
  raw_ta = exp.raw_train_args

  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=journal_storage(journal_path),
    sampler=optuna.samplers.TPESampler(seed=gpu_id))

  def objective(trial):
    ta = suggest_train_args(trial, raw_ta)
    (_m, _tl, medians,
     _mn, fracs) = exp.train(train_args=ta, silent=True)
    best = min(range(len(fracs)),
               key=lambda i: (fracs[i][0].item(), medians[i]))
    trial.set_user_attr("median", float(medians[best]))
    return fracs[best][0].item()

  def log_trial(study_, trial):
    if not quiet:
      print(f"[gpu {gpu_id}] trial {trial.number:3d}  "
            f"frac>0.2 {trial.value:.4f}  "
            f"best {study_.best_value:.4f}  {trial.params}")

  study.optimize(objective,
                 n_trials=n_trials,
                 timeout=timeout,
                 callbacks=[log_trial])


def main():
  parser = argparse.ArgumentParser(
    prog="tune_single_emulator_cosmic_shear")
  # --root / --fileroot / --yaml: the cocoa project layout (data under
  # --root, YAML under --fileroot; train_args may carry [default, min,
  # max, kind] ranges).
  add_cocoa_path_args(parser)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode, fixed across "
                           "the study: 'none' (default), 'rescaled' "
                           "(v1), or 'residual' (v2)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation, fixed across the "
                           "study: 'H', 'power', 'multigate', or "
                           "'gated_power'. Overrides the YAML "
                           "train_args.model.activation; default: "
                           "the YAML's choice, else 'H'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default=None)
  parser.add_argument("--n-trials",
                      dest="n_trials",
                      help="number of Optuna trials (default 50)",
                      type=int,
                      default=50)
  parser.add_argument("--timeout",
                      dest="timeout",
                      help="stop the study after this many seconds "
                           "(optional; default no limit; applies "
                           "per worker in the parallel path)",
                      type=int,
                      default=None)
  parser.add_argument("--n-gpus",
                      dest="n_gpus",
                      help="number of GPUs to spread trials across "
                           "(default: all visible CUDA devices). "
                           "1, or no CUDA, is the serial path.",
                      type=int,
                      default=None)
  parser.add_argument("--journal",
                      dest="journal",
                      help="journal file (under --fileroot) the "
                           "parallel workers share the study "
                           "through (default tune_journal.log). "
                           "Persistent: the same name RESUMES the "
                           "recorded study, a new name starts fresh",
                      type=str,
                      default="tune_journal.log")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (per-trial lines "
                           "and the final summary)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # Resolve the cocoa layout (data under $ROOTDIR/<root>, YAML under
  # <fileroot>); loads the YAML and makes its data paths absolute. The
  # fileroot also hosts the parallel path's journal file.
  cfg, fileroot, _ = resolve_cocoa_config(args)

  # Setup -- config parse, model resolution, device, data staging, geometry,
  # chi2, per-run spec assembly -- lives in EmulatorExperiment, shared with the
  # training driver. Geometry / chi2 / activation are fixed, so build them once
  # here; only the searched train_args vary per trial. Single-emulator choices are
  # EmulatorExperiment defaults; model is the YAML's (train_args.model.name).
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  # the experiment's quiet-gated logger
  log = exp.log
  # print_design (experiment.py): the startup banner -- the resolved
  # model block, run knobs, guards, every train_args sub-block, and the
  # physical cuts (search ranges already collapsed to their defaults; the
  # study varies them per trial from there). A stale YAML here would
  # waste a whole study, not one training. Shared with the train / sweep
  # drivers.
  exp.print_design()

  # raw_train_args keeps the ranges (exp.train_args collapsed them to defaults);
  # suggest_train_args resolves them per trial
  raw_ta = exp.raw_train_args
  ranges = search_defaults(raw_ta)
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args -- every trial is identical.")

  # quiet Optuna's per-trial INFO spam (we print our own line)
  optuna.logging.set_verbosity(optuna.logging.WARNING)

  # how many GPU workers: capped by what is visible, by --n-gpus, and
  # by the trial count. 1 (or the MPS dev machine) keeps the original
  # serial in-memory study.
  n_cuda    = torch.cuda.device_count()
  n_request = n_cuda if args.n_gpus is None else min(args.n_gpus,
                                                     n_cuda)
  n_workers = min(n_request, args.n_trials)

  if n_workers <= 1:
    log("serial study (1 worker, in-memory)")
    log("loading sources:")
    exp.stage_train()
    exp.stage_val()
    exp.build_geometry()

    def objective(trial):
      # this trial's concrete train_args (each range -> a suggestion); exp.train
      # builds the per-run specs (model / optimizer / scheduler + activation +
      # ResCNN geom) on the fixed data + geometry. silent=True keeps each trial
      # quiet even when the study isn't.
      ta = suggest_train_args(trial, raw_ta)
      (_m, _tl, medians,
       _mn, fracs) = exp.train(train_args=ta, silent=True)
      # the run restored its best-frac>0.2 epoch (median tiebreaker); minimize
      best = min(range(len(fracs)),
                 key=lambda i: (fracs[i][0].item(), medians[i]))
      trial.set_user_attr("median", float(medians[best]))
      return fracs[best][0].item()

    def log_trial(study_, trial):
      log(f"trial {trial.number:3d}  frac>0.2 {trial.value:.4f}"
          f"  best {study_.best_value:.4f}  {trial.params}")

    # minimize frac>0.2; TPE seed fixed for reproducibility
    study = optuna.create_study(
      direction="minimize",
      sampler=optuna.samplers.TPESampler(seed=0))
    # warm-start trial 0 from the YAML defaults (range first values) --
    # begin at the known-good config
    if ranges:
      study.enqueue_trial(ranges)
    study.optimize(objective,
                   n_trials=args.n_trials,
                   timeout=args.timeout,
                   callbacks=[log_trial])
  else:
    # Parallel study: one spawned process per GPU, all cooperating on
    # ONE study through the journal file (Optuna's multi-process
    # storage; see journal_storage). The parent creates the study,
    # warm-starts it once, splits the trial budget, and reads the
    # result back after the workers join.
    import torch.multiprocessing as mp

    journal_path = cocoa_output(fileroot, args.journal)
    study = optuna.create_study(
      study_name=STUDY_NAME,
      storage=journal_storage(journal_path),
      direction="minimize",
      load_if_exists=True)
    done = len(study.trials)
    if done:
      log(f"resuming study in {journal_path}: {done} recorded "
          f"trial(s); adding {args.n_trials} more")
    # warm-start only a fresh study -- a resumed one already ran the
    # defaults as its trial 0.
    if ranges and done == 0:
      study.enqueue_trial(ranges)

    # split the TOTAL trial budget across workers (first workers take
    # the remainder), and divide the host-RAM staging budget so the
    # per-worker private copies of the same subset cannot together
    # overflow RAM (stage_source falls back to the shared memmap when
    # its share is too small).
    worker_cfg = dict(cfg)
    worker_cfg["data"] = dict(cfg["data"])
    worker_cfg["data"]["ram_frac"] = (
      float(cfg["data"].get("ram_frac", 0.7)) / n_workers)
    shares = []
    for k in range(n_workers):
      shares.append(args.n_trials // n_workers
                    + (1 if k < args.n_trials % n_workers else 0))
    log(f"parallel study across {n_workers} GPUs "
        f"(trials {shares}, journal {journal_path})")

    # spawn (not fork): each child needs its own CUDA context. The
    # workers log their own "[gpu k] trial ..." lines.
    ctx = mp.get_context("spawn")
    procs = []
    for k in range(n_workers):
      p = ctx.Process(target=_tune_worker,
                      args=(k,
                            shares[k],
                            worker_cfg,
                            args.rescale,
                            args.activation,
                            journal_path,
                            args.timeout,
                            args.quiet))
      p.start()
      procs.append(p)
    for p in procs:
      p.join()

    # re-read the study so the summary reflects every worker's trials.
    study = optuna.load_study(study_name=STUDY_NAME,
                              storage=journal_storage(journal_path))
    finished = False
    for t in study.trials:
      if t.state == optuna.trial.TrialState.COMPLETE:
        finished = True
    if not finished:
      raise RuntimeError(
        "no trial completed -- check the worker stderr above "
        f"(journal: {journal_path})")

  log("\n--- search complete ---")
  log(f"best frac>0.2: {study.best_value:.4f}  "
      f"(median {study.best_trial.user_attrs.get('median', float('nan')):.4f})")
  log("best params:")
  for k, v in study.best_trial.params.items():
    log(f"  {k}: {v}")


if __name__ == "__main__":
  main()
