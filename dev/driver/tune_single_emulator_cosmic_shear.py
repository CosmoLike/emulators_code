#!/usr/bin/env python3
"""Optuna search over single xi emulator hyperparams (ResMLP/ResCNN)."""

#-------------------------------------------------------------------------------
# How to run this program
#-------------------------------------------------------------------------------
# Tuning twin of train_single_emulator_cosmic_shear.py: same single cosmic-shear
# (xi) emulator setup (ResMLP or ResCNN, per the YAML), but runs an Optuna study
# minimizing validation f(delta-chi2 > 0.2) rather than one run.
#
#     python driver/tune_single_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/nla_cosmic_shear/ \
#       --yaml tune.yaml \
#       --n-trials 50 --timeout 4200
#
#- The searched hyperparameters come from the YAML train_args block. Each leaf is
#  a fixed scalar or a range: a 4-item list [default, min, max, kind], kind int /
#  float / log (a whitespace string "default min max kind" also works):
#
#      lr:
#        lr_base: [0.0025, 1.0e-5, 1.0e-1, log]   # searched, log scale
#        bs_base: 64.0                            # fixed
#      model:
#        int_dim_res: [128, 64, 256, int]         # searched, integer
#        n_blocks:    4                           # fixed
#
#  The first value is the default: the training driver uses it and this search
#  warm-starts trial 0 from it -- one YAML serves both drivers.
#
#- `--root` (required): project folder under $ROOTDIR (data resolves under it);
#  `--fileroot` (required): subfolder holding the YAML (e.g.
#  emulators/nla_cosmic_shear). Cocoa layout, as in the training driver.
#- `--yaml` (default test.yaml): config under --fileroot, `data` + `train_args`
#  blocks (training driver schema; train_args may now carry ranges). The `data`
#  block lists bare filenames, resolved under --root/chains.
#- `--n-trials` (default 50) and `--timeout` (seconds, optional) bound the study.
#- `--rescale` / `--activation` set the analytic-R mode and ResBlock activation,
#  fixed across the study, not searched (see the training driver).
#- `--quiet` suppresses all stdout (per-trial lines and final summary).
#
#- The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
#  use_amp = False, report thresholds, resmlp/rescnn registry) are
#  EmulatorExperiment defaults (emulator/experiment.py), shared with the training
#  driver. The model is the YAML's (train_args.model.name = resmlp | rescnn),
#  also fixed; only hyperparameters vary.
#
#- Output: stdout only -- a per-trial line (frac>0.2, running best, params) and a
#  final summary of the best frac>0.2 and params. No files.
#-------------------------------------------------------------------------------

import argparse
import os
import sys

import optuna

# The emulator package sits one directory up; put the repo root on sys.path so
# `import emulator` resolves from any working dir (see the training driver).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from emulator.cocoa import add_cocoa_path_args, resolve_cocoa_config
from emulator.training import suggest_train_args, search_defaults
from emulator.experiment import EmulatorExperiment


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
                           "study: 'H' (default), 'power', "
                           "'multigate', or 'gated_power'",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
  parser.add_argument("--n-trials",
                      dest="n_trials",
                      help="number of Optuna trials (default 50)",
                      type=int,
                      default=50)
  parser.add_argument("--timeout",
                      dest="timeout",
                      help="stop the study after this many seconds "
                           "(optional; default no limit)",
                      type=int,
                      default=None)
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout (per-trial lines "
                           "and the final summary)",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # Resolve the cocoa layout (data under $ROOTDIR/<root>, YAML under
  # <fileroot>); loads the YAML and makes its data paths absolute. This
  # study only prints to stdout, so the output fileroot is discarded.
  cfg, _ = resolve_cocoa_config(args)

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
  log(f"device: {exp.device}  |  rescale: {exp.rescale}  "
      f"|  activation: {exp.activation}")
  log("loading sources:")
  exp.stage_train()
  exp.stage_val()
  exp.build_geometry()

  # raw_train_args keeps the ranges (exp.train_args collapsed them to defaults);
  # suggest_train_args resolves them per trial
  raw_ta = exp.raw_train_args
  ranges = search_defaults(raw_ta)
  if not ranges:
    log("WARNING: no [default, min, max, kind] search ranges in "
        "train_args -- every trial is identical.")

  def objective(trial):
    # this trial's concrete train_args (each range -> a suggestion); exp.train
    # builds the per-run specs (model / optimizer / scheduler + activation +
    # ResCNN geom) on the fixed data + geometry. silent=True keeps each trial
    # quiet even when the study isn't.
    ta = suggest_train_args(trial, raw_ta)
    (_m, _tl, medians,
     _mn, fracs) = exp.train(train_args=ta, silent=True)
    # the run restored its best-frac>0.2 epoch (median tiebreaker); minimize that
    best = min(range(len(fracs)),
               key=lambda i: (fracs[i][0].item(), medians[i]))
    trial.set_user_attr("median", float(medians[best]))
    return fracs[best][0].item()

  def log_trial(study, trial):
    log(f"trial {trial.number:3d}  frac>0.2 {trial.value:.4f}"
        f"  best {study.best_value:.4f}  {trial.params}")

  # quiet Optuna's per-trial INFO spam (we print our own line)
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  # minimize frac>0.2; TPE seed fixed for reproducibility
  study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=0))
  # warm-start trial 0 from the YAML defaults (range first values) -- begin at
  # the known-good config
  if ranges:
    study.enqueue_trial(ranges)
  study.optimize(objective,
                 n_trials=args.n_trials,
                 timeout=args.timeout,
                 callbacks=[log_trial])

  log("\n--- search complete ---")
  log(f"best frac>0.2: {study.best_value:.4f}  "
      f"(median {study.best_trial.user_attrs.get('median', float('nan')):.4f})")
  log("best params:")
  for k, v in study.best_trial.params.items():
    log(f"  {k}: {v}")


if __name__ == "__main__":
  main()
