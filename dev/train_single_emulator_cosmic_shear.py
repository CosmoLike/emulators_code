#!/usr/bin/env python3
"""Train one cosmic-shear (xi) emulator (ResMLP or ResCNN) from a YAML."""

#-------------------------------------------------------------------------------
# Example how to run this program
#-------------------------------------------------------------------------------
# Trains one cosmic-shear (xi) emulator -- ResMLP or ResCNN (ResMLP trunk +
# 1D-CNN appendix), chosen in the YAML -- from cosmological parameters to the
# whitened, masked xi data vector. Loss = full-3x2pt chi2 (cosmolike's masked
# inverse covariance).
#
#     python external_modules/code/emulators/emultrf/dev/train_single_emulator_cosmic_shear.py \
#       --root projects/lsst_y1/ \
#       --fileroot emulators/training_scripts/ \
#       --yaml train_single_emulator_cosmic_shear.yaml \
#       --diagnostic diagnostic.pdf
#
#- Cocoa layout: export $ROOTDIR, then --root names the project folder under it
#  ($ROOTDIR/projects/lsst_y1) and --fileroot a subfolder of it holding this
#  emulator's YAML configs ($ROOTDIR/projects/lsst_y1/emulators/
#  training_scripts). The data files (dv / params / covmat) and the run
#  products (the diagnostics PDF) live under --root/chains; the YAML under
#  --fileroot. The driver resolves every path, so it runs from $ROOTDIR
#  regardless of cwd. cosmolike's own dataset still resolves under
#  $ROOTDIR/external_modules/data. cosmolike runs only on the workstation; train
#  there.
#
#- This script sits beside the emulator/ package (same .../emultrf/dev/ folder),
#  so `import emulator` needs no sys.path edit; just run it from $ROOTDIR.
#
#- `--root` (required): project folder under $ROOTDIR (e.g. projects/lsst_y1);
#  the data files resolve under --root/chains.
#- `--fileroot` (required): subfolder of --root holding this emulator's YAML
#  configs (e.g. emulators/training_scripts).
#- `--yaml` (default test.yaml): config file under --fileroot, holding every
#  hyperparameter (no magic numbers in code). Two blocks:
#  - `data`: input file names (train_dv, train_params, train_covmat, val_dv,
#    val_params -- bare filenames, resolved under --root/chains), cut/split
#    settings (omegabh2_cut, train_divisor, val_divisor, split_seed, ram_frac),
#    cosmolike dataset (cosmolike_data_dir, cosmolike_dataset; resolved under
#    $ROOTDIR/external_modules/data, not --root).
#  - `train_args`: knobs (nepochs, bs, loss_mode, silent) plus sub-blocks model
#    (name = resmlp | rescnn, then kwargs: int_dim_res, n_blocks, and for rescnn
#    kernel_size / channels / n_blocks_cnn / gate_init), optimizer (weight_decay),
#    lr (lr_base, bs_base, warmup_epochs), scheduler (mode, patience, factor),
#    trim / focus (robustness schedules).
#
#- `--diagnostic` (optional): the name root of a multipage diagnostics PDF,
#  saved under --root/chains (an absolute path keeps its folder). The driver
#  appends the run's identity, so `--diagnostic diagnostic.pdf` writes e.g.
#  diagnostic_resmlp_t256_ntrain250000.pdf (model name, training temperature
#  from the train-dv's _cs_<T> tag, staged N_train). Page 1
#  (2x2): training history + coverage (do failures sit in sparse training
#  regions?). Page 2: local-linear data-only floor (model vs floor delta-chi2;
#  plain chi2fn only, skipped under --rescale). Page 3: hard-direction regression
#  (which log-param combo predicts hardness). Page 4: getdist triangle of the
#  val cosmologies over the basic LCDM parameters (A_s, n_s, H0, Omega_b,
#  Omega_m; no tau in the dumps) plus the derived omega_m h^2, every point
#  colored by its log10 delta-chi2, showing where in parameter space the
#  emulator fails. Page 5: the val cosmologies on the first two principal
#  components of the ln parameters (sample-covariance PCA; a PC in ln space
#  is a product of parameter powers, e.g. As^a H0^b omegam^c, and the axis
#  labels spell the exponents out), colored the same way; a color gradient
#  along a PC names the power-law combination the emulator finds hard. Omit
#  for no figure.
#
#- `--rescale` (optional, default `none`): divides out a fast analytic R so the
#  net emulates a flatter target (chi2 stays on the original dv). `rescaled` =
#  RescaledChi2 (v1: R divides the net output, so the chi2 gradient carries a
#  per-cosmology 1/R factor); `residual` = ResidualBaseChi2 (v2: R moves the
#  baseline only, plain chi2). Both need cosmolike's angle map.
#
#- `--activation` (optional, default `H`): ResBlock activation -- `H` (paper's
#  leaky/Swish gate), `power` (bounded learnable tail exponent), `multigate` (K=3
#  gates), or `gated_power` (K=3 gates + tail exponent).
#
#- `--quiet` (optional): suppresses all stdout -- driver prints, load_source's
#  per-source line, run_emulator's per-epoch log. The --diagnostic PDF still writes.
#
#- Fixed single-emulator choices -- probe = xi, AdamW, ReduceLROnPlateau,
#  use_amp = False, reported delta-chi2 thresholds [0.2, 0.5, 1, 10, 100]
#  (0.2 = goal and model-selection metric), resmlp/rescnn registry -- are
#  EmulatorExperiment defaults (emulator/experiment.py, which also holds the
#  setup for a sweep to reuse). The model is the YAML's choice
#  (train_args.model.name).
#
#- Inputs (filenames set in the YAML `data` block, resolved under --root/chains):
#
#      <train_dv>.npy      training data vectors   (memmapped)
#      <train_params>.txt  training parameters     (weight, lnp, <params>, chi2)
#      <train_covmat>      parameter covmat        (header line = param names)
#      <val_dv>.npy        validation data vectors
#      <val_params>.txt    validation parameters
#
#- Outputs:
#
#      stdout            per-epoch progress (unless train_args.silent: true) plus
#                        a final "best epoch N: frac>0.2 ... median ..." line.
#      <--diagnostic>_<model>_t<T>_ntrain<N>.pdf   the multipage diagnostics
#                        PDF (under --root/chains), if --diagnostic is set.
#-------------------------------------------------------------------------------

import argparse
import os
import re

# This script sits beside the emulator/ package (same .../emultrf/dev/ folder),
# so launching it by path makes its own directory sys.path[0] and
# `import emulator` resolves with no path manipulation. Run it from $ROOTDIR;
# emulator.cocoa reads $ROOTDIR to resolve the data paths.

from emulator.cocoa import (
  add_cocoa_path_args, resolve_cocoa_config, cocoa_output)
from emulator.experiment import EmulatorExperiment


def main():
  parser = argparse.ArgumentParser(
    prog="train_single_emulator_cosmic_shear")
  # --root / --fileroot / --yaml: the cocoa project layout (data + run
  # products under --root/chains, YAML configs under --fileroot).
  add_cocoa_path_args(parser)
  parser.add_argument("--diagnostic",
                      dest="diagnostic",
                      help="if set, save a multipage diagnostics PDF "
                           "under --root/chains; this is the name "
                           "root, and the run identity is appended "
                           "(diagnostic.pdf -> diagnostic_resmlp_"
                           "t256_ntrain250000.pdf)",
                      type=str,
                      default=None)
  parser.add_argument("--rescale",
                      dest="rescale",
                      help="analytic-R rescaling mode: 'none' "
                           "(plain chi2, default), 'rescaled' "
                           "(RescaledChi2 / v1: R divides the net "
                           "output), or 'residual' "
                           "(ResidualBaseChi2 / v2: R moves only "
                           "the baseline)",
                      type=str,
                      choices=["none", "rescaled", "residual"],
                      default="none")
  parser.add_argument("--activation",
                      dest="activation",
                      help="ResBlock activation: 'H' (the paper's "
                           "H, default), 'power', 'multigate' "
                           "(K=3), or 'gated_power' (K=3)",
                      type=str,
                      choices=["H", "power", "multigate",
                               "gated_power"],
                      default="H")
  parser.add_argument("--quiet",
                      dest="quiet",
                      help="suppress all stdout: the driver's "
                           "prints, load_source's per-source line, "
                           "and run_emulator's per-epoch log",
                      action="store_true")
  args, unknown = parser.parse_known_args()

  # Resolve the cocoa layout: $ROOTDIR/<root> holds the data, <fileroot>
  # (under root) holds this emulator's YAML; run products (the diagnostics
  # PDF) go to the project chains/ folder. Loads the YAML and rewrites its
  # data paths to absolute, so the run does not depend on the launch
  # directory.
  cfg, _, chains = resolve_cocoa_config(args)

  # All setup -- config parse + model resolution + device + data staging +
  # geometry + chi2 + spec assembly -- lives in EmulatorExperiment, so a sweep
  # script reuses it rather than copying it. The fixed single-emulator choices
  # are its defaults; the model is the YAML's choice. This driver passes only
  # what it varies (rescale, activation, quiet).
  exp = EmulatorExperiment.from_config(cfg,
                                       rescale=args.rescale,
                                       activation=args.activation,
                                       quiet=args.quiet)
  # the experiment's quiet-gated logger, reused below
  log = exp.log
  log(f"device: {exp.device}  |  rescale: {exp.rescale}")
  log("loading sources:")
  (model, train_losses, medians,
   means, fracs) = exp.run()

  # run_emulator already restored the best-frac>0.2 epoch; report which one.
  # fracs[i][0] is frac>0.2 at epoch i+1, median the tiebreaker (loop's rule).
  best = min(range(len(fracs)),
             key=lambda i: (fracs[i][0].item(), medians[i]))
  log(f"best epoch {best + 1}: "
      f"frac>0.2 {fracs[best][0].item():.4f}  "
      f"median {medians[best]:.4f}")

  if args.diagnostic is not None:
    # --diagnostic is a name root: append the run's identity so runs do
    # not overwrite each other and the file says what produced it,
    #   diagnostic.pdf -> diagnostic_resmlp_t256_ntrain250000.pdf
    # tags = model name (YAML train_args.model.name), training
    # temperature (the _cs_<T> tag in the train-dv file name, skipped
    # when absent), and the N_train actually staged.
    stem, ext = os.path.splitext(args.diagnostic)
    tags = [str(cfg["train_args"]["model"].get("name", "resmlp")).lower()]
    tmatch = re.search(r"_cs_(\d+)",
                       os.path.basename(cfg["data"]["train_dv"]))
    if tmatch is not None:
      tags.append(f"t{tmatch.group(1)}")
    tags.append(f"ntrain{exp.train_set['idx'].shape[0]}")
    diag_name = f"{stem}_{'_'.join(tags)}{ext or '.pdf'}"
    # a run product goes to the project chains/ folder (with the dvs),
    # not the fileroot (which holds the YAML configs).
    diag_path = cocoa_output(chains, diag_name)
    # headless output: pick a non-interactive matplotlib backend before pyplot
    # is imported (emulator.plotting imports it at load), then build it.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from emulator.diagnostics import (
      coverage_diagnostic, local_linear_floor,
      hard_direction_regression)
    from emulator.plotting import plot_diagnostics
    # (1) coverage: do failing val points sit in sparse training regions? (local
    # kNN sparsity vs delta-chi2).
    cov = coverage_diagnostic(model=model,
                              param_geometry=exp.pgeom,
                              chi2fn=exp.chi2fn,
                              train_set=exp.train_set,
                              val_set=exp.val_set,
                              device=exp.device)
    log(f"coverage: spearman(knn_dist, log dchi2) "
        f"{cov['spearman']:+.3f}  |  median knn good "
        f"{cov['median_good']:.3f} bad {cov['median_bad']:.3f}  "
        f"|  frac>0.2 dense {cov['frac_dense']:.3f} sparse "
        f"{cov['frac_sparse']:.3f}")
    log("=> " + ("COVERAGE-limited: failures sit in sparse regions"
                 if cov["coverage_limited"]
                 else "NOT clearly coverage: failures not sparser"))
    # (2) hard-direction regression (works for any chi2fn).
    hd = hard_direction_regression(model=model,
                                   param_geometry=exp.pgeom,
                                   chi2fn=exp.chi2fn,
                                   val_set=exp.val_set,
                                   device=exp.device)
    log(f"hardness: joint log-linear R2 {hd['r2']:.3f}  |  "
        f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")
    # (3) local-linear data floor -- plain chi2fn only (rescaled encode/chi2
    # would need each point's own R).
    floor = None
    if not getattr(exp.chi2fn, "needs_params", False):
      floor = local_linear_floor(model=model,
                                 param_geometry=exp.pgeom,
                                 chi2fn=exp.chi2fn,
                                 train_set=exp.train_set,
                                 val_set=exp.val_set,
                                 device=exp.device)
      log(f"floor: f_model {floor['f_model']:.3f}  "
          f"f_floor {floor['f_floor']:.3f}  "
          f"pure hardness {floor['f_hard']:.3f}")
    else:
      log("floor: skipped (local-linear floor needs a plain "
          "chi2fn; --rescale is on)")
    # val_set + names add page 4: the getdist LCDM triangle of the val
    # cosmologies colored by log10 delta-chi2 (cov["dchi2"], same rows).
    plot_diagnostics(train_losses=train_losses,
                     medians=medians,
                     means=means,
                     fracs=fracs,
                     thresholds=exp.thresholds,
                     coverage=cov,
                     floor=floor,
                     hard_dir=hd,
                     val_set=exp.val_set,
                     names=exp.names,
                     savepath=diag_path)
    log(f"saved diagnostics -> {diag_path}")


if __name__ == "__main__":
  main()
