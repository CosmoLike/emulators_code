---
name: emulator-pipeline-and-goal
description: "Cosmic-shear ResMLP emulator (pytorch1.ipynb): goal is 90% of val cosmologies with delta-chi2 < 0.2; select model by lowest frac>0.2; pipeline overview."
metadata: 
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

pytorch1.ipynb builds a cosmic-shear data-vector emulator: a ResMLP maps
cosmological parameters to the whitened masked data vector; loss = full 3x2pt
chi2 with cosmolike's masked inverse covariance.

**Goal:** 90% of validation cosmologies with emulator error delta-chi2 < 0.2.
Track and select on `frac>0.2` (fraction of val points over 0.2), not the loss.

**Selection + scheduling discipline:**
- Best model = epoch with the lowest frac>0.2, ties broken by the lower median;
  the loop snapshots and restores those weights, so the returned model is that
  epoch (not the last).
- ReduceLROnPlateau steps on the val MEDIAN (a smooth signal), not frac>0.2
  (coarse k/Nval, would trip the patience counter erratically). Evaluation never
  trims; training trims the worst 5% (sqrt loss mode).

**Components (by name, never cell numbers — [[locate-notebook-edits-by-context]]):**
ParamGeometry / DataVectorGeometry+CosmolikeChi2 (input/output whitening + chi2;
fast sub-block path `self.Cinv_sq`, proven equal to the full-Cinv path);
build_loaders (3 data-placement regimes — resident / RAM-stream / disk-memmap —
chosen against a VRAM budget); training_loop_batched (warmup, best-model restore,
silent flag); run_emulator + make_model / make_optimizer / make_scheduler (see
[[construction-via-spec-dicts]]); plot_history. Dev on Mac MPS, real training on
NVIDIA ([[dev-machine-mac-m2-32gb]]).
