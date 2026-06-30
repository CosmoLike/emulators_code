---
name: emulator-python-package
description: "2026-06-29: the 'Data Vector emulator exercise 1' section of pytorch1.ipynb (READ-ONLY) was TRANSLATED into a real Python package emulator/ + CLI drivers driver/ -- the [[notebook-to-python-translation]] TODO is now substantially DONE for STRUCTURE (science unchanged). Layout: flat modules (data_staging, geometries_parameter, geometries_output, analytics, activations, emulator_designs_building_blocks, emulator_designs, loss_functions, batching, training, plotting, diagnostics) PLUS three subsystem SUBFOLDERS (parallel/, PCE/, IA/) each holding the same two-file shape emulator_designs.py + loss_functions.py (parallel/ also has activations.py + emulator_designs_building_blocks.py). THE FOLDER CARRIES THE QUALIFIER, so files inside DROP the suffix (parallel/activations.py, NOT activations_parallel.py); imports disambiguate by package path. ResCNNPerBin renamed ParallelResCNN. The port was BYTE-FAITHFUL: extract defs straight from the .ipynb JSON, scope to section cells (index>=240) to dodge earlier-chapter duplicates, dedup the twice-defined build_shear_angle_map / compute_model_size_bytes, verify with ast-parse + binding + unused-import + keyword-validity. cosmolike (ci) is imported ONLY in geometries_output. Drivers are dataset_generator_lensing-style (sys.path bootstrap, argparse, --yaml with data+train_args blocks, fixed choices hardcoded). New library construction helpers: build_run_specs (config->the six run_emulator spec dicts, KEYED for **splat), make_chi2 (geom+rescale->chi2fn), pick_device, make_activation, load_source/read_param_names, plus the diagnostics module and the [default,min,max,kind] search-range resolvers."
metadata:
  node_type: memory
  type: project
---

This session translated the "Data Vector emulator exercise 1" section of
pytorch1.ipynb (which stays the READ-ONLY reference) into a Python package
`emulator/` and CLI drivers `driver/`. Structure is done;
[[notebook-to-python-translation]] is the now-mostly-complete TODO.

## Package layout (emulator/)

Flat modules: data_staging (stream_* / param_stats / stage_source /
phys_cut_idx + read_param_names + load_source), geometries_parameter
(ParamGeometry, LogParamGeometry, NLAInputGeometry, AmplitudeFactorGeometry),
geometries_output (DataVectorGeometry, DiagonalGeometry, BlockDiagonalGeometry,
build_shear_angle_map), analytics (_analytic_R, analytic_shape_ratio,
rescale_xi), activations (activation_fcn + Gated/Power/GatedPower +
make_activation), emulator_designs_building_blocks (Affine, ResBlock, CNNBlock),
emulator_designs (ResMLP, ResCNN), loss_functions (anneal_value, CosmolikeChi2,
RescaledChi2, ResidualBaseChi2, ElementWeightedChi2, make_chi2), batching
(compute_* / batches_per_load / _build_loaders_one / build_loaders), training
(pick_device, make_model/optimizer/scheduler, build_run_specs, the
default/suggest/search train_args resolvers, eval_val, eval_source_chi2,
training_loop_batched, run_emulator), plotting (plot_history, plot_diagnostics +
the _history/_coverage/_floor/_hard_direction panel helpers,
source_param_samples, dv_to_xi, plot_xi), diagnostics (coverage_diagnostic,
local_linear_floor, hard_direction_regression).

Three SUBSYSTEM SUBFOLDERS, each the SAME two-file shape so the convention is
learnable: parallel/ (the FAILED per-bin variant -- Grouped* blocks +
ParallelResMLP + ParallelResCNN[was ResCNNPerBin]), PCE/ (NPCE -- the PCE
machinery + PCEEmulator, PCEResidualChi2 / PCERatioChi2), IA/ (factored
intrinsic-alignment -- NLATemplateMLP / TemplateMLP, NLAAmpFactoredChi2 /
TemplateFactoredChi2 + tatt_coeffs). THE FOLDER CARRIES THE QUALIFIER: a file
inside drops the suffix (parallel/activations.py, not activations_parallel.py);
`emulator.activations` vs `emulator.parallel.activations` disambiguate. The NLA-
specific trio is kept alongside the general one (not yet retired).

## The port was BYTE-FAITHFUL (reusable methodology)

Extracted each def/class straight from the .ipynb JSON (NO retyping), SCOPED to
the section's code cells (index >= 240) so earlier chapters' same-named helpers
did not leak (training_loop_batched appears 7x in the notebook;
compute_model_size_bytes had a stale GPU_MEM-global twin; stream_* / eval_val
also duplicated). Deduped the TWICE-defined build_shear_angle_map (kept the
bin_sizes / pm_kept version) and compute_model_size_bytes (kept the budget-arg
version). Verified mechanically every time: ast-parse all modules + binding
check (every cross-module symbol is defined-or-imported in its file) +
unused-import scan + validate every keyword arg against the callee's REAL
signature. cosmolike (ci) is imported only in geometries_output, so pure-torch
modules import anywhere and the package is reviewed statically (cosmolike runs
only on the workstation, [[dev-machine-mac-m2-32gb]]).

## Drivers (driver/) -- dataset_generator_lensing style

A 3-line sys.path bootstrap puts the repo ROOT on sys.path so `import emulator`
resolves regardless of launch dir (running `python driver/foo.py` puts driver/,
NOT the repo root, on sys.path -- the relative-import gotcha; `..` in a submodule
is PACKAGE-relative, never filesystem-relative). Config = a --yaml with `data`
(paths, cut/split, cosmolike dataset) + `train_args` (run knobs) blocks; the
script HARDCODES what makes it that driver (probe=xi, ResMLP, AdamW,
ReduceLROnPlateau, use_amp=False, thresholds).

- train_single_resmlp_emulator_cosmic_shear.py (+ .yaml): one training run. CLI:
  --yaml; --diagnostic <pdf> = a MULTIPAGE diagnostics PDF (page 1 = history +
  coverage 2x2; page 2 = local-linear data floor; page 3 = hard-direction
  regression; the floor page is skipped for a rescaled chi2fn); --rescale
  {none,rescaled,residual}; --activation {H,power,multigate,gated_power};
  --quiet.
- tune_single_resmlp_emulator_cosmic_shear.py (+ .yaml): an Optuna study
  minimizing val frac>0.2; CLI adds --n-trials, --timeout.

## Construction helpers (the run_emulator INPUT layer, in training.py)

- build_run_specs(train_args, model_cls, opt_cls, sched_cls) -> a DICT keyed by
  run_emulator's six spec args (model_opts / opt_opts / lr_opts / sched_opts /
  trim_opts / focus_opts) so a driver splats **specs. Each spec = {"cls": cls,
  **yaml_block} (caller picks the class, settings spread from the YAML). Keyed,
  NOT a positional 6-tuple (the "position X" trap, [[construction-via-spec-dicts]]).
- make_chi2(geom, rescale, param_geometry, cosmo_mid, data_dir, dataset,
  include_amp) -> the chi2fn (plain / RescaledChi2 v1 / ResidualBaseChi2 v2,
  [[geometry-loss-composition]]); LAZY-imports build_shear_angle_map so a plain
  build never pulls in the cosmolike geometry module.
- pick_device(name=None); make_activation(name, n_gates=3) (named act factory:
  H / power / multigate / gated_power, [[activation-function-generalizations]]).

## The [default, min, max, kind] search convention (ONE YAML, TWO drivers)

A train_args leaf is a fixed scalar OR a SEARCH range [default, min, max, kind]
with kind in {int, float, log} (a whitespace string "d min max kind" also
parses). FIRST value = the default. Resolvers in training.py:
default_train_args(ta) collapses ranges to defaults (the TRAIN driver uses this,
so its YAML can carry ranges and still train); suggest_train_args(trial, ta)
turns each range into an Optuna suggestion named by its dotted path
("lr.lr_base") and never imports optuna (it calls the passed trial.suggest_*);
search_defaults(ta) gives {path: default} to enqueue trial 0 (warm start). Casts
min/max to float so a YAML 1e-5 that PyYAML parsed as a string still works.

## EmulatorExperiment: the setup-object (ADDED 2026-06-30)

`emulator/experiment.py` holds **class EmulatorExperiment**, which factors the
WHOLE driver setup (config parse + model resolution + device + data staging +
geometry + chi2 + spec assembly + train) so the drivers and sweep scripts do not
copy it. Classmethods `from_yaml(path)` / `from_config(cfg)` (the dict path) both
resolve the model class from `train_args.model.name` through `MODELS = {"resmlp":
ResMLP, "rescnn": ResCNN}` (lives here, shared); the instance also keeps
`raw_train_args` (the UN-collapsed train_args, so a tuner suggests ranges per
trial). Composable methods: `stage_train(n_train=)` / `stage_val(n_val=)` (each a
FRESH gen seeded from split_seed, so N_train subsets are NESTED -- train set
identical to the old driver, only val rows shift), `build_geometry()`,
`build_specs()` (build_run_specs + activation inject + ResCNN geom inject),
`train(train_args=, silent=)`, `run()` (the full pipeline), `frac_above(thr)`
(the sweep metric, via eval_source_chi2), `pool_size()` (physical-cut row count =
the sweep's top N). The fixed single-emulator choices (probe=xi, AdamW,
ReduceLROnPlateau, use_amp=False, DEFAULT_THRESHOLDS, the registry) are
constructor DEFAULTS. BOTH train + tune drivers are refactored onto it (no setup
duplication; the tune loop is just `exp.train(train_args=suggest_train_args(trial,
exp.raw_train_args), silent=True)`).

## Driver family (renamed 2026-06-30; model chosen in the YAML)

The drivers DROPPED "resmlp" from their names: the model is the YAML's choice via
`train_args.model.name` (resmlp | rescnn) + the MODELS registry; ResCNN
additionally needs `geom` injected (build_specs / the drivers do it for it), and
on CUDA may need `compile_mode: default` in the model block.
- `train_single_emulator_cosmic_shear.{py,yaml}` -- one run.
- `tune_single_emulator_cosmic_shear.{py,yaml}` -- Optuna study.
- `sweep_ntrain_emulator_cosmic_shear.py` -- f(dchi2>thr) vs N_train for ONE
  config (run once per architecture / rescale, overlay the saved curves).
- `bakeoff_activation_emulator_cosmic_shear.py` -- activation x N_train DOUBLE
  loop, one curve per activation; N is the OUTER loop so the geometry is built
  once per N and shared across the inner activation loop.

Sweep output is PLAIN TEXT (np.loadtxt columns + a "#" metadata header), NOT json
-- the user prefers text matching the .txt / .covmat workflow. Helper
`save_learning_curves` in **emulator/results.py**; `plot_learning_curves`
(multi-curve overlay, {label: {N:frac}} or {label:(sizes,fracs)}) in plotting.py.
`load_source` gained `n_keep` (an ABSOLUTE row count; pass exactly one of
divisor / n_keep) so a sweep hits exact sizes from the deterministic nested pool.
`make_logger` (the --quiet print-gate factory) lives in training.py.

## Multi-GPU sweep + README (ADDED 2026-07-01)

sweep_ntrain + bakeoff_activation now run MULTI-GPU on a single node (NVWULF, up
to 8 H200): one torch.multiprocessing SPAWN process per GPU, each its own
EmulatorExperiment. New module **emulator/scheduling.py** `lpt_assign(sizes,
n_workers)` balances the N_train sweep by Longest-Processing-Time (cost ~ N); the
bake-off instead splits by ACTIVATION (equal cost, no LPT). Correctness bits:
spawn not fork, `torch.cuda.set_device(k)` per worker, del-refs + empty_cache
between points, `ram_frac=0` in the parallel path (stream the SHARED dump memmap,
no private per-worker copy). Serial fallback when n_workers<=1 (Mac/MPS); pure CPU
refused. Full methodology -> [[multi-gpu-sweep-pattern]].

**README.md** (repo root, cocoa-style): a numbered nested TOC, a teaching pipeline
section with ASCII flow diagrams (params -> whiten -> model -> chi2; the
experiment -> drivers orchestration; the dump -> subset -> 3 loader regimes; the
ResCNN W_fd/W_df basis-change), and per-file APPENDICES indexing every function.
All non-hot comprehensions in the package were converted to explicit C-style loops
([[py-module-style-conventions]]).

## NEXT

More drivers/variants (ResCNN, IA/TATT once the high-T TATT dataset exists,
[[npce-and-ia-template-factoring]]), wider CLI coverage of the remaining notebook
cells, the per-module documentation double-check, and (optional) retiring the
NLA-specific trio + the IA* renames. Style for this code:
[[py-module-style-conventions]].

**Why:** records that the notebook IS now a package + drivers, the exact layout
and naming convention, the faithful-port + verify methodology, the new
construction helpers, and the search-range convention -- so the next session
edits the package directly instead of re-deriving where everything went.
