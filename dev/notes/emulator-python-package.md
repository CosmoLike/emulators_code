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

## Drivers moved beside emulator/ + cocoa CLI (UPDATED 2026-06-30b)

The 4 drivers were MOVED out of `driver/` to the package root (beside
`emulator/`). In a cocoa install that root is
`external_modules/code/emulators/emultrf/dev/`; launch
`python .../dev/<driver>.py` from `$ROOTDIR`. Because the script's own folder is
`sys.path[0]`, `import emulator` resolves with NO bootstrap: the old
`ROOT = dirname(dirname(abspath(__file__)))` + `sys.path.insert` hack is GONE from
all 4 (living in `driver/` is what forced the two-level dirname, and `abspath` is
not symlink-safe in cocoa's tree). Dropped `import sys` from all 4 and `import os`
from tune (each used os/sys only in the bootstrap; train/sweep/bakeoff keep os for
`os.environ.setdefault("MPLBACKEND","Agg")`). Verified: with the script dir on
sys.path[0], `import emulator` + `import emulator.cocoa` both resolve.

New module **emulator/cocoa.py** gives the cocoa CLI: `add_cocoa_path_args`
(registers `--root` / `--fileroot` / `--yaml`), `resolve_cocoa_config` (reads
`$ROOTDIR`; `root = $ROOTDIR/<--root>`, `fileroot = root/<--fileroot>`; mkdir
`root/chains`; load `fileroot/<yaml>` defaulting to `test.yaml`; rewrite the data
block's train/val dv/params/covmat keys to absolute under `root/chains`), and
`cocoa_output(fileroot, path)` (place an output under fileroot). So the YAML
`data` block now lists BARE filenames (resolved under `--root/chains`), while
`cosmolike_data_dir` / `cosmolike_dataset` still resolve under
`$ROOTDIR/external_modules/data`. Canonical call:
`python .../dev/train_single_emulator_cosmic_shear.py --root projects/lsst_y1/
--fileroot emulators/nla_cosmic_shear/ --yaml test.yaml --diagnostic diag.pdf`.
The example YAML templates were moved to `example_yamls/` (runtime still reads
the YAML from `--fileroot`, so these are copy-from templates, not read in place).
README updated to match (drivers beside emulator/, the cocoa invocation, deploy
path). ROOTDIR itself: [[cocoa-rootdir-env]].

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

## Session 2026-07-04 (architecture day; full design history in
## [[nla-as-design-spec]], blocks 04c..04j)

THE INTERFACE NOW (all verified, 5 venv suites green; NOT yet run in
production): train_args.model = {name: resmlp|rescnn|restrf, ia:
absent|nla (tatt reserved), mlp: {width, n_blocks} (required),
activation: {type, n_gates} or bare string, cnn: {kernel_size,
n_blocks, gate_init}, trf: {n_heads, n_blocks, n_mlp_blocks,
shared_mlp, gate_init}, compile_mode}. MODELS keyed by (name, ia);
IA_DESIGNS + MODEL_BLOCK_KEYS + ARCH_HEAD in experiment.py. INACTIVE
head blocks silently ignored (switch models by name: alone); unknown
keys in ACTIVE blocks are loud errors; old flat keys error.
run_tag/display names unchanged (rescnn_nla etc.).

MODELS: ResCNN = bins-as-channels single kernel (Conv1d G->G; nla
T*G->T*G; pad_idx scatter/gather; only kernel_size+n_blocks; tensors
never exceed padded dv -> bandwidth-safe by construction). ResTRF =
bin tokens at NATURAL width max_bin, NO embed/out adapters (paper's
adapters existed only for its synthetic latent sequence), nla =
(template,bin) pairs = 90 tokens, n_heads must divide max_bin
(26->1|2|13, default 2), per-token unique MLPs (BinLinear) or
shared_mlp: true (textbook ablation; position-blind -- see TO
DISCUSS), TRFBlock == identity at init (zeroed branch outputs),
corr = blocks(h)-h. All correction heads zero-init identity;
set_train_phase on the Template variants -> trunk_epochs two-phase +
SYMMETRIC trunk:/head: per-phase override blocks (lr_base/loss_mode/
trim/focus over shared defaults; either without trunk_epochs>0
raises).

TRAINING-LOOP FIXES (04j, found by the user's 300+700 test): warmup
lr now applied BEFORE each epoch (was after -> epoch 1 of every pass
trained at FULL base lr, wrecking the phase-2 handoff); baseline
epoch-0 eval seeds best-tracking (a pass can never end worse than it
started; new "epoch 0 baseline" stdout line).

RESUME STATE: (1) rerun two-phase rescnn+nla with the fixed loop;
head block advice = gate_init 0.1, chi2 + small annealed trim (start
0.05 -> 0) until the trunk is mature; expect phase-2 epoch 1 ~=
phase-1 best now. (1b) 07-04m: conv-as-matmul REVERTED (no GPU gain
-- CPU-only pathology; native Conv1d restored) and clip/rewind
stability guards added (YAML: clip / rewind, top-level or per-phase;
next chi2-head run wants head {trim end 0.01, clip 1.0} + rewind
true) -- re-sync before the next run. (2) restrf first runs (bin
tokens; expect near-nla
epoch cost). (3) POSITIONAL-ENCODING discussion pending (see TO
DISCUSS in [[nla-as-design-spec]]). (4) production YAML gotchas: trf
block must NOT have width (deleted knob) and n_heads must divide 26
(use 2). (5) cocoa deploy: re-sync the WHOLE dev tree (models,
building blocks, experiment, training, geometries_parameter,
IA/loss_functions, parallel/ (activations.py DELETED), all 4 drivers,
both example_yamls, README). GOAL REFRAME (2026-07-04, user): the
0.1105 rescnn+nla number (07-04l) is STALE -- do not treat it or
"goal 0.10" as the target of the current runs. Current two-phase
runs are "non-absurd tests" only: does training run end to end,
handoff loss-continuous, no loss jumps/explosions. SMOKE TESTS
PASSED (07-04, user: "everything is in order"): rewind verified at
a live lr cut (near-no-op at a healthy plateau, as designed);
trunk-vs-head param print exact to the digit incl. separable
(4,427); handoff baseline seeded from phase-1 best; groups=6 +
rescale_kernel + separable all ran in production. The whole 07-04
feature stack is validated; next stop is the TATT + w0wa dumps.
CURRENT PRODUCTION TEMPLATE (07-04 end of day, user-authored):
sqrt loss both phases; trunk trim 0.1 -> 0.01 (hold 50, anneal
400) + focus 0 -> 2 (kappa 0.15); head lr_base 0.001, trim 0.03 ->
0.01 (hold 15, anneal 100), focus now ENABLED 0 -> 2 kappa 0.2
with a halved schedule (hold ~15, anneal ~50; rationale in
[[nla-as-design-spec]] 04s); trim floors 0.01 everywhere; rewind
true recommended always; gate_init 1 in recent smokes (safe with
the zero-init identity). Full cnn block: kernel_size 11 +
rescale_kernel true (-> k=7 at 2 blocks), groups 6, separable
true (head 4,427 params; separable = parameter economy NOT speed,
3.9s vs 2.9s plain on the small-SM GPU). YAML style: block style
only, never {...}. README is a first-class doc surface (it went
stale repeatedly). The REAL
comparison arrives with TATT + w0wa dumps: that is where a
ResMLP-only trunk demands many more training points and the lean
factored conv head (rescale_kernel + groups; production smoke config
07-04q: kernel_size 11 target, 2 blocks -> k=7, groups=6 -> head
19,187 vs trunk 76,048 excluding-linear) is supposed to buy sample
efficiency with structure, not capacity. TATT code is LIVE as of 07-04v
(ia: tatt -> IA_DESIGNS entry with amps LSST_A1_1/LSST_A2_1/
LSST_BTA_1, tatt_coeffs, 10 templates; same Template* classes;
groups 1|10|20; film works at tatt dims) -- BLOCKED only on the
template training dumps, which do not exist yet. film flag now on
BOTH head families (cnn + trf).
END OF DAY 07-04 (authoritative; supersedes items 1/1b/5 above --
all DONE and production-verified). The day's full arc lives in
[[nla-as-design-spec]] blocks 04k-04w; state at close:
LOOP/PERF: compiled fwd_loss (model+loss in one graph; static-shape
_reduce = sort+mask == topk, tensor trim/focus/KAPPA scalars -- the
kappa float was the primals_90 CPU-lift that broke CUDA-graph
replay on recompile, fixed + CONFIRMED gone in production) + eval
twin fwd_chi2 (consume-per-batch, ~1GB/epoch churn gone) +
per-chunk pre-shuffle. Contended trunk epochs 2.2-2.4 -> 1.5s;
head 5.1s with replay. audit_devices() names off-device tensors at
run start. Option 4 (full-step graph) parked with estimates; at
10M dvs the game becomes prefetch overlap (H2D), not graphs.
HEADS: film on BOTH families (cnn + trf; identity-init
FiLMGenerator per block, conditioning ALWAYS amplitude-blind);
first evidence STRONG: truncated 150-epoch trunk (frac 0.5000) +
film head -> 0.2766 by head epoch 22 where the film-less head
stalled ~0.37; result reproduced across runs. groups (2|3|6,
pm_kept-validated), separable, rescale_kernel all composed in
production. TATT LIVE, blocked only on template dumps.
NEXT (in rough order): (a) transfer-learning practice run (04w:
per-phase training subsets -- head on independent N/2 rows;
mechanism NOT built yet: a head: n_train knob slicing disjoint
rows); (b) n_blocks_cnn sweep 3->5 on the truncated-trunk film
setup (~4.2k params/block); (c) TATT dumps -> the real
sample-efficiency comparison; (d) quiet-machine timing numbers
when amypond's MCMC ends; (e) still parked: positional encoding
(TRF), film_grouped, conditional LayerNorm.
TESTS: 14 scratchpad suites, all green at close (rescnn_bins,
restrf, yaml_activation, phase_lr, warmup_baseline, clip_rewind,
rescale_kernel, param_split, groups, separable, fwdloss_compile,
film, trf_film_tatt, device_audit) in
/private/tmp/claude-501/-Users-vivianmiranda-data-COCOA-june2026-
emulators-code-dev/b054b93a-.../scratchpad (venv gdvenv there;
scratchpad survives compaction of THIS session, not a new one --
recreate from the notes' per-block test descriptions if lost).
Working tree has everything since the user's last commit.
POST-COMPACT ADDENDUM (same day, 07-04x; full detail in
[[nla-as-design-spec]] 04x): (a) EmulatorExperiment.print_design()
= the shared startup banner (model spec, two-phase split,
clip/rewind guards, trunk:/head: blocks, cuts); all three drivers
call it; train/sweep/tune headers brought current (tatt, the 5 cnn
knobs, trf film, clip/rewind; tune's flat-schema example was
PRE-NESTED and would error -- fixed; search ranges nest at any
depth, docs now say so). (b) parallel/: ParallelResCNN +=
needs_geom/needs_bins flags + block_opts act threaded into
GroupedCNNBlock; full doc rewrite with forward graphs (the user's
flagged "incomplete and nonformal" example). (c) PCE/: verified
current by construction (delegated _reduce; target_dim +
needs_params honored by the loaders and the compiled twins; the
recorded pack-at-load to-do was already built) + doc rewrite with
graphs. (d) batching regime-ladder + data_staging staging-pipeline
graphs; comment-header fns -> formal docstrings (the batching
sizing trio, param_stats, plot_xi). Test battery now 16 suites
(new: test_print_design, test_pce_parallel), all green.
SECOND ADDENDUM (07-04y; full detail in [[nla-as-design-spec]]
04y): NEW DRIVER sweep_hyperparam_emulator_cosmic_shear.py (one
YAML-chosen train_args leaf via a sweep: block -- dotted path +
values; activation special-cased through exp.activation; name/ia
+ typo paths refused; save_sweep_table + plot_sweep_curve outputs).
scheduling.py grew even_assign + run_gpu_pool (spawn pool, lanes
per GPU, keepalive fix for the py3.14 Process-releases-args ->
SemLock-unlink crash) + the --gpu-pack token machinery
(estimate_train_vram_fraction + vram_tokens: <=20% -> 4/GPU,
<=40% -> 2/GPU, else exclusive; off by default; engages on a
single GPU too). sweep_ntrain rewired onto the pool.
tune_single now multi-GPU: --n-gpus + --journal, one worker per
GPU sharing an optuna JournalStorage study (per-worker sampler
seeds; same journal resumes). DOC SURFACES: README section 6 has
anchored subsections "The sweep: block" + "Multi-GPU execution
and packing" (driver x split table, token ladder, journal
semantics); NEW example_yamls/sweep_hyperparam_emulator_cosmic_
shear.yaml (active lr sweep + 5 commented swap-ins); tune YAML
header documents --n-gpus/--journal. Battery 17 suites
(+test_gpu_pool_pack, 21 checks); GPU-side verification pending
on amypond (Mac has no CUDA).

Historical context (07-04l run): phase 2 collapsed to frac ~0.305
after head epoch ~272 (untrimmed-chi2 fit the monster tail; full
post-mortem in [[nla-as-design-spec]] 04l); fixes since = head trim
floor 0.01 + clip + rewind.

## Session 2026-07-03 (T=256 production day; the big feature batch)

LATE ADDITIONS (same day, after the rescnn_nla build): activation is
now a YAML key, train_args.model.activation (H | power | multigate |
gated_power) + model.n_gates (K for the gated families, default 3).
Precedence resolved ONCE in from_config: an explicit --activation (the
3 drivers' argparse default changed "H" -> None so an absent flag
defers) > YAML > "H"; build_specs STRIPS activation/n_gates from the
model-block spread (like name) and threads n_gates into
make_activation -- do NOT re-read activation in build_specs, that
would flip precedence back to the YAML. bakeoff_activation is
unaffected (passes activation explicitly, wins over YAML by design).
Also: capability flags replaced the model isinstance checks
(factored=True on TemplateMLP/TemplateResCNN -> AmplitudeFactor
geometry + template loss; conv_head=True on ResCNN/TemplateResCNN ->
geom injection + compile_mode setdefault "default"), and the rescale
guard moved above build_geometry's lazy cosmolike import (fail fast,
testable off-workstation).

Driver/infra added (all in repo, verified): per-epoch + steady s/epoch
timing in training_loop_batched; save_emulator in results.py (.emul =
cpu state_dict with _orig_mod stripped; .h5 = geometry state() groups
written RECURSIVELY + histories + config_yaml + attrs; always saved,
BEFORE diagnostics); run_tag naming (<root>_<model>_t<T>_ntrain<N>);
run products -> --root/chains (resolve_cocoa_config returns chains);
startup banner prints the full resolved design (model spec + all
train_args blocks + cuts) to catch stale YAMLs; physical cuts
omegabh2_lo + omegam2h2_lo/hi in phys_cut_idx/YAML; diagnostics pages
4-6 (chi2 triangle + omegamh2 derived axis + GRAY CUT SHADING via
_cut_exclusion; standardized-ln-PCA colored by chi2 AND by sparsity
with fitted monomial + R^2; colors clipped to [-2, 1.5] band, sorted
draw order, pinned colorbars); coverage histogram FD-binning fix.
Factored-IA wired: model.name nla / nla_as (registry shares
TemplateMLP -> exp.model_name disambiguates; AmplitudeFactorGeometry
carry support + names + state round-trip; encoded_dim property +
run_emulator getattr injection; nla_coeffs + AsScaledNLAChi2).
RESULTS: resmlp 0.1558 / nla 0.1472 (winner) / nla_as 0.1559
(abandoned) at T=256, 250k, in-window; runs 17 min at bs 768.
DIAGNOSIS STATE: in-window coverage ISOTROPIC (sparsity fit R^2 0.01),
hardness diffuse (R^2 0.18) -> levers are N (sweep, ~1M pool, 4x
headroom) + representation (NEXT BUILD: rescnn_nla = TemplateMLP trunk
+ shared per-template ResCNN conv head, W_fd/W_df buffers + act_mid).
REUSABLE METHOD: PDF forensics -- extract a getdist scatter's vector
points + viridis-invert colors from the diagnostic PDF (PyMuPDF
get_drawings + tick-label calibration), validated by the physical
wedge identity; powered the omegam2h2 cut-scan without workstation
access.
