**Warning** This pipeline is still in alpha stage `v0.05` and not ready for production. 

# Cosmic-shear data-vector emulator

A neural emulator that maps cosmological parameters to the masked cosmic-shear
(`xi`) data vector, trained against the full-3x2pt chi2 from cosmolike.

One line: raw dumps → stage → whiten params (input) and data vector (output) →
ResMLP / ResCNN / ResTRF → chi2 loss → train. `EmulatorExperiment` wires it together; each
driver varies one thing (one run, a tune, an `N_train` sweep, an activation
bake-off).

## Contents

1. [Layout](#1-layout)
2. [Pipeline](#2-pipeline)
3. [What each file does](#3-what-each-file-does)
4. [Change X → edit Y](#4-change-x--edit-y)
5. [Variants](#5-variants)
6. [Run it](#6-run-it)
    1. [The `sweep:` block (one-knob sweeps)](#6a-sweep-block)
    2. [Multi-GPU execution and packing](#6a-multi-gpu)
7. [Appendix: AI-Usage](#7-appendix-ai-usage)
8. [Appendix: the chi2 metric (Mahalanobis)](#8-appendix-the-chi2-metric-mahalanobis)
9. [Appendix: activation functions](#9-appendix-activation-functions)
10. [Appendix: every file's functions](#10-appendix-every-files-functions)
    1. [`data_staging.py`](#apx-data_staging)
    2. [`geometries_parameter.py`](#apx-geometries_parameter)
    3. [`geometries_output.py`](#apx-geometries_output)
    4. [`analytics.py`](#apx-analytics)
    5. [`activations.py`](#apx-activations)
    6. [`emulator_designs_building_blocks.py`](#apx-building_blocks)
    7. [`emulator_designs.py`](#apx-emulator_designs)
    8. [`loss_functions.py`](#apx-loss_functions)
    9. [`batching.py`](#apx-batching)
    10. [`training.py`](#apx-training)
    11. [`experiment.py`](#apx-experiment)
    12. [`scheduling.py`](#apx-scheduling)
    13. [`results.py`](#apx-results)
    14. [`plotting.py`](#apx-plotting)
    15. [`diagnostics.py`](#apx-diagnostics)
    16. [`parallel/`](#apx-parallel)
    17. [`PCE/`](#apx-pce)
    18. [`IA/`](#apx-ia)
    19. [drivers](#apx-drivers)

---

## 1. Layout

```
emulator/                              the library (pure torch, except geometries_output)
  data_staging.py                      load dumps -> "source" dicts; the physical cut
  geometries_parameter.py              INPUT whitening (params -> network input)
  geometries_output.py                 OUTPUT geometry + chi2 covariance (imports cosmolike)
  analytics.py                         analytic xi rescaling R (optional preprocessing)
  activations.py                       learnable activations (H + variants)
  emulator_designs_building_blocks.py  Affine, ResBlock, CNNBlock, BinLinear, TRFBlock
  emulator_designs.py                  ResMLP, ResCNN, ResTRF
  loss_functions.py                    chi2 losses + make_chi2
  batching.py                          memory sizing + regime-aware data loaders
  training.py                          build model/opt/sched, training loop, run_emulator
  experiment.py                        EmulatorExperiment: the whole setup as one object
  scheduling.py                        GPU job balancing + worker pool + VRAM packing
  results.py                           save_learning_curves; save_emulator (.emul + .h5)
  plotting.py                          history / learning-curve / coverage / xi plots
  diagnostics.py                       coverage, local-linear floor, hard-direction fits
  parallel/  PCE/  IA/                 experimental variants (section 5)

train_single_*.py                      CLI: one training run (+ optional diagnostics PDF)
tune_single_*.py                       CLI: Optuna hyperparameter search (multi-GPU)
sweep_ntrain_*.py                      CLI: f(dchi2 > thr) vs N_train   (multi-GPU)
sweep_hyperparam_*.py                  CLI: sweep ONE YAML-chosen knob  (multi-GPU)
bakeoff_activation_*.py                CLI: one curve per activation    (multi-GPU)
example_yamls/                         template YAMLs; copy one into a project's --fileroot
```

The driver scripts sit beside `emulator/` (no `driver/` subfolder): launching one
puts its own folder on `sys.path`, so `import emulator` resolves with no path
setup. In a cocoa install this folder is
`external_modules/code/emulators/emultrf/dev/`; run the drivers from `$ROOTDIR`.

The library is pure PyTorch and reviewable anywhere; only `geometries_output.py`
imports cosmolike, so training runs on the workstation where cosmolike lives.

---

## 2. Pipeline

The goal is to replace an expensive physics code with a network that maps a
handful of cosmological parameters to the cosmic-shear data vector, fast enough
to call inside a cosmological inference and accurate enough that the data
vector's [**chi2**](#7-appendix-the-chi2-metric-mahalanobis) — its distance from
truth measured in the data covariance (a Mahalanobis distance; see the appendix),
the quantity inference actually cares about — stays small. Two ideas run through the
whole pipeline. **Whitening**: both the inputs and the outputs are rotated and
rescaled so the network sees a decorrelated, unit-variance, well-conditioned
problem instead of raw correlated numbers. **The chi2 metric**: training and
evaluation are judged in the covariance's natural geometry, not in raw
per-element error, because that is what an analysis uses.

```
cosmological parameters
   │   geometries_parameter.py   center, rotate, unit-scale          (whiten in)
   ▼
whitened inputs
   │   emulator_designs.py       ResMLP, ResCNN, or ResTRF
   ▼
whitened data vector
   │   geometries_output.py      un-whiten + scatter to full length  (whiten out)
   ▼
physical residual vs truth
   │   loss_functions.py         contract with the inverse covariance
   ▼
chi2  =  r^T Cinv r
```

**1. Stage the data** (`data_staging.py`). The training set is a large dump of
`(parameters, data vector)` pairs the physics code wrote to disk. The dump is far
too big to hold in RAM, so it is memmapped and read in slices; physical cuts
drop the sparse corners no real posterior visits (the high-`omega_b h^2`
corner, and both tails of the `omegam^2 h^2` window when configured),
and only `N_train` rows are kept. The result is a "source" dict (`C`, `dv`,
`idx`) the rest of the pipeline consumes.

```
1.  Stage the data                            load_source · data_staging.py

      dv dump (.npy)                    params table (.txt)
          │                                  │
   np.load(mmap_mode=r)              loadtxt[:, 2:-1]
   never loaded whole                drop weight / lnp / chi2 cols
          │                                  │
          └────────────────┬─────────────────┘
                           ▼
                 seeded shuffle             randperm(n, gen)   (split_seed)
                           ▼
                 physical cuts              phys_cut_idx: keep lo < omega_b h^2 < cut
                           ▼                 + lo < omegam^2 h^2 < hi (lower bounds
                           │                 + window optional); omegab/omegam/H0
                           │                 columns found by name
                 keep N_train               idx = phys[:n_keep  or  N // divisor]
                           ▼
            stage_source:  subset bytes < ram_frac · available RAM ?
                 ┌─────────┴──────────┐
                yes                    no
                 ▼                     ▼
       ┌────────────────────┐  ┌────────────────────┐
       │ materialize the    │  │ keep the memmap    │
       │ compact subset,    │  │ + global idx       │
       │ reindex → arange   │  │ (stream from disk) │
       └────────────────────┘  └────────────────────┘
                 └─────────┬──────────┘
                           ▼
       source dict  { C, dv, idx  (+ C_mean, dv_mean — train only) }
                    ──────────────────────────────────────────────▶  build_loaders
```

The local `arange` reindex is the trick: every consumer reads `C` / `dv` only
through `idx`, so it does not matter whether `idx` points into the full memmap or
the compact in-RAM subset — the pipeline is identical either way.

**2. Whiten the inputs** (`geometries_parameter.py`). Raw cosmological parameters
are correlated and span wildly different scales. `ParamGeometry` centers them,
rotates into the parameter-covariance eigenbasis, and scales each direction to
unit variance, so the network receives decorrelated, unit-variance inputs rather
than strongly correlated physical numbers.

```
2.  Whiten the inputs                          ParamGeometry · geometries_parameter.py

      raw params  θ                            (B, n_param)  physical, correlated
          │
          │  − center                          c = training mean
          ▼
      centered
          │  @ evecs                           rotate into the param-covmat eigenbasis
          ▼                                     evecs, √λ = eigh(covmat)   (from_covmat)
      decorrelated
          │  / √λ                              scale every axis to unit variance
          ▼
      whitened input  X  ────────────────────▶ model sees decorrelated, σ = 1 inputs

      encode(θ) = (θ − c) @ evecs / √λ          decode = (X · √λ) @ evecsᵀ + c  (exact inverse)
```

**3. Whiten the output, and keep the metric** (`geometries_output.py`). The data
vector is *masked* (the analysis keeps only some entries) and strongly
correlated. `DataVectorGeometry` *squeezes* to the unmasked entries and whitens
them in the data-covariance eigenbasis, so every network output is decorrelated
and equally hard to fit. The same object holds `Cinv`, the masked inverse
covariance the chi2 contracts against — geometry and metric live together.

```
3.  Whiten the output, keep the metric         DataVectorGeometry · geometries_output.py

      raw data vector  d                        (B, total_size)  full 3x2pt
          │
          │  squeeze  d[:, dest_idx]            keep only the unmasked entries → (B, n_keep)
          ▼
          │  − center                           c = training mean of the kept entries
          ▼
          │  @ evecs / √λ                       whiten in the kept-block cov eigenbasis
          ▼                                      evecs, √λ = eigh(kept cov)
      whitened target  t  ◀──────────────────── network predicts  ŷ ≈ t
                                                 decorrelated, σ = 1 → every output equally hard

      ── the loss un-whitens, and scores the true metric ──────────────────────
          ŷ − t  ──unwhiten──▶  r               r = physical residual  (· √λ) @ evecsᵀ
                                    │
                χ² = rᵀ · Cinv · r  ◀┘            full masked precision (geom.Cinv_sq)
                                                 for full whitening this equals ‖ŷ − t‖²
                                                 (the whitening basis is the χ² basis)
```

**4. Build the loss** (`loss_functions.py`). `make_chi2` wraps the output
geometry in a chi2. Because the targets are whitened, plain squared error in the
whitened space *is* the chi2, so the optimization is well-conditioned; the loss
un-whitens the residual and contracts it with `Cinv` to report the true chi2, and
adds optional robustness (trim the worst points, up-weight the still-hard ones).

```
4.  Build the loss                            make_chi2 · loss_functions.py

      cosmolike cov / mask / inv-cov
                │  DataVectorGeometry.from_cosmolike   (one cosmolike read + one eigh)
                ▼
         geom   (built once)        owns squeeze · center · whiten · decode · Cinv
                │
                │  wrap  (composition: the loss HAS-A geom — self.geom, not inheritance)
                ▼
       rescale = ?
      ┌──────────────────┬─────────────────────────┬──────────────────────────┐
    "none"            "rescaled" (A)             "residual" (B)
      ▼                  ▼                          ▼
┌──────────────┐  ┌────────────────────┐    ┌────────────────────┐
│ CosmolikeChi2│  │ RescaledChi2       │    │ ResidualBaseChi2   │
│ plain masked │  │ R divides ŷ →      │    │ R moves the        │
│ Mahalanobis  │  │ diag(1/R) in grad  │    │ baseline; plain χ² │
└──────────────┘  └────────────────────┘    └────────────────────┘
 needs_params      needs_params = True          needs_params = True
  unset → False    + build_shear_angle_map(geom) + configure_rescaling(...)
                │
                ▼
        chi2fn  ─────────────────────────────────▶  run_emulator
        forwards encode / decode / dest_idx / total_size → self.geom
        pipeline branches on getattr(chi2fn, "needs_params", False)   (not isinstance)
```

The two ideas the diagram is built around: **composition** (one `geom` built once,
wrapped by whichever loss — never re-read, never inherited) and the
**`needs_params` capability flag** (a future param-aware loss just sets the flag;
nothing branches on `isinstance`).

**5. Choose the model** (`emulator_designs.py`, `IA/emulator_designs.py`).
`ResMLP` is the baseline: an input projection, a stack of residual blocks, an
output projection. `ResCNN` adds a 1D-CNN correction on top of the ResMLP trunk,
acting in *theta order* so a convolution can exploit smoothness along the angular
axis. Two orthogonal YAML keys pick the class: `train_args.model.name` is the
architecture (`resmlp` | `rescnn` | `restrf`), and the separate `train_args.model.ia` key
layers a factored intrinsic-alignment design on it (omit for the plain
emulator; `ia: nla` (1 amplitude, 3 templates) or `ia: tatt` (3 amplitudes,
10 templates) makes the model emit templates the loss combines in closed
form, so the IA amplitudes never enter the network — `TemplateMLP` for
`resmlp`, `TemplateResCNN` for `rescnn`, whose gated conv corrects each
template before the combine).

**6. Feed the GPU** (`batching.py`). The staged data may or may not fit in GPU
memory, so the loaders pick a regime — hold the whole encoded set resident on the
GPU if it fits, otherwise stream it from RAM, or from the disk memmap, a chunk at
a time — and hand the training loop two closures (`load_C`, `load_dv`) that hide
which regime is in play, so the loop code is identical no matter the data size.

```
6.  Feed the GPU                              _build_loaders_one · batching.py

     one source's data vectors                budget = free VRAM (CUDA) | GPU_MEM (else)
               │
               │  encoded set fits?    enc_dvs + resident < 0.8 · budget
       ┌───────┴───────────────────────────────────┐
      yes                                           no
       │                                    dv still an in-RAM array?
       │                                ┌───────────┴───────────┐
       │                               yes                      no (np.memmap)
       ▼                                ▼                        ▼
 ┌───────────────┐             ┌──────────────────┐    ┌──────────────────┐
 │ Regime 1      │             │ Regime 2         │    │ Regime 3         │
 │ resident GPU  │             │ RAM → GPU        │    │ disk → GPU       │
 │ pre-encode    │             │ stream a chunk,  │    │ stream a chunk,  │
 │ once; batch = │             │ encode on the fly│    │ encode on the fly│
 │ on-GPU index  │             │ (pinned on CUDA) │    │ (memmap read)    │
 └───────────────┘             └──────────────────┘    └──────────────────┘
  no transfer or                re-stream the subset once per epoch, in VRAM-sized
  re-encode, ever               chunks (load = bs · batches_per_load) — per chunk,
                                not per minibatch
```

**7. Train** (`training.py`). `run_emulator` builds the model, optimizer, and
scheduler from spec dicts, runs the per-epoch loop (annealed robustness, a
validation pass each epoch, keeping the best epoch by `f(dchi2 > 0.2)`), and
returns the histories.

`experiment.py` (`EmulatorExperiment`) ties steps 1–7 into one object, so each
driver is a thin wrapper that varies one knob:

```
                       EmulatorExperiment   (steps 1–7)
                                │
     ┌────────────┬─────────────┼───────────────┬──────────────────┐
     ▼            ▼             ▼               ▼                  ▼
train_single  tune_single  sweep_ntrain  sweep_hyperparam  bakeoff_activation
  one run    Optuna search f(dchi2) vs N  one YAML knob      one curve per act
             (multi-GPU,   (multi-GPU,    (multi-GPU,        (multi-GPU, by act)
              journal)      LPT, --gpu-    even split,
                            pack)          --gpu-pack)
```

---

## 3. What each file does

**Data & geometry**

| File | Role |
|---|---|
| `data_staging.py` | On-disk dumps → in-memory "source" dicts; streaming per-column stats; the physical cuts (`omega_b h^2` bound, optional `omegam^2 h^2` window). Memmaps the dv dump (never loads it whole). |
| `geometries_parameter.py` | Input whitening: `ParamGeometry` (center + rotate into the covmat eigenbasis + unit-scale), `LogParamGeometry`, and the IA-factoring `NLAInputGeometry` / `AmplitudeFactorGeometry`. |
| `geometries_output.py` | Output side: `DataVectorGeometry` (squeeze to unmasked entries, whiten, own the chi2 `Cinv`), `DiagonalGeometry` (theta order, for a CNN), `BlockDiagonalGeometry`, `build_shear_angle_map`. **Only file importing cosmolike.** |
| `analytics.py` | Closed-form analytic xi (Eisenstein-Hu) to divide out broadband cosmology dependence — the optional rescaling `R`. |

**Model**

| File | Role |
|---|---|
| `activations.py` | Learnable activations: the paper's `H` plus Power / Gated / GatedPower variants; `make_activation` maps a name → factory. |
| `emulator_designs_building_blocks.py` | The small `nn.Module`s models are built from: `Affine`, `ResBlock`, `CNNBlock`. |
| `emulator_designs.py` | The full networks: `ResMLP` (baseline) and `ResCNN` (ResMLP trunk + a gated 1D-CNN correction in theta order). |

**Loss & training**

| File | Role |
|---|---|
| `loss_functions.py` | chi2 losses on the whitened residual: `CosmolikeChi2` (plain), `RescaledChi2` / `ResidualBaseChi2` (analytic-R), `ElementWeightedChi2`; `anneal_value` (trim/focus schedule); `make_chi2`. |
| `batching.py` | Memory sizing + the regime-aware loaders (GPU-resident / RAM-stream / memmap-stream) that feed the training loop. |
| `training.py` | Device pick, the `make_model/optimizer/scheduler` factories, `build_run_specs`, the `[default, min, max, kind]` search resolvers, the per-epoch loop, and `run_emulator`. |

**Orchestration & output**

| File | Role |
|---|---|
| `experiment.py` | `EmulatorExperiment`: config → device → data → geometry → chi2 → spec → train as one reusable object (`from_yaml` / `from_config`). The drivers compose it. |
| `scheduling.py` | GPU job balancing (`lpt_assign` by cost, `even_assign` round-robin), the spawned worker pool (`run_gpu_pool`: one process per GPU lane, per-GPU job queues), and the `--gpu-pack` VRAM-token machinery (`estimate_train_vram_fraction`, `vram_tokens`). |
| `results.py` | `save_learning_curves` / `save_sweep_table`: `np.loadtxt`-friendly plain-text tables. `save_emulator`: a trained run as `.emul` (weights, cpu state_dict) + `.h5` (whitening geometries, histories, config). |
| `plotting.py` | Training history, learning-curve overlays, coverage panels, xi curves. |
| `diagnostics.py` | Post-training analyses: coverage (kNN distance vs error), the local-linear data floor, the hard-direction regression. |

**Drivers** (beside `emulator/`; each reads `--root` / `--fileroot` / `--yaml`)

| File | Role |
|---|---|
| `train_single_emulator_cosmic_shear.py` | One training run; `--diagnostic` writes a multipage PDF. |
| `tune_single_emulator_cosmic_shear.py` | Optuna study over the YAML's `[default, min, max, kind]` ranges; multi-GPU via a shared journal-file study (`--n-gpus`, `--journal`). |
| `sweep_ntrain_emulator_cosmic_shear.py` | `f(dchi2 > thr)` vs `N_train`; multi-GPU, LPT-balanced; `--gpu-pack` co-locates small points on big cards. |
| `sweep_hyperparam_emulator_cosmic_shear.py` | Sweep ONE hyperparameter chosen in the YAML `sweep:` block (any dotted `train_args` path, e.g. `bs`, `lr.lr_base`, `model.cnn.film`, `model.activation`); multi-GPU, `--gpu-pack`. |
| `bakeoff_activation_emulator_cosmic_shear.py` | One learning curve per activation; multi-GPU, split by activation. |

---

## 4. Change X → edit Y

| To change… | Edit |
|---|---|
| a model architecture | `emulator_designs.py` (+ `emulator_designs_building_blocks.py`) |
| an activation function | `activations.py` (register it in `make_activation`) |
| the loss / add a chi2 variant | `loss_functions.py` |
| how parameters are whitened (input) | `geometries_parameter.py` |
| dv whitening / cosmolike reading (output) | `geometries_output.py` |
| data loading / the physical cut / staging | `data_staging.py` |
| the GPU-memory regime / batching | `batching.py` |
| the optimizer/scheduler build or the training loop | `training.py` |
| the end-to-end setup wiring | `experiment.py` |
| a CLI driver (add/modify) | `*_emulator_cosmic_shear.py` (beside `emulator/`; compose `EmulatorExperiment`) |
| which hyperparameters are searched | the driver YAML (`[default, min, max, kind]`) + resolvers in `training.py` |
| multi-GPU balancing | `scheduling.py` |
| the output file format | `results.py` |
| a plot | `plotting.py` |
| a diagnostic | `diagnostics.py` |
| analytic preprocessing | `analytics.py` |

---

## 5. Variants

Each subfolder mirrors the two-file shape (`emulator_designs.py` +
`loss_functions.py`) and is an experiment, not the main path.

| Folder | What it is |
|---|---|
| `parallel/` | The per-bin CNN head: a grouped conv refines each tomographic bin independently on a shared ResMLP trunk. (The per-bin ResMLP split was tested, lost to a single ResMLP, and was removed.) |
| `PCE/` | NPCE: a sparse-Legendre polynomial-chaos base plus a neural refiner. |
| `IA/` | Factored intrinsic alignment: emulate cosmology-only templates and apply the IA-amplitude polynomial in closed form (the amplitudes never enter the network). |

---

## 6. Run it

cosmolike runs only on the workstation, so train there.

```bash
# run from $ROOTDIR (cocoa exports it). --root = project folder under $ROOTDIR;
# --fileroot = a subfolder of it holding this emulator's YAML + outputs; --yaml =
# a bare filename under --fileroot. Data (dv/params/covmat) lives in --root/chains.
D=external_modules/code/emulators/emultrf/dev

# one run
python $D/train_single_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --diagnostic diagnostic

# N_train learning curve across all GPUs
python $D/sweep_ntrain_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --n-points 8 --out curve

# one-knob sweep (the knob + values live in the YAML's sweep: block)
python $D/sweep_hyperparam_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --out lrsweep

# Optuna search across all GPUs (one shared study via a journal file)
python $D/tune_single_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml tune_single_emulator_cosmic_shear.yaml --n-trials 64

# activation bake-off across GPUs
python $D/bakeoff_activation_emulator_cosmic_shear.py \
  --root projects/lsst_y1/ --fileroot emulators/training_scripts/ \
  --yaml train_single_emulator_cosmic_shear.yaml --out bakeoff
```

On a card with far more memory than one training needs (an H200), add
`--gpu-pack` to either sweep: points estimated at ≤ 20% of the GPU run four
to a card, ≤ 40% two to a card, bigger ones exclusive (off by default — on a
12 GB RTX 3060 one training is the card). The details live in
[Multi-GPU execution and packing](#6a-multi-gpu) below.

The YAML has two blocks: `data` (bare input filenames resolved under
`--root/chains`, the cut/split, the cosmolike dataset) and `train_args` (`nepochs`, `bs`, `loss_mode`, the `model` /
`optimizer` / `lr` / `scheduler` / `trim` / `focus` sub-blocks, the two-phase
schedule — `trunk_epochs` plus the symmetric `trunk:` / `head:` per-phase
override blocks — and the stability guards `clip` (per-step gradient-norm
ceiling) and `rewind` (reload the best weights + optimizer on every plateau
lr cut)). Pick the model
with `train_args.model.name` (the architecture, `resmlp` | `rescnn` | `restrf`) plus the
optional `train_args.model.ia` key (the factored IA design, `nla` | `tatt`;
omit for plain). The same YAML drives both
`train_single` and `tune_single` — a scalar trains, a `[default, min, max, kind]`
list is searched. Templates live in `example_yamls/` — one per driver style:
`train_single_…` (fully documented train_args), `tune_single_…` (search
ranges), `sweep_hyperparam_…` (the `sweep:` block, with common sweeps ready
to swap in). Copy one into your `--fileroot` and edit it.

### The `sweep:` block (one-knob sweeps) <a name="6a-sweep-block"></a>

`sweep_hyperparam_emulator_cosmic_shear.py` reads one extra top-level YAML
block (the other drivers ignore it) naming exactly one `train_args` leaf by
its dotted path, and the values to try — one full training per value at
fixed `N_train`:

```yaml
sweep:
  parameter: lr.lr_base
  values:
    - 0.0010
    - 0.0025
    - 0.0063
```

| Rule | Why |
|---|---|
| any `train_args` leaf sweeps by dotted path (`bs`, `trim.start`, `model.cnn.kernel_size`, `model.cnn.film`, `head.lr_base`, …) | the sweep deep-copies `train_args` and sets that one leaf per point |
| `model.activation` (or `.type`) is a special case | the activation family is resolved onto the experiment at build, not read from `train_args`; the driver sets it per value — leave `--activation` unset |
| `model.name` / `model.ia` are refused | they change the model *class*; run one sweep per architecture and overlay the tables |
| an unknown first segment is refused | a typo'd path would otherwise silently train the same config N times |
| a missing intermediate block is created (`head.lr_base` with no `head:` block) | the usual guards still fire (`trunk`/`head` overrides need `trunk_epochs > 0`) |

Outputs under `--fileroot`: `<--out>.txt` (`save_sweep_table`: numeric values
as a value/frac table; categorical or boolean values as an index/frac table
with a `# values: 0=…, 1=…` label line — `np.loadtxt` reads either) and
`<--out>.pdf` (`plot_sweep_curve`). The full template is
`example_yamls/sweep_hyperparam_emulator_cosmic_shear.yaml`, with the common
sweeps (bs, activation family, film on/off, conv depth, head lr) ready to
swap in.

### Multi-GPU execution and packing <a name="6a-multi-gpu"></a>

Every hyperparameter driver runs on all visible CUDA devices by default
(`--n-gpus` caps it; one GPU or Apple MPS falls back to a serial loop). Jobs
never split across GPUs — each training fits one card, so the parallelism is
one whole training per worker process (spawn, so each child owns its CUDA
context and its own cosmolike state):

| Driver | Jobs | Split across GPUs | Extra flags |
|---|---|---|---|
| `sweep_ntrain` | one training per `N_train` | LPT (cost ∝ N: biggest first to the least-loaded GPU) | `--gpu-pack` |
| `sweep_hyperparam` | one training per value | round-robin (equal cost) | `--gpu-pack` |
| `bakeoff_activation` | one learning curve per activation | by activation | |
| `tune_single` | Optuna trials | one worker per GPU, one shared study | `--journal` |

**`--gpu-pack` (both sweep drivers; off by default).** Co-locates several
trainings on one GPU when they are small. Each GPU is modeled as 4 capacity
tokens; a point's tokens come from a conservative VRAM estimate
(`2 · N · dv_width` float32 — the resident targets plus the pre-shuffle
transient — plus a 2 GiB fixed overhead for the CUDA context, model, `Cinv`,
and compile workspaces):

```
    estimated share of the card        tokens    concurrency
    ≤ 20%                                1       up to 4 per GPU
    20% – 40%                            2       up to 2 per GPU
    > 40%                                4       exclusive
```

A per-GPU lock serializes token grabs (no multi-token deadlock); the flag
engages even with a single visible GPU (a lone H200 allocation runs up to 4
small points at once). Why it works: a small training is launch-bound — the
CPU dispatch, not the GPU, sets the epoch time — so co-located processes
time-slice into each other's idle gaps. When to use it: large cards (H200)
with small-to-mid `N_train` points. When not to: small cards (the 2 GiB
overhead alone is ~17% of a 3060 — though the estimate then marks points
exclusive anyway, so the flag degrades to plain one-per-GPU), and any run
whose per-epoch timings you want to quote — co-located points contend and
are not comparable to exclusive runs. If a point outgrows its estimate, the
loaders degrade to streaming against the GPU's real free memory rather than
crash. On NVWULF, enabling CUDA MPS (`nvidia-cuda-mps-control`) tightens the
time-slicing further; the flag works without it.

**Parallel Optuna (`tune_single --n-gpus N`).** The workers cooperate on ONE
study through an Optuna journal file (`--journal`, default
`tune_journal.log` under `--fileroot`): the parent creates the study and
enqueues the YAML-defaults warm-start once, `--n-trials` is the total split
across workers, and each worker proposes with its own sampler seed (a shared
seed would duplicate proposals). The journal persists — rerunning with the
same `--journal` RESUMES the study (the recorded trials inform the new
ones); delete the file or pass a new name to start fresh. The serial path
(1 GPU / MPS) stays in-memory and writes nothing.

---

## 7. Appendix: AI-Usage

**AI Usage**: This library (under the `dev` folder) was developed with Claude Code assistance. However, Prof. Miranda heavily influenced 
the code at every level, from macro-designed implementation and changes to minute Python choices.
The initial version of this library was developed in a [Jupyter Notebook](https://github.com/vivianmiranda/learningpytorch/blob/main/pytorch1.ipynb) in which the AI didn't have writing privileges. Prof. Miranda spent hundreds of hours (it will reach 1000 hours for sure by the time this pipeline reaches v1.0) working on this pipeline and studying the Book Deep Learning with PyTorch to make sure she was not `vibe coding` (which Prof. Miranda strongly opposes).  Claude AI was a fast pair of hands and a super-smart compiler. There were a few exceptions where AI took the driving seat, mostly related to some obscure PyTorch functions and multi-gpu scheduling. Even in these cases, Prof. Miranda studied and edited every corner of the code, and also tested it in numerous ways (and it is continuing to test it). Prof. Miranda is ready to go on the blackboard and discuss every choice, every function, every quirk of the code at any time. 

When I asked Claude code to review the paragraph above, the AI answer was (this repository has a copy of the Claude memory files)

<img width="1000" alt="Screenshot 2026-06-30 at 7 12 00 PM" src="https://github.com/user-attachments/assets/e9a59983-cbbc-4a21-b962-9ee4ece4df7f" />

---

## 8. Appendix: the chi2 metric (Mahalanobis)

The loss and the reported metric are both a **chi2**, which is a squared
**Mahalanobis distance** — the distance between two points measured *in units of
the data's own spread and correlations*, not in raw coordinate units. For a
residual `r = pred − truth`, the squared Mahalanobis distance is

```
   d²  =  rᵀ · C⁻¹ · r          C = data covariance,  C⁻¹ = precision (Cinv)
```

and that *is* the chi2. The contrast with plain distance is the whole point:

| | formula | what it does |
|---|---|---|
| plain Euclidean | `rᵀ r = Σ rᵢ²` | every entry counts equally |
| **Mahalanobis** | `rᵀ C⁻¹ r` | divide each direction by its variance, remove correlations → "how many **σ** off", not "how many raw units off" |

Two limits make it concrete:

- **Diagonal `C`** (variances σᵢ², no correlations): `d² = Σ (rᵢ / σᵢ)²` — just
  z-scores squared. A 1-unit error on a tight bin (small σ) costs far more than on
  a loose bin.
- **`C = I`**: collapses back to plain Euclidean.

The tie to this codebase: **whitening is exactly the coordinate change that turns
Mahalanobis into Euclidean.** In the whitened basis the covariance becomes the
identity, so

```
   rᵀ · C⁻¹ · r   =   ‖ whiten(r) ‖²
```

That is the "whiten the output, keep the metric" line from step 3: the network
trains on a clean `‖·‖²` in the whitened basis, while the loss still scores the
true correlated metric — it un-whitens the residual and contracts the masked
`Cinv`. For the standard full whitening the two are equal to rounding; the
diagonal-whitening variant (`DiagonalGeometry`, used for the CNN) breaks that
equality, which is why the loss always keeps the explicit `Cinv` contraction
rather than collapsing to a mean squared error.

**Mnemonic:** Mahalanobis = Euclidean distance *after whitening* (distance in σ
units, with correlations removed).

---

## 9. Appendix: activation functions

The `ResBlock` nonlinearity is a **learnable, per-feature activation**: every
feature (one entry of the vector) carries its own shape parameters, trained with
the network. The default is the paper's $H(x)$; three generalizations are
available, selected by name (`--activation`, or `make_activation` in
[`activations.py`](emulator/activations.py)). Throughout, $\odot$ is the
elementwise (Hadamard) product, $\sigma(z) = 1/(1 + e^{-z})$ is the logistic
sigmoid, and each Greek symbol is a length-`dim` vector (one value per feature).

### The paper's $H(x)$

From eq. (6) of [arXiv:2505.22574](https://arxiv.org/pdf/2505.22574):

$$H(x) = \Big( \gamma + (1 + e^{-\beta \odot x})^{-1} \odot (1 - \gamma) \Big) \odot x$$

Because $(1 + e^{-\beta x})^{-1} = \sigma(\beta x)$, this is a per-feature
**interpolation between the identity and a Swish gate**:

$$H(x) = \gamma \odot x + (1 - \gamma) \odot \mathrm{Swish}_\beta(x), \qquad \mathrm{Swish}_\beta(x) = x \sigma(\beta x).$$

The gate $\gamma + (1-\gamma) \sigma(\beta x)$ runs from $\gamma$ (as
$x \to -\infty$) to $1$ (as $x \to +\infty$), so $H$ is **asymptotically linear
on both tails** — slope $\gamma$ on the left, $1$ on the right. That
non-saturation is why it beats $\tanh$ (whose slope vanishes) for these
emulators. $\gamma$ sets the left-tail slope (the linear-vs-nonlinear mix),
$\beta$ the kink sharpness near $x = 0$. At init $\gamma = \beta = 0$, so $H$
starts as $0.5 x$ (since $\sigma(0) = 0.5$) and training shapes each feature's
curve. Two learnable vectors per feature ($\gamma$, $\beta$).

### Generalizations

Each is a strict superset of $H$ and recovers it at initialization. They give the
activation more freedom where the target is hard; $H$ stays the default.

**Multi-gate** (`multigate`, `GatedActivation`) — replace the single Swish gate
with a sum of $K$ sigmoids, a learnable slope-vs-$x$ schedule in the bulk:

$$\mathrm{gate}(x) = a_0 + \sum_{k=1}^{K} w_k \sigma\big(\beta_k (x - \mu_k)\big), \qquad \mathrm{out} = \mathrm{gate}(x) \odot x.$$

Every term is a bounded sigmoid times $x$, so the output stays asymptotically
linear (slope $a_0$ to the left, $a_0 + \sum_k w_k$ to the right). $H$ is the
$K = 1$ case ($a_0 = \gamma$, $w_1 = 1 - \gamma$, $\mu_1 = 0$); the general form
also frees the right-tail slope and the kink center $\mu$. $3K + 1$ vectors per
feature.

**Bounded power tail** (`power`, `PowerGatedActivation`) — keep $H$'s gate but
apply it to a signed power transform $\psi_p$, linear near $0$ and $\sim |x|^p$
in the tail, with $p$ learnable and boxed into $[p_{\min}, p_{\max}]$ (default
$[0.5, 1.5]$, between $\sqrt{x}$ and $x^{1.5}$):

$$\psi_p(x) = \mathrm{sign}(x) \frac{(1 + |x|)^p - 1}{p}, \qquad p = p_{\min} + (p_{\max} - p_{\min}) \sigma(\rho),$$

$$\mathrm{out} = \big(\gamma + (1 - \gamma) \sigma(\beta x)\big) \odot \psi_p(x).$$

The $/p$ normalization gives $\psi_p$ slope $1$ at $x = 0$ for **any** $p$, so
$p$ reshapes only the tail; the base $1 + |x| \ge 1$ keeps any real $p$ finite
(no `NaN`), and the sigmoid box prevents a runaway exponent — safe
superlinearity, unlike a raw $x^n$. $p = 1$ (at $\rho = 0$) recovers $H$. Three
vectors per feature ($\gamma$, $\beta$, $\rho$).

**Both** (`gated_power`, `GatedPowerActivation`) — the multi-gate bulk times the
bounded power tail:

$$\mathrm{out} = \Big(a_0 + \sum_{k=1}^{K} w_k \sigma\big(\beta_k(x - \mu_k)\big)\Big) \odot \psi_p(x).$$

$3K + 2$ vectors per feature.

### Selecting one

| `--activation` | class | adds over `H` | params / feature |
|---|---|---|---|
| `H` (default) | `activation_fcn` | — (the paper's gate) | 2 |
| `multigate` | `GatedActivation` | K-sigmoid bulk slope schedule | 3K + 1 |
| `power` | `PowerGatedActivation` | bounded learnable tail exponent | 3 |
| `gated_power` | `GatedPowerActivation` | both of the above | 3K + 2 |

$K$ (the gate count for the multi-gate families) is `make_activation`'s
`n_gates`, default 3.

---

## 10. Appendix: every file's functions

One line per function / class / method. For full detail, read the docstring in
the file itself; this is the index.

### `emulator/data_staging.py` <a name="apx-data_staging"></a>

Turns on-disk dumps into in-memory "source" dicts.

- `load_source(...)` — orchestrator: memmap the dv, load + cut the params, keep `N_train` rows, stage, return `{C, dv, idx (+ means)}`.
- `stage_source(C, dv, idx, ram_frac)` — materialize the used rows in RAM if they fit, else keep the memmap (reindex local).
- `phys_cut_idx(C, idx, names, cut, omegabh2_lo, omegam2h2_lo, omegam2h2_hi)` — keep the rows with `omega_b h^2` in `(omegabh2_lo, cut)` and (optionally) `lo < omegam^2 h^2 < hi`.
- `stream_chunks(idx, chunk)` — yield sorted row-index blocks (sequential disk reads).
- `stream_stats(mm, idx, method, CHUNK)` — per-column mean/std (or min/max) over the used rows, streamed (never loads the dump whole).
- `param_stats(arr, idx, method)` — the same stats for the in-RAM parameter array.
- `read_param_names(covmat_path, comment)` — parameter names from the covmat header line.

### `emulator/geometries_parameter.py` <a name="apx-geometries_parameter"></a>

Input side: raw parameters → whitened network input.

- `ParamGeometry` — center, rotate into the parameter-covariance eigenbasis, unit-scale.
  - `from_covmat` / `from_state` / `state` — build from a covmat file / saved tensors; tensors to save.
  - `whiten` / `unwhiten`, `encode` / `decode` — the transform and its exact inverse.
- `LogParamGeometry` — `ParamGeometry` that whitens in log space for the multiplicative params (`from_samples`, `_to_t` / `_from_t`).
- `NLAInputGeometry` — whiten all but the IA amplitude `A1_1`, append it raw (factored NLA).
- `AmplitudeFactorGeometry` — same, generalized to any number of IA amplitudes (TATT).

### `emulator/geometries_output.py` <a name="apx-geometries_output"></a>

Output side: raw dv ↔ whitened masked target; holds the chi2 covariance. The only file importing cosmolike.

- `DataVectorGeometry` — the base geometry for one probe.
  - `from_cosmolike` / `from_state` / `state` — build from cosmolike / saved tensors; tensors to save.
  - `squeeze` / `unsqueeze` — keep the unmasked entries / scatter them back to full length.
  - `whiten` / `unwhiten`, `encode` / `decode` — covariance-eigenbasis whitening and its inverse.
- `DiagonalGeometry` — whiten by the marginal sigma only (theta order kept, for a CNN).
- `BlockDiagonalGeometry` — whiten each tomographic bin by its own sub-block.
- `build_shear_angle_map(geom, ...)` — attach per-element theta / source-z / xi± branch / per-bin sizes.

### `emulator/analytics.py` <a name="apx-analytics"></a>

Analytic xi rescaling `R` (Eisenstein-Hu zero-baryon preprocessor).

- `_analytic_R(...)` — the formula (numpy or torch); divides out the broadband cosmology dependence.
- `analytic_shape_ratio(...)` — `R` over the masked data vector (the emulator path).
- `rescale_xi(...)` — `R` over the (theta, xi+, xi−) matrix layout (plotting / visual checks).

### `emulator/activations.py` <a name="apx-activations"></a>

Learnable activations for the ResBlock `act` slot.

- `activation_fcn` — the paper's `H` (a learnable identity↔Swish interpolation).
- `GatedActivation` / `PowerGatedActivation` / `GatedPowerActivation` — generalizations (more gates, a bounded power tail, both).
- `make_activation(name, n_gates)` — map a name to a factory `act(dim) -> module`.

### `emulator/emulator_designs_building_blocks.py` <a name="apx-building_blocks"></a>

The small `nn.Module`s the models are assembled from.

- `Affine` — a learnable scalar scale + shift.
- `ResBlock` — width-preserving residual block (n dense layers, each with a norm + activation factory, pre-activation skip).
- `BinLinear` — G per-token *unique* linear layers as one batched einsum; the unique weights also replace the positional encoding.
- `TRFBlock` — one pre-LN transformer block over tokens at their *natural* width (the padded bin length — no embedding/output adapters): shared-weight attention across tokens + a per-token unique MLP stack (the deviation from the textbook shared FFN). Exactly the identity at init (zero-initialized branch outputs), so a stack satisfies `blocks(x) == x`.

### `emulator/emulator_designs.py` <a name="apx-emulator_designs"></a>

The full networks.

- `ResMLP` — input projection → residual blocks → output projection → Affine.
- `ResCNN` — ResMLP trunk + a gated bins-as-channels 1D-CNN correction in theta order (one `Conv1d(n_bins → n_bins, k)` kernel over the padded per-bin layout — theta-local and cross-bin, no channel expansion), via fixed basis-change buffers `W_fd` / `W_df` and the `pad_idx` scatter/gather. Head knobs (YAML `model.cnn`): `kernel_size` (tuned as if one block) + `rescale_kernel` (shrink the per-block kernel with depth at a fixed receptive field), `groups` (physical channel cuts: `2` = xi+ never mixes with xi−; on the factored head `3` = GG/GI/II isolated, `6` = both cuts — validated against the mask, other values error), `separable` (factor each block into a depthwise theta filter + pointwise channel mix — a low-rank factorization of the same conv, ~k/2 fewer weights), `film` (re-inject the non-amplitude parameters into every block as an identity-initialized per-channel affine — the head becomes cosmology-aware instead of one fixed map; see `notes/film-conditioning.md`), `n_blocks`, `gate_init`:

```
  params ─▶ ResMLP trunk ─▶ y    (full-whitened, well-conditioned)
                            │     y @ W_fd   full basis ─▶ theta order
                            ▼
                       1D-CNN blocks         fix theta-local structure
                            │     h @ W_df   theta order ─▶ full basis
                            ▼
              y + gate · correction   ─▶   whitened data vector
```

- `ResTRF` — ResMLP trunk + a gated bin-token transformer correction: the theta-order dv splits into its (xi+/-, source-pair) bins (`pad_idx` scatter/gather to a padded per-bin layout, `bin_sizes` from `build_shear_angle_map`), each bin is one token at its natural width, `TRFBlock`s attend across bins, and the correction is `blocks(h) − h` — zero at epoch 1 because every block starts as the identity. No embedding or output layers (the sequence structure is physical, unlike the published CMB design's latent sequence).

### `emulator/loss_functions.py` <a name="apx-loss_functions"></a>

chi2 losses; each holds a geometry (composition).

- `anneal_value(epoch, opts)` — the per-epoch trim / focus schedule.
- `CosmolikeChi2` — the plain chi2: `chi2` ([Mahalanobis](#7-appendix-the-chi2-metric-mahalanobis) distance), `loss` (trim / focus / sqrt transform), and thin delegation to the held geometry.
- `RescaledChi2` — analytic-R "A" form (R divides the net output); `configure_rescaling`, `_R`, `encode` / `decode` / `chi2` / `loss`.
- `ResidualBaseChi2` — analytic-R "B" form (R moves only the baseline; the chi2 stays plain).
- `ElementWeightedChi2` — a per-element focal weight in the training loss (`set_elem_weight`).
- `make_chi2(geom, rescale, ...)` — build the right loss from a geometry and a rescale mode.

### `emulator/batching.py` <a name="apx-batching"></a>

Memory sizing and the regime-aware data loaders. Where the data lives:

```
  dv dump (.npy on disk, memmapped)             never loaded whole
        │   load_source  ─▶  the N_train subset
        ▼
  build_loaders picks a regime by what fits the VRAM budget:
        ├─▶ regime 1   resident on the GPU          encode once; a batch is an on-device index
        ├─▶ regime 2   streamed from host RAM        re-encode each chunk, every epoch
        └─▶ regime 3   streamed from the disk memmap  same, read from disk
```

- `compute_batch_size_bytes` / `compute_model_size_bytes` / `batches_per_load` — per-batch and resident memory estimates.
- `_build_loaders_one(...)` — pick a regime for one source (GPU-resident / RAM-stream / memmap-stream); return `load_C`, `load_dv`, the chunk size, and the bytes it made resident.
- `build_loaders(...)` — run it once per source (train, then val against the reduced budget); return the data dict the loop consumes.

### `emulator/training.py` <a name="apx-training"></a>

The run layer that ties everything together.

- `pick_device` / `make_logger` — setup helpers.
- `make_model` / `make_optimizer` / `make_scheduler` — build one component from a `{cls, **kwargs}` spec dict.
- `build_run_specs(...)` — config → the six `run_emulator` spec dicts.
- `default_train_args` / `suggest_train_args` / `search_defaults` (+ `_as_search_range`, `_range_default`, `_suggest_range`, `_walk_train_args`) — the `[default, min, max, kind]` search resolvers.
- `eval_val` / `eval_source_chi2` — score the model on the val set / per-cosmology delta-chi2.
- `training_loop_batched(...)` — the per-epoch loop (trim / focus annealing, best-epoch tracking).
- `run_emulator(...)` — top-level: build model + optimizer + scheduler + loaders, train, return the histories.

### `emulator/experiment.py` <a name="apx-experiment"></a>

`EmulatorExperiment`: the whole setup as one reusable object.

- `from_yaml` / `from_config` — build from a YAML file / an already-parsed dict.
- `stage_train` / `stage_val` / `pool_size` — stage the sources; the physical-cut pool size (the sweep's top N).
- `build_geometry` / `build_specs` — the input/output geometry + chi2; the `run_emulator` spec dicts.
- `train` / `run` — train on the staged data; the full stage→build→train pipeline in one call.
- `frac_above(threshold, ...)` — the sweep metric (fraction of points with delta-chi2 over a cutoff).

### `emulator/scheduling.py` <a name="apx-scheduling"></a>

- `lpt_assign(sizes, n_workers)` — split sweep jobs across GPUs by total cost (Longest-Processing-Time).
- `even_assign(jobs, n_workers)` — round-robin split for equal-cost jobs.
- `run_gpu_pool(setup_fn, job_fn, buckets, extra, lanes_per_gpu, job_tokens, on_result)` — the spawned worker pool: one process per (GPU, lane), per-GPU job queues, token gate under packing, parent-side result drain.
- `estimate_train_vram_fraction(n_rows, dv_width, total_bytes)` — conservative per-training VRAM share (`--gpu-pack`).
- `vram_tokens(fraction)` — the packing rule: ≤20% → 1 token, ≤40% → 2, else 4 (exclusive) of `GPU_TOKENS = 4`.

### `emulator/results.py` <a name="apx-results"></a>

- `save_learning_curves(path, sizes, curves, meta)` — write a `np.loadtxt`-friendly plain-text table.
- `save_sweep_table(path, param, values, fracs, meta)` — the one-knob sweep table (numeric values as a column; categorical as an index + label map).
- `save_emulator(path_root, model, param_geometry, geometry, config, histories, train_args, attrs)` — persist a trained run: `.emul` (cpu state_dict, compile prefix stripped) + `.h5` (geometry `state()` groups, per-epoch histories, config YAML, run-identity attrs).

### `emulator/plotting.py` <a name="apx-plotting"></a>

Figures (colorblind-safe palette, no red/green).

- `plot_history` / `plot_learning_curves` / `plot_diagnostics` — the public figures (training history; learning-curve overlay; the multipage diagnostics PDF).
- `plot_xi` / `dv_to_xi` / `source_param_samples` — xi correlation-function curves, the dv→matrix reshape, and the coverage-triangle samples.
- `_history_panels` / `_coverage_panels` / `_floor_panel` / `_hard_direction_panels` / `_finish` / `_save_pages` — the shared panel and save helpers.

### `emulator/diagnostics.py` <a name="apx-diagnostics"></a>

Post-training analyses (each returns a dict the plotting reads).

- `coverage_diagnostic(...)` — do the failing val points sit in sparse training regions? (kNN distance vs delta-chi2).
- `local_linear_floor(...)` — the model vs a local-linear interpolation of the data (the data-only floor; plain chi2 only).
- `hard_direction_regression(...)` — which log-parameter combination predicts the per-point hardness.

### `emulator/parallel/` <a name="apx-parallel"></a>

The per-bin CNN variant: a shared ResMLP trunk with a grouped conv that
refines each tomographic bin independently (no smoothing across bin-boundary
jumps). Only the conv gets a per-bin twin — the per-bin ResMLP split was
tested, underperformed a single ResMLP at matched parameters (it re-learns the
shared cosmology map once per bin), and was removed.

- `emulator_designs.py` — `ParallelResCNN`.
- `emulator_designs_building_blocks.py` — `GroupedCNNBlock`.

### `emulator/PCE/` <a name="apx-pce"></a>

NPCE: a sparse-Legendre polynomial-chaos base plus a neural refiner.

- `emulator_designs.py` — `PCEEmulator` (the closed-form base) + `pce_multi_index`, `pce_design`, `select_lars_loo`.
- `loss_functions.py` — `PCEResidualChi2` (refine the residual), `PCERatioChi2` (refine the ratio) of a frozen PCE base.

### `emulator/IA/` <a name="apx-ia"></a>

Factored intrinsic alignment: emulate cosmology-only templates, apply the IA-amplitude polynomial in closed form.

- `emulator_designs.py` — `NLATemplateMLP`, `TemplateMLP` (emit the templates).
- `loss_functions.py` — `NLAAmpFactoredChi2`, `TemplateFactoredChi2`, `tatt_coeffs` (apply the amplitude polynomial in the loss).

### drivers (beside `emulator/`) <a name="apx-drivers"></a>

Each `main()` reads `--root` / `--fileroot` / `--yaml`; the sweep / bake-off add
per-GPU workers.

- `train_single_emulator_cosmic_shear.py` — `main`: one training run + the diagnostics PDF.
- `tune_single_emulator_cosmic_shear.py` — `main` + `_tune_worker` + `journal_storage`: an Optuna study over the YAML's search ranges; serial in-memory, or one worker per GPU sharing a journal-file study.
- `sweep_ntrain_emulator_cosmic_shear.py` — `main` + `_sweep_setup` / `_sweep_job` + `_run_parallel` (LPT split through `run_gpu_pool`) / the serial path; `f(dchi2>thr)` vs `N_train`; `--gpu-pack`.
- `sweep_hyperparam_emulator_cosmic_shear.py` — `main` + `set_by_path` / `read_sweep_block` + `_hyper_setup` / `_hyper_job`; one YAML-chosen knob (`sweep:` block), even split through `run_gpu_pool`; `--gpu-pack`.
- `bakeoff_activation_emulator_cosmic_shear.py` — `main` + `_bakeoff_worker` + `_run_parallel_bakeoff` (activation split) / the serial path; one curve per activation.
