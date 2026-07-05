---
name: multi-gpu-sweep-pattern
description: "Single-node multi-GPU PARAMETER-SWEEP methodology (2026-07-01; the sweep_ntrain + bakeoff_activation drivers got it). Each loop iteration is an INDEPENDENT job -> TASK-parallel (one whole training per GPU), NOT data-parallel/DDP (DDP splits ONE training and is the wrong tool; each training fits one GPU). Use PROCESSES not threads: the GIL, cosmolike (ci) holds GLOBAL C state (concurrent build_geometry corrupts it), and EmulatorExperiment caches on self. torch.multiprocessing SPAWN not fork (a forked child can't reuse the parent CUDA context); worker fn must be MODULE-LEVEL, build the exp INSIDE it, args picklable. TWO must-dos so run_emulator's batching ladder sizes right: torch.cuda.set_device(k) per worker (mem_get_info reads the CURRENT device, else every worker reads GPU0) + del refs & torch.cuda.empty_cache() between points (the caching allocator hides freed VRAM). Balance: sweep over N_train uses LPT (lpt_assign in emulator/scheduling.py; plain min-scan, no heapq) since cost~N varies; bakeoff splits by ACTIVATION (equal cost, no LPT; every GPU runs ALL N). Long-pole = the single biggest job. Host RAM: the dv dump is a SHARED memmap, but stage_source's np.asarray(dv[rows]) makes a PRIVATE per-worker copy -> set ram_frac=0 in the parallel path so workers stream the shared memmap (the [[shared-budget-across-sequential-calls]] trap in PARALLEL form). Streaming is per-CHUNK (load=bs*batches_per_load) not per-minibatch. Serial fallback when n_workers<=1 (Mac/MPS); refuse pure CPU. 50M-row wall = np.loadtxt of the param file per worker."
metadata:
  node_type: memory
  type: project
---

When a driver loops over independent runs (`sweep_ntrain` over `N_train`;
`bakeoff_activation` over activation x `N_train`), parallelize it across the
node's GPUs. Built 2026-07-01 for NVWULF (one node, up to 8 H200).

## It is TASK-parallel, not data-parallel

Each loop iteration is an INDEPENDENT job: it reads immutable config + the
read-only dump, writes only a fresh per-iteration `exp`, and returns one number
(`frac_above`). No shared mutable state across iterations. So run DIFFERENT whole
trainings on DIFFERENT GPUs at once (task-parallel) -- do NOT use DDP to split
ONE training across GPUs (data-parallel): each training is small, fits a single
GPU, and DDP would only add gradient-sync overhead. Litmus test + recipe: if you
can write the body as a pure `run_one(job) -> (job, result)` that builds its OWN
`exp` and touches nothing outside it, it is embarrassingly parallel and that
function is the atom of work.

## Processes, not threads (three independent reasons)

1. The GIL: threads do not run the orchestration (staging, cosmolike calls) in
   true parallel.
2. cosmolike (`ci`) has module-GLOBAL C state (`init_probes` / `init_binning` /
   ...). Two threads in one process calling `build_geometry` concurrently stomp
   each other's config and silently produce wrong geometries. Separate processes
   each get their own copy.
3. `EmulatorExperiment` CACHES on `self` (`train_set` / `geom` / `model` are
   overwritten every iteration), so two iterations cannot share one `exp`.

So: one PROCESS per GPU, each building its OWN `exp`.

## spawn, not fork; set_device; empty_cache

- `torch.multiprocessing.get_context("spawn")`. A FORKED child inherits the
  parent's initialized CUDA context, which CUDA forbids using (hang/error). spawn
  = a fresh interpreter + fresh CUDA per child. Linux defaults to fork, so ask
  for spawn explicitly (macOS is already spawn). The worker fn must be
  MODULE-LEVEL (spawn pickles it by qualified name and re-imports the module);
  the args (the cfg dict, scalars, the `ctx.Queue`) must be picklable; build the
  `exp` INSIDE the worker (never pickle it across).
- `torch.cuda.set_device(k)` in each worker. `run_emulator` sizes the GPU budget
  from `torch.cuda.mem_get_info()[0]`, which reads the CURRENT device -- without
  `set_device` every worker reads GPU 0's free VRAM and the regime decision is
  wrong (it can conclude "fits" against card 0 and then OOM card k).
- `del` the previous run's refs (`exp.model = None`, `exp.geom = None`, ...) +
  `torch.cuda.empty_cache()` BETWEEN points. The caching allocator keeps the
  previous run's tensors RESERVED, so `mem_get_info` under-reports free VRAM and
  the next point wrongly streams instead of going resident. `empty_cache` returns
  it. With these two, `_build_loaders_one`'s existing regime ladder (resident /
  RAM-stream / memmap-stream) adapts per point automatically -- no new per-case
  code; the parallel layer only has to FEED it the right numbers.

## Load balancing: LPT for the sweep, activation-split for the bake-off

Cost per point ~ proportional to `N_train` (fixed nepochs / bs).

- `sweep_ntrain`: `N` varies, so balance with LPT (Longest-Processing-Time):
  sort points largest-`N` first, give each to the currently least-loaded GPU.
  `emulator/scheduling.py` `lpt_assign(sizes, n_workers)` -- a plain min-scan over
  per-GPU load totals, NO heapq (the user is a C coder; `heapq` only earns its
  keep at thousands of bins). Naive round-robin imbalances badly (it hands the
  largest of every grid-triple to the same GPU: `N=1..9, 3 GPUs` -> 12/15/18 vs
  LPT's 16/15/14). Static LPT is safe here because the whole grid is known up
  front -- the cost model IS the `N` values.
- `bakeoff_activation`: parallelize on a DIFFERENT axis -- split by ACTIVATION
  (round-robin), not by `N`. Each activation costs ~the same and every GPU runs
  the WHOLE `N` grid, so each GPU does (its activations) x (all `N`) = equal total
  work with NO cost model / LPT. `n_workers` capped at `len(activations)` (4 acts
  + 8 GPUs leaves 4 idle -- the simple scheme's limit; using all 8 would need an
  LPT split of the `(N, act)` cross product). Geometry reuse preserved: `N` outer,
  `build_geometry` once per `N`, activations inner.
- LONG-POLE LIMIT (any scheme): wall-clock >= the single most expensive job (the
  largest `N`, possibly the full pool). One indivisible job sets the floor; the
  only way below it is DDP-splitting that one point (reintroduces data-parallel,
  rarely worth it).

## Host RAM: shared memmap vs the private materialization trap

The dv DUMP is a SHARED memmap -- read-only, one copy in the OS page cache across
all P workers, cheap. But `stage_source`'s `np.asarray(dv[rows])` makes a PRIVATE
per-process COPY of the worker's subset (a single-process optimization, to dodge
memmap random access). With P workers that is P private copies in one host RAM ->
overflow. In the multi-GPU path DON'T materialize: set
`worker_cfg["data"]["ram_frac"] = 0.0` so `stage_source` keeps the shared memmap
and streams the subset from it (it usually lands GPU-resident anyway on a big
card). This is the [[shared-budget-across-sequential-calls]] failure mode in its
PARALLEL form (each of P workers independently thinks it owns `ram_frac` of the
same RAM). The materialization is a FALSE optimization in multi-GPU (P x private
RAM for ~nothing when the subset is GPU-resident, or impossible when it is too
big to copy P times) -- but LEGIT in single-process (it avoids memmap disk
thrashing across hundreds of epochs). The real sin is not optimizing; it is an
UNCONDITIONAL optimization off the measured critical path that spends a shared
resource. Make it context-aware: off in parallel.

## Streaming is per-CHUNK, not per-minibatch (corrected)

The regime-2/3 loaders copy + encode a CHUNK of `load = bs * batches_per_load`
rows (sized to fill the spare VRAM = X minibatches) host->GPU ONCE, then the
inner minibatch loop indexes that now-resident chunk. So the transfer + encode is
amortized over X minibatches. Cost vs regime-1: regime-1 (fits) encodes the whole
subset ONCE total and holds it resident (zero transfer/re-encode across all
epochs); regime-2/3 (does not fit) re-streams the subset ONCE PER EPOCH in
VRAM-sized chunks. Not once per batch.

## Serial fallback + GPU-only

`n_workers = min(--n-gpus or device_count, device_count, len(jobs))`; if `<= 1`
take the plain serial loop (so the Mac/MPS dev box and single-GPU boxes still
run). Build the parent `exp` on the real device and refuse pure CPU
(`exp.device.type == "cpu"` -> `RuntimeError`); it is a GPU tool. The parent may
hold a GPU-0 context and still spawn workers (spawn, not fork, so safe).

## The 50M-row wall (flagged, not fixed)

At research scale (pool up to ~50M dvs) the bigger RAM/time sink than the dv copy
is `load_source`'s `np.loadtxt` of the full PARAMETER text file -- loaded whole,
re-parsed on every `stage_train`, x P workers. There the params want a memmapped
binary like the dv dump. Separate change from the GPU parallelization.

## 2026-07-04y: the pattern grew a shared pool, VRAM packing, and a parallel study

The per-driver spawn code is now ONE machine, `run_gpu_pool` (scheduling.py):
one spawned process per (GPU, lane), per-GPU job queues, a driver-provided
setup_fn (build + stage the experiment once per lane) and job_fn (one point;
must be total -- return nan, never raise), the parent draining results with a
timeout + liveness check so a dead worker raises instead of hanging. Both
sweep drivers (ntrain + the new sweep_hyperparam) run on it; bakeoff still
carries its own older copy of the pattern.

Multi-training-per-GPU (`--gpu-pack`, OFF by default -- the user wants plain
one-per-GPU on amypond's 3060s): each GPU = 4 capacity tokens; a job's tokens
come from a conservative VRAM estimate (2*N*dv_width*4 bytes + 2 GiB
overhead): <=20% of the card -> 1 token (4 co-located), <=40% -> 2, else
exclusive. A lane holds a per-GPU lock while acquiring its tokens (multi-token
deadlock impossible: releases need no lock). Engages even with ONE visible GPU
(a lone H200 allocation). Why it works: small trainings are launch-bound (CPU
dispatch, low SM occupancy), so co-located processes time-slice into each
other's idle gaps; the compiled fwd_loss (fewer, bigger replays) makes
co-location friendlier, and CUDA MPS on NVWULF would tighten it further. An
under-estimate degrades to streaming (loaders size against real mem_get_info);
co-located timings are not comparable to exclusive ones.

PYTHON 3.14 SPAWN TRAP (cost a debugging session; remember it): Process
releases target/args after start(), so a Queue/Lock/Semaphore created in a
loop and referenced ONLY by Process args gets GC'd in the parent -> its named
OS semaphore is unlinked -> a still-booting child dies with FileNotFoundError
in SemLock._rebuild. Keep parent-side references (a keepalive list) to every
sync object until join.

Parallel Optuna (tune_single): one worker per GPU cooperating on ONE study
through a JournalStorage file (append-only, file-locked; the multi-process
storage Optuna recommends -- no database). Parent creates + warm-starts the
study (only when fresh), splits --n-trials, workers load_study with
per-worker sampler seeds (same seed = duplicated proposals). Same journal
file RESUMES; serial path stays in-memory. ram_frac is divided by the worker
count -- every worker stages the SAME subset, so materialization is P private
copies (the shared-budget trap in yet another costume).

**Why:** the full single-node multi-GPU sweep methodology, so the next session
does not re-derive task-vs-data parallel, processes-vs-threads, spawn +
set_device + empty_cache, LPT-vs-activation-split, the long-pole, the
shared-memmap-vs-private-copy RAM trap, the pool/token/packing design, the
py3.14 keepalive trap, or the journal-study pattern. Pairs with
[[emulator-python-package]] (the drivers) and [[py-module-style-conventions]]
(the .py style).
