"""Work-balancing and process-pool helpers for multi-GPU sweeps.

A sweep of many independent jobs (one training run per N_train value,
or per hyperparameter value) must split the jobs so every GPU finishes
at about the same time. lpt_assign partitions jobs of unequal cost by
the Longest-Processing-Time rule; even_assign splits equal-cost jobs
round-robin. run_gpu_pool then executes the buckets: one spawned
process per (GPU, lane), each building its own experiment once and
draining its GPU's job queue -- with optional VRAM-token packing
(estimate_train_vram_fraction + vram_tokens) so several small
trainings can share one large GPU.

The pool's execution model, one GPU shown:

    buckets[g] = [job, job, ...]       from lpt_assign / even_assign
       │  one Queue per GPU, jobs in bucket order (big first
       │  under LPT), one None sentinel per lane
       ▼
    job queue ──┬─ lane 0 process ─┐
                ├─ lane 1 process ─┤  each lane: setup_fn once
                ├─ ...             ├─▶ (build the experiment), then
                └─ lane L-1 proc  ─┘  job_fn per job pulled
       │
       │  gate (packing only): a per-GPU Lock + Semaphore(4); a
       │  lane holds the lock while acquiring its job's tokens, so
       │  token grabs never interleave (no deadlock -- releases by
       │  finishing lanes need no lock and unblock the holder)
       ▼
    result queue -> the parent drains one result per job

    (legend: g = GPU index; lane = one worker process pinned to GPU
     g (lanes_per_gpu of them under packing, else 1); tokens = the
     job's VRAM share in quarters of a GPU, from vram_tokens;
     setup_fn / job_fn = the driver's module-level callables, which
     spawn pickles by qualified name.)

PS: spawn = the multiprocessing start method that launches a fresh
interpreter per child (a forked child cannot reuse the parent's CUDA
context); sentinel = a special queue item (None here) telling a
worker to exit; LPT = Longest-Processing-Time, biggest job first to
the least-loaded worker; token = one quarter of a GPU's capacity in
the packing model.
"""

import queue as queue_mod

import torch


def lpt_assign(sizes, n_workers):
  """
  Balance the sweep points across GPUs by total N_train.

  Longest-Processing-Time rule: hand the points out largest-N first, each to
  the GPU with the least work so far. A point's cost is about proportional to
  its N_train (nepochs and bs are fixed across points), so even per-GPU sums
  of N keep the wall-clock even. Going big-first balances those sums; a naive
  round-robin would pile every grid triple's largest point onto one GPU.

  Arguments:
    sizes     = the N_train values of the sweep (any order; cast to int).
    n_workers = number of GPUs to split across (>= 1).

  Returns:
    buckets = a list of length n_workers; buckets[k] is the list of N_train
              values assigned to GPU k, in the largest-first order they were
              handed out.
  """
  # buckets[k] = N values assigned to GPU k; one empty list per GPU.
  buckets = []
  for _ in range(n_workers):
    buckets.append([])
  # loads[k] = running sum of N given to GPU k (its "load"), starting at 0.
  loads = []
  for _ in range(n_workers):
    loads.append(0)

  # points largest N first (int() because the grid comes from numpy).
  points = []
  for N in sizes:
    points.append(int(N))
  points.sort(reverse=True)

  for N in points:
    # least-loaded GPU by a plain scan: assume GPU 0, then keep any later
    # GPU carrying less work.
    k = 0
    for g in range(1, n_workers):
      if loads[g] < loads[k]:
        k = g
    # assign the point to GPU k and add its cost (N) to that load.
    buckets[k].append(N)
    loads[k] += N

  return buckets


def even_assign(jobs, n_workers):
  """
  Round-robin split for equal-cost jobs.

  Job j goes to bucket j mod n_workers, preserving order within a
  bucket. Use it when every job costs about the same (one training
  per hyperparameter value at a fixed N_train); lpt_assign is the
  cost-aware variant for jobs whose cost scales with a known size.

  Arguments:
    jobs      = the job payloads (any values; kept as given).
    n_workers = number of buckets (>= 1).

  Returns:
    buckets = a list of n_workers lists of jobs.
  """
  buckets = []
  for _ in range(n_workers):
    buckets.append([])
  for j, job in enumerate(jobs):
    buckets[j % n_workers].append(job)
  return buckets


# --- VRAM-token packing (the --gpu-pack machinery) ---
# One GPU is modeled as GPU_TOKENS = 4 capacity tokens (quarters). A
# job's token count comes from its estimated VRAM fraction; jobs run
# concurrently on a GPU only while their tokens sum to at most 4.
# The quantization implements the packing rule exactly: a job above
# 40% of the GPU runs exclusive, one between 20% and 40% may share
# with one other, one at or below 20% may share four ways.
GPU_TOKENS = 4

# Fixed per-process GPU overhead the fraction estimate charges before
# any data: the CUDA context (~0.6 GB), model + optimizer state
# (small at these model sizes), the chi2 precision matrix (a 3000^2
# float64 Cinv is ~72 MB), torch.compile workspaces, and allocator
# fragmentation. 2 GiB is deliberately conservative; on an H200
# (141 GB) it is ~1.5% of the card.
VRAM_OVERHEAD_BYTES = 2 * 1024 ** 3


def estimate_train_vram_fraction(n_rows, dv_width, total_bytes):
  """
  Conservative fraction of one GPU a single training run needs.

  The resident-regime peak is dominated by the encoded target set
  plus the pre-shuffle's transient copy of the current chunk (worst
  case: the whole resident set again), so the data term is budgeted
  at 2 * n_rows * dv_width float32 values. dv_width is the full
  on-disk dv width, an upper bound on the kept (masked) width the
  loaders actually stage, so the estimate errs high. The fixed
  VRAM_OVERHEAD_BYTES rides on top. If a run overflows the estimate
  anyway, the loaders degrade to streaming against the GPU's real
  free memory rather than crashing (batching.py sizes each source
  against torch.cuda.mem_get_info at build time).

  Arguments:
    n_rows      = training rows this job stages (its N_train).
    dv_width    = columns of the dv dump (the full data-vector
                  length; upper-bounds the encoded target width).
    total_bytes = the GPU's total memory, e.g.
                  torch.cuda.get_device_properties(k).total_memory.

  Returns:
    the estimated fraction of the GPU (a float; may exceed 1.0 --
    vram_tokens saturates anything above 40% at exclusive).
  """
  data = 2.0 * float(n_rows) * float(dv_width) * 4.0
  return (VRAM_OVERHEAD_BYTES + data) / float(total_bytes)


def vram_tokens(fraction):
  """
  Map an estimated VRAM fraction to capacity tokens (of GPU_TOKENS).

  The packing rule, quantized to quarters of a GPU:

      fraction <= 0.2        -> 1 token  (up to 4 share a GPU)
      0.2 < fraction <= 0.4  -> 2 tokens (up to 2 share a GPU)
      fraction > 0.4         -> 4 tokens (exclusive)

  Arguments:
    fraction = estimated share of one GPU's memory (from
               estimate_train_vram_fraction).

  Returns:
    tokens this job must hold while it runs (int).
  """
  if fraction <= 0.2:
    return 1
  if fraction <= 0.4:
    return 2
  return GPU_TOKENS


def _lane_main(gpu_id, lane_id, setup_fn, job_fn, jobs_q, result_q,
               gate, extra):
  """
  One worker process: pin the GPU, set up once, drain the job queue.

  Runs in a spawned child. Claims GPU gpu_id (so every default-device
  op and mem_get_info read this card), builds the per-worker state
  once via setup_fn (the expensive part: an EmulatorExperiment with
  its staged data), then loops: pull (payload, tokens) items until
  the None sentinel, run job_fn on each, put its result. Under
  packing the gate (lock, semaphore) serializes token acquisition --
  see the module docstring's graph.

  job_fn must be total (catch its own per-job exceptions and return
  an error-marked result), so one bad point never kills the lane; a
  setup_fn failure puts one ("__lane_failed__", gpu_id, repr(err))
  result and exits, and the parent's liveness check reports it.

  Arguments:
    gpu_id   = CUDA device index this lane owns.
    lane_id  = lane index on that GPU (0 when packing is off).
    setup_fn = module-level callable setup_fn(gpu_id, extra) -> state,
               run once per lane.
    job_fn   = module-level callable
               job_fn(gpu_id, state, payload, extra) -> result.
    jobs_q   = this GPU's job queue of (payload, tokens) + sentinels.
    result_q = queue the parent drains, one item per job.
    gate     = None (no packing) or (lock, semaphore) for this GPU.
    extra    = driver payload forwarded to setup_fn / job_fn.
  """
  if torch.cuda.is_available():
    torch.cuda.set_device(gpu_id)

  try:
    state = setup_fn(gpu_id, extra)
  except Exception as err:              # noqa: BLE001 -- report, not raise
    result_q.put(("__lane_failed__", gpu_id, repr(err)))
    return

  while True:
    item = jobs_q.get()
    if item is None:                    # sentinel: no more jobs
      break
    payload, tokens = item
    if gate is not None:
      lock, sem = gate
      # hold the per-GPU lock while collecting this job's tokens so
      # two lanes never interleave partial grabs (the classic
      # multi-token deadlock); finishing lanes release without the
      # lock, which is what unblocks a waiting holder.
      with lock:
        for _ in range(tokens):
          sem.acquire()
    try:
      result_q.put(job_fn(gpu_id, state, payload, extra))
    finally:
      if gate is not None:
        for _ in range(tokens):
          sem.release()


def run_gpu_pool(setup_fn, job_fn, buckets, extra,
                 lanes_per_gpu=1, job_tokens=None, on_result=None):
  """
  Run pre-assigned job buckets across GPUs, one process per lane.

  Spawns min(lanes_per_gpu, len(bucket)) worker processes per GPU
  (spawn start method: each child is a fresh interpreter with its
  own CUDA context), fills one job queue per GPU, and drains one
  result per job, calling on_result as each arrives (the parent does
  all logging, so worker streams do not interleave). With
  lanes_per_gpu > 1 a per-GPU (lock, Semaphore(GPU_TOKENS)) gate
  enforces the token packing; job_tokens maps each payload to its
  token count (vram_tokens of its estimated fraction).

  Arguments:
    setup_fn      = module-level setup_fn(gpu_id, extra) -> state.
    job_fn        = module-level
                    job_fn(gpu_id, state, payload, extra) -> result;
                    must catch its own exceptions (see _lane_main).
    buckets       = list per GPU of job payloads (lpt_assign /
                    even_assign output).
    extra         = picklable payload forwarded to both callables.
    lanes_per_gpu = concurrent worker processes per GPU (1 = the
                    plain one-training-per-GPU mode; GPU_TOKENS is
                    the useful maximum under packing).
    job_tokens    = callable payload -> tokens (required when
                    lanes_per_gpu > 1; ignored otherwise).
    on_result     = optional callable(result), invoked in the parent
                    per drained result (logging).

  Returns:
    the list of results, in arrival order.
  """
  import torch.multiprocessing as mp

  ctx = mp.get_context("spawn")
  result_q = ctx.Queue()

  total = 0
  procs = []
  # The parent must hold its own references to every per-GPU queue
  # and gate until the workers exit: a Process releases its args
  # after start() (Python 3.14), and a garbage-collected Lock /
  # Semaphore / Queue unlinks its named OS semaphore -- a child that
  # has not finished booting then dies rebuilding it
  # (FileNotFoundError in SemLock._rebuild).
  keepalive = []
  for g, bucket in enumerate(buckets):
    if not bucket:
      continue
    total += len(bucket)
    n_lanes = min(lanes_per_gpu, len(bucket))
    gate = None
    if n_lanes > 1:
      gate = (ctx.Lock(), ctx.Semaphore(GPU_TOKENS))
    jobs_q = ctx.Queue()
    keepalive.append((jobs_q, gate))
    for payload in bucket:
      tokens = job_tokens(payload) if gate is not None else 1
      jobs_q.put((payload, tokens))
    for _ in range(n_lanes):
      jobs_q.put(None)                  # one exit sentinel per lane
    for lane in range(n_lanes):
      p = ctx.Process(target=_lane_main,
                      args=(g,
                            lane,
                            setup_fn,
                            job_fn,
                            jobs_q,
                            result_q,
                            gate,
                            extra))
      p.start()
      procs.append(p)

  # Drain one result per job. The timeout + liveness check turns a
  # dead worker (setup crash, OOM kill) into a loud error instead of
  # a silent hang; a "__lane_failed__" marker names the setup error.
  results = []
  got = 0
  while got < total:
    try:
      r = result_q.get(timeout=60)
    except queue_mod.Empty:
      alive = False
      for p in procs:
        if p.is_alive():
          alive = True
      if not alive:
        raise RuntimeError(
          f"GPU pool workers all exited with {total - got} job(s) "
          "unreported -- check the worker stderr above")
      continue
    if isinstance(r, tuple) and len(r) == 3 and r[0] == "__lane_failed__":
      raise RuntimeError(
        f"worker setup failed on gpu {r[1]}: {r[2]}")
    results.append(r)
    got += 1
    if on_result is not None:
      on_result(r)

  for p in procs:
    p.join()
  del keepalive                       # workers are done; release now
  return results
