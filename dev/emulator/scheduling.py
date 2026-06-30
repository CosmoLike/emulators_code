"""Work-balancing helpers for spreading a sweep across GPUs.

A sweep of many independent jobs (here, one training run per N_train value)
must split the jobs so every GPU finishes at about the same time. This module
holds the pure, framework-free part -- which job goes to which GPU; process
spawning lives in the driver. The one function so far, lpt_assign, partitions
jobs by the Longest-Processing-Time rule for roughly equal per-GPU cost.
"""


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
