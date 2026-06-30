---
name: shared-budget-across-sequential-calls
description: "Refactoring one resource-allocating call into N sequential calls against a shared finite budget (GPU VRAM, RAM, handles) needs the running remainder threaded: each call subtracts what earlier ones made resident. A by-value budget handed to sequential allocators is a bug smell. And finish resource accounting in BOTH directions."
metadata:
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

When code that allocates from a finite SHARED resource (GPU VRAM, RAM, file
handles, a connection/rate budget) is split or duplicated into several SEQUENTIAL
calls, the accounting must thread the running remainder: call N plans against the
budget minus what calls 1..N-1 made resident. Handing the same by-value
`budget`/`limit` to each of N sequential allocators is the smell -- each sees the
full pool, so the total can overrun it (OOM).

**The miss (build_loaders):** split into `_build_loaders_one`, called once for
train then once for val. The val call used the FULL budget while the train set
was already resident on the GPU -> over-estimated free VRAM. Fix:
`_build_loaders_one` returns the bytes it made resident; the val call gets
`budget - used_train`.

**Why it slipped past me (the deeper lesson):** I half-saw it -- I had even
written a docstring note that "model + Cinv are counted in every call, so it's
conservative" -- but I stopped at the SAFE direction (double-counting
under-estimates -> harmless) and never checked the UNSAFE direction (the resident
train pool was not subtracted -> over-estimate -> OOM). "It's conservative /
everything fits on a big GPU" closed the analysis prematurely.

**How to apply:**
- When one allocation/sizing op becomes several sequential ones against one
  resource, ask: does each later call see the budget reduced by earlier
  allocations? If not, thread a running remainder (return bytes-used, subtract).
- A by-value budget/limit passed into a loop or a fixed sequence of allocators
  almost always needs to become a shrinking remainder.
- When a budget interacts with a shared resource, finish the accounting in BOTH
  directions -- name what is over-counted AND what is under-counted -- never stop
  at the reassuring half.

Pairs with [[emulator-pipeline-and-goal]]; captured in the
[[pytorch-teaching-style]] skill (Performance + Review) too.
