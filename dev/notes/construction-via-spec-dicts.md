---
name: construction-via-spec-dicts
description: "Generalize constructible components (model/optimizer/scheduler) via spec dicts {cls: Class, **kwargs} + a make_X helper; keep computed/injected args out of the dict."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

The user prefers every constructible component to be parameterized by a spec
dict whose `"cls"` key holds the CLASS (a first-class value, the same factory
trick as the norm/act callables) and whose other keys are its constructor
kwargs. A small `make_X(spec, injected...)` helper pops `"cls"` and forwards the
rest:

  make_model(model_opts, input_dim, output_dim, device)  # cls + arch kwargs
  make_optimizer(model, opt_opts, lr, device)            # cls + weight_decay/...
  make_scheduler(optimizer, sched_opts)                  # cls + mode/patience/...

Arguments that are COMPUTED or device-dependent stay INJECTED by the helper, not
in the dict: lr (sqrt-batch rule), fused / torch.compile (gated on
device.type), input/output dims (from data + geometry), the optimizer handed to
the scheduler. `run_emulator` takes one spec dict per component (model_opts,
opt_opts, lr_opts, sched_opts).

**Why:** the user asked for this generalization three times (optimizer, model,
scheduler) — apply it to any new constructible component too.

**How to apply:** new component -> spec dict {cls, **kwargs} + a `make_X` helper
(with a formal Arguments docstring noting which args are injected). Caveat:
generalizing the scheduler CLASS does NOT generalize STEPPING — ReduceLROnPlateau
steps with the metric, others with none (branch on `isinstance`); per-batch
schedulers (OneCycleLR) step in the batch loop, not once per epoch.

Captured in the [[pytorch-teaching-style]] skill (Recurring idioms) too.
