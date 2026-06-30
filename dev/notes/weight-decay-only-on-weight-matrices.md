---
name: weight-decay-only-on-weight-matrices
description: "Don't weight-decay learned-activation / norm-gain / bias params; use wd=0 or an ndim>=2 param-group split."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

In the cosmic-shear ResMLP emulator, weight decay must NOT be applied to
shape/scale/bias parameters: `activation_fcn.gamma/beta` (learned-activation
shape), `Affine.gain/bias` (the per-block "norm"; `gain` inits to 1.0 and decay
would pull it toward 0, attenuating the residual signal — the worst offender),
and all `Linear` biases. Decaying these distorts the activation or kills signal
rather than regularizing.

**Why:** weight decay (L2) pulls every parameter it touches toward 0. That's
meaningful only for connection-weight matrices; for activation/norm/bias params
0 is an arbitrary or harmful target.

**How to apply:** the user's chosen default is `weight_decay = 0` (simple, safe;
emulators with abundant data rarely need L2). The surgical alternative — keep
decay on the `nn.Linear` weight matrices only — uses the standard `ndim >= 2`
split (weight matrices are 2D; biases, `Affine`, `gamma`/`beta` are 1D):

```python
decay, no_decay = [], []
for _, p in model.named_parameters():
  (decay if p.ndim >= 2 else no_decay).append(p)
opt = torch.optim.Adam(
  [{"params": decay,    "weight_decay": weight_decay},
   {"params": no_decay, "weight_decay": 0.0}],
  lr=learning_rate)
```

Captured in the [[pytorch-teaching-style]] skill too.
