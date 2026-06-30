---
name: geometry-loss-composition
description: "Emulator architecture after the 2026-06-24 refactor: CosmolikeChi2 no longer INHERITS DataVectorGeometry -- it HOLDS one (self.geom, composition). Build the geometry ONCE (DataVectorGeometry.from_cosmolike = one cosmolike read + one eigh), then wrap it in any loss: CosmolikeChi2(geom) / RescaledChi2(geom) / ResidualBaseChi2(geom) / ElementWeightedChi2(geom). The pipeline tells the loss variants apart by a needs_params capability flag (getattr(chi2fn, 'needs_params', False)), NOT isinstance. Records the loss family (plain / A=rescaled / B=residual-base / element-weighted), the A-vs-B conditioning distinction, and the diagnostic rule (branch decode/encode/chi2 on needs_params + pass the whitened params)."
metadata:
  node_type: memory
  type: project
---

The emulator ([[emulator-pipeline-and-goal]]) loss/geometry layer was refactored
(2026-06-24) from inheritance to **composition**. Next session: do NOT assume
CosmolikeChi2 inherits DataVectorGeometry, and do NOT branch on
isinstance(chi2fn, RescaledChi2).

**Composition.** `DataVectorGeometry` (the object usually named `geom`) owns ALL
geometry: squeeze/whiten/encode/decode/unsqueeze, dest_idx, total_size,
Cinv/Cinv_sq, center, and (after `build_shear_angle_map(geom)`) the angle map
theta_kept/zsrc_i/zsrc_j. `CosmolikeChi2(geom)` HAS-A geom (self.geom) and adds
only chi2 + loss. Build the geometry ONCE and wrap it in whatever loss:
`geom = DataVectorGeometry.from_cosmolike(device, dv_mean, probe="xi")` then
`chi2fn = CosmolikeChi2(geom)` (or RescaledChi2(geom), etc.). Every construction
site is now this two-step split (9 of them); `*.from_cosmolike` building a loss
directly is gone.

**Thin delegation keeps the pipeline unchanged.** CosmolikeChi2 forwards four
things to self.geom -- `dest_idx`, `total_size` (properties), `encode`, `decode`
(methods) -- so the loaders / run_emulator / eval read them off `chi2fn`
untouched. Only (1) construction, (2) the loss classes' internals (self.geom.X),
and (3) diagnostics that read geometry (now off `geom`, e.g. geom.evecs,
geom.squeeze, geom.theta_kept) changed. `build_shear_angle_map(geom)` (not chi2fn)
attaches the angle map to the geometry, so any loss variant and the input-feature
work reuse it.

**needs_params capability flag (replaces isinstance).** Every branch site
(`_build_loaders_one`, `training_loop_batched`, `eval_val`, `eval_source_chi2`,
the decode diagnostics) uses `getattr(chi2fn, "needs_params", False)`. It means
"this loss's encode/decode/chi2/loss take the whitened params (to build R)".
RescaledChi2 sets `needs_params = True` (inherited by ResidualBaseChi2);
CosmolikeChi2 and ElementWeightedChi2 leave it unset -> getattr default False. A
future param-aware loss only sets the flag; it need NOT subclass RescaledChi2. The
old `isinstance(chi2fn, RescaledChi2)` was a degenerate-case coincidence -- it
worked only because every R-aware loss happened to subclass A
([[probe-generalization-bugs]] family: a capability inferred from a type).

**The loss family** (`encode` builds the whitened target; `chi2` is the metric;
u = unwhiten(net output), c = center, R = analytic ratio per [[analytic-scaling-preprocessing]]):
- `CosmolikeChi2` (plain): target = whiten(squeeze(dv) - c); chi2 = plain masked
  Mahalanobis. needs_params False.
- `RescaledChi2` "A": R in encode AND chi2. d_pred = (u + c)/R; target =
  whiten(squeeze(dv)*R - c); chi2 divides the residual by R. needs_params True.
- `ResidualBaseChi2` "B": R in encode ONLY. d_pred = u + c/R; target =
  whiten(squeeze(dv) - c/R); chi2 is the INHERITED PLAIN one (R lives in the
  target, never the loss). Subclasses A to reuse _R / configure_rescaling / loss;
  overrides encode (c -> c/R), decode, and chi2 (delegates to CosmolikeChi2.chi2,
  accepts-and-ignores the params). Verified: plain chi2 on B's target == the true
  physical Δχ².
- `ElementWeightedChi2`: per-element focal weight on the residual (hard,
  tight-error-bar elements scaled up); plain geometry; eval chi2 unchanged.
  needs_params False.

**A vs B (the conditioning point, sized to evidence).** Same R; they differ only
in WHERE R touches d_pred. A divides the NET OUTPUT, so the chi2 gradient carries
diag(1/R) -- a per-cosmology conditioning factor. B moves only the constant
baseline (c -> c/R), so the net output enters at unit gain and the chi2 is plain
(no 1/R). Same global optimum; they differ ONLY in optimization conditioning,
which is order-unity (R spans ~a factor of 2; Adam launders the constant part).
So B is NOT expected to be a big win -- it is the CLEAN EXPERIMENT that isolates
that one variable: if B tracks plain while A sits above it at small N, the
conditioning hypothesis is confirmed; if A and B coincide, conditioning was not
the bottleneck and the whole inject-R-on-the-output direction is dead. The lever
with a real (representational, not just conditioning) mechanism is still the
analytic-as-INPUT-feature, untested ([[emulator-sample-efficiency-is-the-goal]]).
The 3-method sweep (build_chi2 with plain/rescaled/residual branches + a raise on
unknown) is wired; read it at small N (2k-10k).

**Diagnostic rule (a recurring bug source).** Any diagnostic that reconstructs the
physical dv (decode), builds a target (encode), or scores chi2 MUST branch on
needs_params and pass the whitened params Xb for A/B -- e.g.
`pred_phys = chi2fn.decode(pred, Xb) if getattr(chi2fn,"needs_params",False) else
chi2fn.decode(pred)`. A diagnostic that hardcodes `geom.decode(pred)` (plain) is
silently WRONG for A/B (A is off by /R, B by the c/R baseline); one that calls
`chi2fn.encode(dv)` with one arg raises TypeError on A/B. per_elem_rms / chi2_stats
/ the per-element + train-vs-val diagnostics now branch correctly. The chi2 ==
||pred-target||^2 identity (whitening basis = chi2 basis) holds for plain / B /
element-weighted but is expected to diverge for A (its chi2 has the /R) -- not a bug.

**Why:** the refactor changed the spine every cell depends on; without this note a
future session re-derives the wrong inheritance/isinstance model. Pairs with
[[emulator-pipeline-and-goal]] (component map), [[analytic-scaling-preprocessing]]
(the R and why A lost), [[no-global-variables-in-functions]] (the diagnostics
still read model/chi2fn as globals -- thread them).
