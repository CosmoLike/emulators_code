---
name: emulator-floor-is-data-coverage
description: "The cosmic-shear emulator's frac>0.2 floor is DATA-LIMITED (training-set size), settled by the data-scaling / learning curve: the baseline used only ~1/10 of the omega_b h^2<0.035 pool, and growing N_train drops val frac>0.2 from 0.219 (10k) to 0.100 (46k, the GOAL) toward the full 82k pool. Fix: train on more of the available data. A mid-session 'it is capacity' call (from train==val underfitting) was WRONG and was refuted by this curve -- train==val rules out overfitting but does NOT prove capacity; only the learning curve (metric vs N_train) settles capacity-vs-data. Loss-shaping / LR / batch / per-element weight were all correctly exhausted as the lever; the real lever was always more data."
metadata:
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

**FINAL CONCLUSION (2026-06-24, CORRECTED): the floor is DATA (training-set
size), not capacity.** A mid-session read called it capacity from train == val
underfitting; the data-scaling experiment REFUTED that. Training a fresh model at
increasing N_train (drawn from the omega_b h^2<0.035 pool; the baseline used only
~1/10 of it) drops val frac>0.2 decisively:

    N_train 10k -> val frac>0.2 0.219
    N_train 46k -> val frac>0.2 0.100   (the goal)
    pool = 82k (full physical cs_16; 82k point pending)

More data helps -- a lot -- and the 0.10 goal is already reached at 46k. So the
floor was simply the 10% training subset being too small; on the TOY problem the
fix is to train on more of the available pool. The note's original name (data
coverage) was right.

BUT do not stop at "add data": that is the toy answer. The real target (T=512 +
w0wa + TATT) makes N_train the binding constraint, so the project goal is SAMPLE
EFFICIENCY -- the position of this learning curve, the smallest N_train to reach
the target -- not the floor. See [[emulator-sample-efficiency-is-the-goal]]. The
levers (rescaling, architecture, features) are to be judged by whether they shift
this curve LEFT, which is exactly the test they were NOT given this session.

Why the capacity call was wrong (the lesson worth keeping): train == val rules
out OVERFITTING (low variance) but does NOT prove capacity. A regularized model
that fits a small training set only partway looks identical -- train == val, both
high -- whether the limit is capacity or data sparsity. The capacity-vs-data
question is settled only by the LEARNING CURVE (metric vs N_train), which here
falls steeply -> data-limited. The train-vs-val gap diagnoses overfitting; the
learning curve diagnoses the floor. Mistake made: the gap was read as proving
capacity, asserted as "earned," and written down before running the curve.

**The diagnostic ladder that earned this (the transferable part):**
- Threshold ladder (0.2, 0.5, 1, 10, 100): the failures are a SHOULDER piled just
  above 0.2 (~half of the >0.2 are in 0.2-0.5), not a heavy tail. A log-normal fit
  to the bulk (median 0.05, sigma_log~1.65) reproduces the 0.5 and 1 fractions ->
  the 0.2-1 shoulder is the upper tail of ONE broad distribution, not a separable
  population. You cannot narrow a spread by reweighting -> loss-shaping is dead on
  arrival.
- Optimization ruled out: halving the LR tightened the late-epoch bounce but did
  not move the floor (bounce = step-size on near-threshold points; floor is
  underneath). Batch size is also a dead lever -- under the sqrt-lr coupling all
  batch sizes converge to the same frac (the coupling holds the gradient-noise
  scale fixed), so bigger batch cannot shrink the floor; 512 only broke because
  the coupled lr overshot.
- Loss-shaping ruled out empirically: per-cosmology focal + kappa sweep, a
  threshold-centered bump, and a per-ELEMENT focal (ElementWeightedChi2, beta=4)
  were all neutral.
- The MARGINAL per-element lens MISLED us: it flagged the highest source bin
  (z~1.34) at small theta. But the network leaves ~1.5 sigma MARGINAL residuals
  there even on TRAINING and the loss tolerates it -> those are correlated
  common-mode directions the chi2 barely charges for. Diagnose in the METRIC's
  own coordinates (the chi2 is a sum of squares in the whitened/decorrelated
  space), not a convenient marginal one.
- No conditioning bug: chi2 == ||pred-target||^2 to ~0.1% (max rel 1.6e-3), so
  the whitening basis IS the chi2 basis.
- DECISIVE (corrected): the LEARNING CURVE (frac>0.2 vs N_train), NOT the
  train-vs-val gap. train == val only rules out overfitting; the curve here falls
  0.22 -> 0.10 from 10k -> 46k cosmologies = DATA-limited (more data helps). The
  earlier "train == val -> capacity" read was WRONG. (Tempering confound for the
  per-element view: T_val = T_train/2, so val < train per element is just the
  temperature; compare at the metric, not per-element rms.)

Everything below is the earlier investigation history (intermediate, now
superseded by the FINAL CONCLUSION above). The line "NOT a loss-shaping or
model-capacity problem" was an early read; loss-shaping is correctly exhausted,
but the floor IS capacity.

---

The emulator ([[emulator-pipeline-and-goal]]) hit frac>0.2 ~= 0.36 (goal < 0.10).
An early investigation framed this as NOT a loss-shaping or model-capacity problem
(the capacity half is corrected above).

**Loss-shaping is exhausted (all tried, all failed):**
- Trim annealing (5% -> 0, several schedules): never beat the const-5% baseline.
- Focal/hardness reweighting (w=(c/(c+kappa))^gamma, kappa=0.2, gamma->2): made
  frac>0.2 WORSE (0.36 -> 0.44). Monotone focal up-weights the unfittable >0.2
  points (no gain) and de-protects the points just under 0.2, which drift up.
  (median 0.08->0.16 is irrelevant -- only frac>0.2 matters.)
Both fail because the hard points cannot be fit from the available training data.

**Real cause:** the OLD validation set was a random subset of the SAME T-sampled
training file, so it spanned the full T distribution INCLUDING its sparse edges.
The emulator is accurate in the dense interior, inaccurate at the under-covered
edges; the persistently-hard (high-chi2) val points are the edge cosmologies, and
no loss shape fits them (little training data there).

**Fix (the user's plan):** validate on a SEPARATE file whose cosmologies are
drawn at T_train/2 -- the sampling/proposal covariance tempered to T/2 (same
generation pipeline, so same param columns + order + dv length = full 3x2pt).
This concentrates the val cosmologies in the well-covered interior and AVOIDS the
training edges -- and the interior is where real inference lives (posteriors are
concentrated), so it grades the emulator where it counts. Only the sampling
temperature changes; keep param_geom / chi2fn built from the TRAINING (T) covmat
and apply them unchanged to the val file -- do NOT re-whiten with a T/2 covmat
(the sampling covariance is a different object from the whitening covmat).
Expectation: frac>0.2 drops substantially once val is at T/2. If a floor remains,
only THEN add interior training density.

**Implemented (this session):** two sources -- train file (cs_16, T=16) and val
file (cs_8, T=8) -- each wrapped as a {C, dv, idx} source dict (train_set /
val_set). `_build_loaders_one` builds one source's loaders (returns
load_C/load_dv/load + the bytes it made resident); `build_loaders` calls it once
per source (threading `budget - used_train` into the val call, see
[[shared-budget-across-sequential-calls]]) and returns a nested dict
`{"train": {load_C,load_dv,idx,load}, "val": {...}}`. `eval_val` is
source-agnostic (takes a source sub-dict, called on data["val"]). The geometry
(chi2fn center dvt_off, param_geom center c_off + covmat, Cinv) is built once from
the TRAINING source (derive stats from train_set, use the cs_16 covmat) and
applied unchanged to val. The dv-width assert lives in `_build_loaders_one`, so it
guards both sources for free.

**Result (T/2 baseline run):** with the two-source T/2 validation live and a
long run (cs_16 train / cs_8 val, 1500 epochs, bs 256, trim-anneal 0.1->0.025
cosine, focal gamma->2 at kappa 0.2, AdamW wd 1e-4, plateau patience 25 x0.8),
frac>0.2 still plateaus at ~0.36 (val median ~0.14, mean ~200; >1 ~0.18,
>10 ~0.09, >100 ~0.06). So validating at T/2 did not clear the floor -- the 0.36
is not merely an edge-evaluation artifact (caveat: not a clean A/B, since
epochs / bs / loss all changed vs the earlier 0.36). The bulk already meets the
goal (median 0.14 < 0.2); the blocker is a ~36% heavy tail above 0.2. Per the
plan above, the remaining lever is interior training density / model capacity,
not more loss shaping (trim-anneal + focal here were neutral-to-worse vs the
const-5% baseline, as before). Next diagnostic before picking a lever: pull the
parameters of the worst val cosmologies -- clustered at extremes = under-covered
(add density), scattered = contamination. The train set is only ~10% of the
cs_16 file (Ntrain = Ndvs0 // 10), so more / denser training is the cheap first
thing to try.

**Coverage check (train T=16 vs val T=8 triangle):** the two parameter
distributions nearly coincide -- val sits fully inside train (no extrapolation)
but is only marginally tighter, not the concentrated interior the T/2 plan
assumed. Likely because several params (n_s, Omega_b, H0) are prior-range-bound
with boxy marginals, so halving the sampling temperature barely narrows them
(only the curvature-constrained params -- Omega_m, A1, DZ -- tighten a little).
So T/2 did not separate an interior from the edges; val still spans the full
training range, sparse edges included -- which is why the T/2 baseline did not
move the 0.36. Extent coverage is fine; the floor is density / capacity within
the covered volume (and/or specific hard sub-regions -- read off the
chi2-colored triangle). To grade on a genuinely tight interior would take a much
lower-T val (e.g. T=1-2), not T/2.

**Confirmed driver (chi2 vs omega_b h^2):** scoring val and plotting
log10(dchi2) against omega_b h^2 = Omega_b * (H0/100)^2 gives a sharp cliff at
omega_b h^2 ~ 0.04: below it the bulk clears the 0.2 goal, above it almost every
point blows up to dchi2 ~ 1e2-1e4. The driver is the derived product (high
Omega_b together with high H0), which a per-parameter scan misses; it is also
the sparse corner (the tail of a product is thinly sampled). Crucially 0.04 is
~2x the physical value (Planck/BBN omega_b h^2 ~ 0.0224), so the failing region
is unphysical -- with a BBN/CMB omega_b h^2 prior the posterior never reaches the
cliff, and even LSST-only posteriors concentrate well below it. So a large part
of the 0.36 frac>0.2 is the emulator failing where no inference ever goes (the
broad T=16 prior just includes these extreme-baryon cosmologies and the metric
counts them). Levers, in order: (1) report/select the metric over the
inference-relevant region (cut high omega_b h^2) -- the honest accuracy where it
matters; (2) add targeted training density at high omega_b h^2 only if that
corner must be covered. Even in-range points (0.02-0.04) trend up with
omega_b h^2, so some genuine hardness remains below the cliff -- (1) alone will
not reach frac<0.10, but drops it a lot. The sharpest form of
grade-where-inference-lives.

**Cut-scan (corrects the "mostly unphysical" read above):** restricting the val
metric by an omega_b h^2 upper cut barely moves frac>0.2 -- all 0.433,
<0.04 0.384, <0.035 0.331, <0.03 0.307, <0.025 0.290 (median 0.155 -> 0.106).
So the >0.04 cliff is only ~0.05 of the 0.433: catastrophic per point
(dchi2 ~ 1e2-1e4, so it dominates the mean) but only ~8% of points, so it barely
moves the count-based frac>0.2. The failure rate does climb monotonically with
omega_b h^2 (~29% below 0.025 -> ~99% above 0.04), confirming it as the hardness
axis -- but even the cleanest physical region (<0.025, near Planck 0.0224) still
fails 29%, ~3x the 0.10 goal. Conclusion: metric-restriction is the honest
number to report (~0.29-0.31, not 0.43) but will not reach 0.10 on its own; the
floor is broad, moderate hardness across the physical region -- a general
accuracy (density / capacity) problem, not the unphysical tail (that was the mean
talking, not the count). The train set is still only ~10% of cs_16, so the first
lever is more training density (helps the sparse high-omega_b h^2 corner and the
in-range baseline both); optionally drop the unphysical >~0.04 tail from training
so capacity is not spent fitting cosmologies no posterior visits.

A separate lever now being explored is [[analytic-scaling-preprocessing]] (ratio
preprocess the dv by an analytic model) -- but it is broadband-only: the
zero-baryon reference has no Omega_b dependence, so by construction it cannot
move this omega_b h^2 floor. It eases the in-region bulk, not the hard tail.

**Why:** stops us re-running loss tricks (trim/focal/bump) or capacity bumps as
the path to the floor -- they were tried; the floor was edge-evaluation, fixed by
validating at T/2. Infra built (run_emulator spec-dicts, anneal_value, focal-loss
knobs, sweeps) is reusable. Pairs with [[probe-generalization-bugs]].
