---
name: per-bin-parallel-resmlp-plan
description: "NEXT-TASK (planned 2026-06-25, not yet built): a per-bin ParallelResMLP -- one ResMLP head per tomographic bin (xi+/-, source pair) -- so each head emulates a simpler, smaller, more coherent target, to lower the f(dchi2>0.2) plateau at a fixed N (the sample-efficiency goal). The enabling insight: the FULL whitening rotates into the full kept-block cov eigenbasis and MIXES all bins, so you cannot split the full-whitened output by bin. The fix (the user's design): split BEFORE whiten + BLOCK-DIAGONAL whitening (whiten each bin with its own within-bin cov sub-block, decorrelating only the thetas inside the bin). The chi2 metric KEEPS the full cross-bin Cinv. Records the design, the critical invariants, the build pieces, and the open choices so we can resume cold."
metadata:
  node_type: memory
  type: project
---

**STATUS: BUILT + RUN 2026-06-25 -- FAILED, SUPERSEDED by
[[resmlp-cnn-perbin-architecture]].** The pure per-bin ResMLP split came out WORSE
than the single ResMLP (~0.27 vs ~0.24 frac>0.2, at matched params). Why: (a) it
threw away the SHARED cosmology->dv map -- 30 heads each re-learned it alone, the
hard part replicated 30x not divided; and (b) a dense MLP is
OUTPUT-PERMUTATION-INVARIANT, so it never exploited the within-bin theta-smoothness
anyway (the dv's smooth-bins / sawtooth-theta / boundary-jumps layout is real but
INVISIBLE to a dense MLP -- reorder the outputs+targets+Cinv and it trains
identically). The fix is an AXIS-AWARE head (1D CNN) on a SHARED trunk, see
[[resmlp-cnn-perbin-architecture]]. The block-diagonal geometry + grouped batching
(GroupedLinear / einsum "gbi,gio->gbo" / groups=n_bins) below are CORRECT and
reusable for the CNN version; only the per-bin DENSE-MLP split is the dead part.
The original plan (still accurate as code) follows.

The lever to try for [[emulator-sample-efficiency-is-the-goal]]: a per-bin
ParallelResMLP.

**The lever.** One ResMLP head per tomographic bin = (xi+/-, source pair (i,j)).
Each head sees the full params, outputs only its bin's kept elements. Rationale:
one bin is a simpler, smaller, more coherent target (a single auto/cross xi over
theta), so each head emulates something easier -> lower effective DOF -> better
generalization from few points -> a lower CONVERGED PLATEAU at a fixed N (the
metric that matters, [[emulator-sample-efficiency-is-the-goal]]). Accepted cost:
more parameters total (~2*n_pair heads, ~30 for 5 source bins); that is fine --
the point is a simpler per-head function, not fewer params.

**The blocker, and the fix (the key design point).** The current model outputs
the FULL-WHITENED dv: whiten rotates into the eigenbasis of the FULL kept-block
covariance, so every output component is a linear combination of ALL pairs and
ALL thetas. Splitting that output by bin gives each head a band of global
eigenmodes, NOT a tomographic bin. The per-bin "smooth function of theta"
structure lives in the PHYSICAL dv, which the full rotation destroys. Fix (user's
correct call): **split BEFORE whiten, then BLOCK-DIAGONAL whitening** -- whiten
each bin with its OWN within-bin covariance sub-block (decorrelate only the thetas
inside that bin, no cross-bin mixing). This keeps unit-variance/decorrelated
outputs (the conditioning the full whitening gave) AND makes the output per-bin
separable. (The cheaper diagonal-only whitening -- scale by per-element sigma, no
rotation -- gives literal xi(theta) curves per head but pays a conditioning cost
because the output scale no longer matches the chi2's precision weighting;
block-diagonal is the better default.)

**CRITICAL INVARIANT -- the chi2 metric keeps the FULL cross-bin Cinv.**
Block-diagonal whitening changes ONLY the target basis (decorrelate within bin);
it must NOT change the metric. The cross-pair correlations are real and the
reported Delta-chi2 needs them. In the loss: un-whiten each bin (within-bin) ->
assemble the full physical residual across all bins -> contract with the FULL
Cinv_sq. CosmolikeChi2.chi2 already does this (r = unwhiten(pred-target);
r @ Cinv_sq @ r) -- just swap whiten/unwhiten for the block-diagonal versions and
leave Cinv_sq as the full kept-block precision.
THE TRAP: with full whitening, ||pred-target||^2 EQUALLED the chi2; with
block-diagonal it does NOT -- ||pred-target||^2 = the SUM of within-bin chi2s,
which DROPS the cross-bin terms. Do not let anyone "simplify" the loss to MSE
thinking it is the chi2; keep the explicit Cinv_sq contraction.
Consequence: the heads are NOT independent -- each only OUTPUTS its bin (the
simpler-target win), but the chi2's cross-bin Cinv_sq couples their gradients, so
they are trained jointly. That is correct.

**Build pieces (three):**
1. A **BlockDiagonalGeometry** (subclass of DataVectorGeometry, see
   [[geometry-loss-composition]]): for each bin, eigh the within-bin cov sub-block;
   store per-bin evecs/sqrt_ev; whiten/unwhiten operate block-wise (bins are
   contiguous runs in dest_idx order). Cinv_sq stays the FULL kept-block precision.
   squeeze / center / encode / decode / chi2 inherit unchanged.
2. **build_shear_angle_map** must ALSO record xi+/- per kept element (it currently
   stores theta_kept / zsrc_i / zsrc_j but NOT the +/- flag, so xi+ and xi- of the
   same pair/theta are indistinguishable). Then a bin = (xi+/-, zsrc_i, zsrc_j);
   the geometry exposes the per-bin kept-element counts (contiguous, summing to
   n_keep).
3. **ParallelResMLP(geom, ...)** reads those per-bin sizes -> one head per bin. The
   current ParallelResMLP uses an EQUAL n_parallel split; change it to take the
   geometry and split by the bin sizes. (The class otherwise works: balanced split
   verified, each head sees full input, concat rebuilds the vector in order.)

**Open choices to lock at the start of the build:**
- Bins = (xi+/-, pair) -> xi+ and xi- as SEPARATE heads (default; they have
  different covariances), vs (pair) -> xi+/- of a pair share one head.
- Confirm one ResMLP per bin despite ~2*n_pair heads (param count up).

**How to judge it.** The CONVERGED PLATEAU of frac>0.2 at a fixed useful N (~10k):
does it drop below ~0.20 toward 0.10? (See [[emulator-sample-efficiency-is-the-goal]]
for why the fixed-N plateau, not the small-N relative curve, is the test.)
Confound to state honestly: this tests per-bin-heads AND block-diagonal-whitening
TOGETHER -- they are inseparable (you cannot split by bin without leaving the
full-rotation basis).

**Why:** a concrete, resumable next step. The output-reparametrization levers (A
= RescaledChi2, B = ResidualBaseChi2) are CLOSED (they share plain's optimum, so a
single converged run can't separate them, and the user saw A == B == plain at
~0.20). The live levers are this per-bin head and the analytic INPUT feature
(AugmentedParamGeometry, [[geometry-loss-composition]]); both reduce the effective
DOF the net must learn from scarce data, which is the only thing that lowers a
DATA-limited plateau.
