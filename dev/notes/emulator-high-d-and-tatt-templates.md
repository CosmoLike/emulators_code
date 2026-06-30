---
name: emulator-high-d-and-tatt-templates
description: "How the cosmic-shear emulator scales to high parameter dimension and many LCDM extensions, WHY TATT intrinsic alignment is orders of magnitude harder to emulate than NLA, and the LEVER that fixes it. Core principle: emulator sample complexity scales with the EFFECTIVE (active, nonlinear) dimension, NOT the nominal parameter count -- nuisances that act ~linearly (photo-z DZ, IA NLA amplitude) are cheap (the hard-direction regression showed DZ/A1 ~0 hardness), so Roman's extra n(z) bins add EASY directions not effective dimension. The curse of dimensionality hits LOCAL methods (kNN/local-linear, which the global emulator beats 1000x), NOT a global net learning a low-eff-dim smooth function. TATT explodes because its ~5 IA params (a1,a2,b_TA + z-evolution) enter the dv as PRODUCTS/cross-terms up to quartic over a WIDE prior = a coupled high-curvature wide-prior subspace. THE LEVER (TATT = poster child of 'reduce effective dim via known physics structure'): the dv is EXACTLY a polynomial in the TATT amplitudes with cosmology-only smooth template coefficients -- emulate the ~10 smooth templates K_t(cosmo,z), apply the TATT polynomial c_t(a1,a2,b_TA) in CLOSED FORM at inference; the amplitudes leave the emulation entirely (zero training points in amplitude space, prior-width-independent), exactly as A_s factors out of C_ell. Also: low-discrepancy sampling helps coverage only in LOW-D (curse kills it high-D); active learning needs CONCENTRATED error, so for DIFFUSE failures the lever is representation not sampling."
metadata:
  node_type: memory
  type: project
---

Forward-looking strategy for scaling the emulator ([[emulator-pipeline-and-goal]],
[[emulator-sample-efficiency-is-the-goal]]) to high dimension and many LCDM
extensions (Roman: more n(z) bins; w0wa, TATT, neutrinos, modified gravity).

**STATUS UPDATE 2026-06-26: the template-factoring LEVER below is now BUILT and
NLA-validated -- see [[npce-and-ia-template-factoring]] for the implementation
(IAFactorGeometry / IATemplateMLP / IATemplateChi2 + nla_coeffs/tatt_coeffs).** Two
refinements learned in building it: (1) implement the factoring as an ARCHITECTURE
on the EXISTING scattered samples (the amplitudes already sampled from their prior,
read per-sample by the loss), NOT as structured re-simulation/extraction -- same
sims, apples-to-apples, the templates are identified implicitly. (2) Only the
amplitudes (linear coefficients) factor out; the redshift-evolution POWERS (eta,
inside the integral) STAY emulated -- the amplitude/power split. NLA factoring came
out NEUTRAL on the toy (the amplitude is the easy/narrow-prior direction) -- the win
SCALES WITH PRIOR WIDTH, so it is the real payoff for TATT's wide coupled amplitudes,
not for the toy. PCE/NPCE (a smoothness prior) was the OTHER thing tried this session
and FAILED to move the floor -- smoothness is not the missing ingredient
([[npce-and-ia-template-factoring]]); exact structure (this) is.

**Governing principle: EFFECTIVE dimension, not nominal parameter count.** Emulator
sample complexity scales with the number of directions the dv depends on STRONGLY
and NONLINEARLY (the active subspace), not with the total parameter count. Evidence
from this project: the hard-direction regression
([[emulator-sample-efficiency-is-the-goal]]) found the photo-z shifts (LSST_DZ_*)
and IA amplitudes (LSST_A1_*) contribute ~0 to hardness -- they are near-linear
nuisances the net learns cheaply. So adding more nuisance-like params (Roman n(z)
bins) adds EASY directions, NOT effective dimension; 5 more photo-z != 2^5x more
data. What costs is extensions that add genuine nonlinear physics (neutrinos, MG,
the w0wa/TATT nonlinearities).

**Why the curse of dimensionality does NOT kill the emulator** even though local
coverage ~ N^{-1/d} is hopeless in 12-20D: the emulator is a GLOBAL learner of a
LOW-effective-dim smooth function, not a local interpolator. The curse hits LOCAL
methods (kNN, local-linear -- which the trained net beats ~1000x, see the failed
local-linear-floor test in [[emulator-sample-efficiency-is-the-goal]]). So do NOT
frame high-D as "cover the space" (impossible); frame it as "keep the effective
dimension low."

**Why TATT IA is orders of magnitude harder than NLA** (the user's struggle). NLA:
the dv depends on ONE amplitude A1, low-order (GI ~ A1, II ~ A1^2), narrow prior ->
IA effective dim ~1, smooth. TATT: ~5 amplitudes (a1 tidal alignment, a2 tidal
torquing, b_TA density weighting, + z-evolution eta1/eta2) entering through PRODUCTS
and cross-terms (a1*a2, a1*b_TA, a1^2; II quadratic in those -> up to ~quartic in
the raw params), over a WIDE prior (a1,a2,b_TA poorly constrained). Four compounding
costs: (1) effective dim 1 -> ~5 and COUPLED (cross-terms force covering the JOINT
volume); (2) high order -> high CURVATURE (the hardness driver); (3) wide prior ->
huge dynamic range in the products; (4) anisotropic/degenerate sensitivity. The
curse lands on the IA subspace, on top of cosmology.

**THE LEVER (poster child of "reduce effective dim via known physics structure"):
factor the analytic polynomial out.** The dv is EXACTLY linear in cosmology-only
templates with the TATT amplitudes as known closed-form coefficients:

    dv = GG(cosmo) + sum_t  c_t(a1, a2, b_TA) * K_t(cosmo, z)

where c_t are the analytic polynomial coefficients (a1, a2, b_TA*a1, a1^2, a1*a2,
a2^2, ...) and K_t are the per-term GI/II template contributions -- depend ONLY on
cosmology+redshift, smooth, ~6-10 of them. So EMULATE the smooth cosmology-only
templates K_t (each easy, like emulating GG) and APPLY the TATT polynomial c_t in
CLOSED FORM at inference. The amplitudes leave the emulation entirely -> zero
training points in amplitude space, INSENSITIVE to the (wide) prior on the
amplitudes. Exactly how A_s factors out of C_ell ([[analytic-scaling-preprocessing]]);
TATT factors the same way as a polynomial instead of a scalar. NLA already half-does
this (a 1-param quadratic). Trades "1 brutal coupled-5D-wide-prior emulation" for
"~10 easy cosmology-only emulations." Needs cosmolike to output the per-term GI/II
templates (it assembles them internally). Weaker fallback if templates can't be
separated: a STRUCTURED low-order grid in (a1,a2,b_TA) -- since the dependence is
low-order polynomial, far cheaper than random over the wide prior. General rule:
NEVER emulate a parameter dependence you can write down.

**High-D sampling (corrected from low-D advice).** Low-discrepancy (scrambled Sobol
-> probit -> Cholesky) draws of the Gaussian-T kill i.i.d. clumping and help
COVERAGE in LOW dimensions, but the advantage VANISHES in high-D (Sobol star-
discrepancy ~ (log N)^d / N beats i.i.d.'s N^{-1/2} only for N >> 2^d). Active
learning (query-by-committee) needs CONCENTRATED error to beat random; for DIFFUSE
failures (as at T=16) it degenerates to uniform -> when difficulty is diffuse the
lever is REPRESENTATION (effective-dim reduction), NOT sampling. The temperature-
Gaussian itself is a good, elegant baseline (posterior-weighted; space-filling is
correctly rejected -- it runs to corners the chain never visits). At big T (e.g.
512) coverage is already generous for well-constrained params, so the binding issue
is hardness/effective-dimension, not coverage. Anisotropic / structure-aware
sampling (dense in the few nonlinear directions, sparse in nuisances) is the high-D
importance-sampling that can actually win -- but only if the hardness concentrates
in a low-D subspace.

**Next diagnostic (offered, not yet built): active-subspace / global-sensitivity
analysis** -- for each param/combination, how much does the dv vary along it (chi2
metric)? Ranks the effective dimension (few strong nonlinear cosmology directions +
a long tail of cheap near-linear nuisances); tells you BEFORE spending sims whether
a given extension blows up the budget and where anisotropic sampling should
concentrate. Generalizes the hard-direction regression from "where are the failures"
to "which directions matter at all."

**Why:** records the effective-dimension principle, the TATT-vs-NLA difficulty and
its template-factoring fix, and the corrected high-D sampling story, so the next
session goes straight to template-factoring TATT and the active-subspace diagnostic
rather than re-deriving why TATT is hard.
