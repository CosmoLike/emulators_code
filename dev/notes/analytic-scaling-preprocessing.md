---
name: analytic-scaling-preprocessing
description: "Physics-informed RATIO preprocessing for the cosmic-shear emulator: multiply each data vector by R = xi_analytic(mid)/xi_analytic(cosmo) from a crude analytic model (E&H zero-baryon, linear, Limber, single-source-plane delta-n(z), H=H0), emulate the flatter residual, divide R back out before the chi2. The ratio cancels cosmology-common nonlinearity so a linear no-halofit model still works. Validated by spread ratio (As-only 0.79, +shape 0.515, +N amplitude 0.456; N=(Om h^2)^ns/h also lifts improved-fraction 0.89->0.98). include_amp=True is the standard. Helps the broadband bulk, NOT the omega_b h^2 floor. Full derivation in texnotes/analytic_scaling.pdf. RESULT (2026-06-24, NARROW): as a TARGET rescaling it did NOT help on THIS emulator -- on the learning curve WORSE than plain at small N (2k: 0.51 vs 0.57; 3.7k: 0.33 vs 0.38), converged to plain by ~7k. Not a bug (chi2 verified exact). A PLAUSIBLE but UNCONFIRMED mechanism (one case -- do NOT over-generalize to 'any reparametrization hurts') is conditioning: the /R undo reweights the net's output gradient per cosmology, vs the plain net whose output IS the chi2 residual. Actionable (this emulator): deprioritize target rescaling; UNTESTED alternatives = analytic as input feature or pretrain init (keep the plain loss). _analytic_R / E&H machinery reusable for those."
metadata: 
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

**RESULT (2026-06-24, NARROW -- one emulator, one test): target rescaling did NOT
help here.** Learning-curve test (plain vs rescaled, val frac>0.2):

    N_train   plain   rescaled
    2000      0.509   0.569
    3713      0.330   0.384
    6896      0.245   0.248   (converged)

It was WORSE at small N (where it was meant to help most,
[[emulator-sample-efficiency-is-the-goal]]) and converged to plain by ~7k. Not a
bug -- the chi2 is verified exact (chi2 == ||pred-target||^2 to 0.1%, round-trips
exact) -- so it is an optimization effect, not a metric error.

HYPOTHESIS for why (ONE CASE, UNCONFIRMED -- do NOT generalize to "any
reparametrization hurts"; most preprocessing reparametrizes and helps): the plain
net outputs the chi2's own whitened residual (clean ||y-t||^2), whereas rescaling
(1) whitens dv*R in the PHYSICAL cov basis though dv*R has a different cov (target
no longer unit-variance), and (2) the /R undo inserts a per-cosmology factor between
the net output and the loss (gradient carries 1/R, varies cosmology-to-cosmology).
Untested: this would be CONFIRMED only if fixing the target whitening / the /R
scaling recovered the benefit -- not attempted.

ACTIONABLE (evidence-based for THIS emulator, not a universal rule): target
rescaling lost the sample-efficiency test, so deprioritize it. UNTESTED
alternatives to use the analytic without touching the loss: as an input feature
(net predicts physical-whitened dv, analytic dv fed as input -> learn the
correction) or a pretrain init (pretrain on cheap analytic dvs, fine-tune on
cosmolike, plain loss). The _analytic_R / E&H machinery is reusable for both.

(Everything below is the original design rationale; correct as physics, but the
TARGET-rescaling application underperformed -- see the result above.)

A preprocessing scheme for the emulator ([[emulator-pipeline-and-goal]]) to
shrink the target's dynamic range so the network has less to fit.

**The idea (a ratio, which was the right call):** pick a reference "mid"
cosmology and preprocess element-wise `d_tilde = d * R`, with
`R = xi_analytic(mid) / xi_analytic(cosmo)`. The net emulates `d_tilde` (it
clusters near the mid cosmology); recover the physical `d = d_tilde / R` before
the chi2, so the covariance and reported metric are unchanged. A ratio (not a
difference, not absolute) is key: the cosmology-common part of
`xi_data/xi_analytic` (most of the nonlinear boost) cancels in mid/cosmo, so the
crude analytic only has to capture the cosmology-DEPENDENCE -- which is mostly
linear (amplitude, tilt, turnover). That is why a linear, no-halofit reference
still works, and best at small scales.

**The analytic model (all just to be a reference, accuracy not needed):** linear
P(k) with the Eisenstein-Hu zero-baryon transfer; Limber; each source bin a
single plane at its n(z) peak (delta-n(z)); pure dark energy H(z)=H0 so
chi = (c/H0) z; Hankel to real space as a delta at l*theta=1; Limber integral at
the kernel peak u_star. Key results:
- A_s carries the WHOLE amplitude: Omega_m and H0 cancel between the lensing
  kernel (q^2 ~ Om^2 H0^4) and P (~ As Om^-2 H0^-4), so C_l = As * G(shape).
  Dividing by As_mid/As is exact and removes the single largest variance.
- The shape collapses to one dimensionless wavenumber
  `q = K/(theta * z_eff * Gamma)`, `K = 100 Theta^2/(c u_star)`,
  `Gamma = Omega_m h`, `Theta = T_cmb/2.7`. So Om, h, z_s enter only through
  `q ~ 1/(theta * Gamma * z_s)`.
- Full factor: `R = R_amp * q_mid^ns_mid T(q_mid)^2 / (q^ns T(q)^2)`, with the
  per-cosmology amplitude scalar
  `R_amp = (As_mid/As) * (Om_mid h_mid^2)^ns_mid/h_mid / ((Om h^2)^ns/h)`.
  The `(Om h^2)^ns/h` piece is the surviving geometric amplitude N (z_s and
  theta cancel in the ratio) -- a SECOND amplitude direction beyond As, gated by
  the `include_amp` flag (default off, but the standard run sets it True). Each
  cosmology uses its own params, reference uses mid's; R=1 at mid.
- Cross tomographic pairs: `z_eff = min(z_i, z_j)` (the cross spectrum is cut off
  at the nearer source, so its effective scale is the lower bin's), NOT the
  geometric mean. Auto pairs: min = the bin's z. z_eff cancels at leading order
  in R, so this is a few-percent (asymmetric-pair) correction.

**Validation (spread ratio = std_rescaled/std_raw across val cosmologies, per
kept element; lower = more variance removed):** As-only (R=As_mid/As scalar)
median 0.791 (~37% of variance); +shape median 0.515 (~74%, 89% improved);
+N amplitude median 0.456 (~79%, 98% improved). So the SHAPE removes more than
As alone, and the N scalar adds more still -- use the full As+shape+N, do not
collapse to the As scalar. N also RAISES the improved fraction 0.89->0.98: the
shape term left the (Om h^2)^ns/h amplitude in the residual, which dominates at
large theta where T->1 and the variation is nearly pure amplitude (exactly the
elements shape made worse); N removes that, and is most reliable there (no
transfer curvature). By theta: best at small theta (2.8'-30', ratio ~0.34-0.5,
the constraining small-error-bar scales), ~1 only at the largest theta (>~500',
biggest covariance, down-weighted by Cinv). A net win on the scales that matter.

**What it does and does not fix:** removes the broadband (As / Om h / ns)
variance; the zero-baryon transfer has NO Omega_b dependence, so R cannot touch
the omega_b h^2 = Omega_b (H0/100)^2 direction -- which is exactly the hard,
high-omega_b h^2 cliff that drives the frac>0.2 floor (see
[[emulator-floor-is-data-coverage]]). Capturing that would need the WIGGLE
(with-baryon) E&H transfer. So treat this as "makes the bulk easier," not "fixes
the baryon tail."

**Code (in the notebook):** `_analytic_R(theta, z_eff, cosmo, cosmo_mid, names)`
is the single core (the q/T/R formula lives once); `analytic_shape_ratio` wraps
it over the masked/kept dv (flat (N, n_keep), for the pipeline) and `rescale_xi`
wraps it over the full unmasked xi matrices (for plotting). `build_shear_angle_map`
attaches the angle/tomography metadata (theta_kept, zsrc_i/j, theta_centers,
z_src, ntheta, source_ntomo, xi_size) by reading the dataset ini + n(z) file
only -- no cosmolike. `dv_to_xi` reshapes a flat dv row into plot_xi's
(theta, xip, xim) matrix layout. `RescaledChi2(CosmolikeChi2)` is the opt-in
subclass: overrides encode/decode (apply R) and chi2/loss (divide R out so the
chi2 stays on the physical dv); base class is untouched, so the two are
A/B-swappable. Derivation written up in `analytic_scaling.pdf` / `.tex` in the
`texnotes/` folder.

**Status: WIRED, VERIFIED, RUNNING (2026-06-24).** The %%time run now passes the
configured RescaledChi2; the only open task is reading frac>0.2. The pre-flight
sanity cell passes: max|R(mid)-1| = 0.0 exactly, and the xi layout checks out
(theta_kept rises 2.8'->33.78' = theta inner; (zsrc_i,zsrc_j) stays (0.33,0.33)
across the first pair = pairs outer), matching build_shear_angle_map's assumption.
Driver order that works: from_cosmolike -> build_shear_angle_map(chi2fn) ->
configure_rescaling(pgeom, cosmo_mid, names, include_amp=True) -> sanity ->
run_emulator (model_opts = plain ResMLP, chi2fn = the configured RescaledChi2). A
model-side variant (RescaledResMLP, rescaling in forward()) was explored and proven
identical in value+gradient (/tmp/test_model_wrapper.py) but rejected to keep the
plain ResMLP; the loss-side path with the loop branch was chosen instead.

**Compute R on the fly, do NOT store it.** R is a deterministic function of the
cosmological params, so an `(N_rows, n_keep)` stored R (the rejected `load_R`
plan) is pure waste and ~doubles the resident-target memory at scale. The loss is
handed the cosmology it already has and computes R itself. R is needed (1) to
build the target the net learns (`encode(dv*R)`, once at pre-encode) and (2) to
undo in the chi2 (`unwhiten(pred-target)/R`, every step); BOTH derive R from
`param_geometry.decode(whitened_params)`, so they use bit-identical R and the
giant array never exists.

`_analytic_R` made dual numpy/torch (one formula; library by input type; only
`log` + coercion differ). numpy path bit-identical to the old; torch path runs
on-device from the resident params. Verified numpy==torch to ~1e-15 (float64),
float32 within ~1e-6, R=1 at the reference (throwaway-venv test). `RescaledChi2`
holds the rescale config (param_geometry, cosmo_mid, names, include_amp, u_star),
caches the kept-element geometry tensors, and its `encode`/`decode`/`chi2`/`loss`
take the whitened params instead of a precomputed R (`loss` stashes them for the
inherited trim/focal reduction). Plumbing is branches only, gated on
`isinstance(chi2fn, RescaledChi2)`: `_build_loaders_one` pre-encode, the
`training_loop_batched` batch loop, and `eval_val` each pass the whitened params
they already hold; run_emulator/build_loaders unchanged (config rides on chi2fn),
base path A/B-runnable. Then the decisive test is frac>0.2 on the rescaled target
via the cut-scan -- the spread ratio is only a cheap proxy.

(The earlier "20 random val" `xi_resc` restructure bug is already fixed in the
current notebook -- the `xi_resc = rescale_xi(...)` line is present again.)

**Why:** records a validated, physically-motivated lever (and its ceiling) so we
do not re-derive it or re-litigate ratio-vs-difference; now that the integration
is done and verified, the next session goes straight to reading the frac>0.2
result rather than re-running the spread/visual checks or re-wiring.
