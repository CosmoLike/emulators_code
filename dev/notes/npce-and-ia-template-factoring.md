---
name: npce-and-ia-template-factoring
description: "2026-06-26 session: built NPCE (Neural-PCE, the user's paper method) and the GG+IA TEMPLATE FACTORING on pytorch1.ipynb. RESULT 1 -- NPCE is NOT useful for cosmic-shear xi: a PCE base only adds CAPACITY, it cannot lower the f(dchi2>0.2)~0.2 DATA floor, because a smoothness prior only helps an under-smoothed (overfitting) model and the ResMLP is already smooth; the PCE nails mode 0 (the amplitude, the easy part the net already gets) but the SHAPE modes are not low-degree polynomial (LOO 0.3-0.6), so it is right where the net wins and wrong where it struggles (NPCE-128 0.208 == ResMLP-256 0.212, capacity-equivalent). RESULT 2 (THE PICKUP POINT) -- the dv is EXACTLY a polynomial in the IA AMPLITUDES with cosmology-only template coefficients, so factor them out: emulate cosmology-only TEMPLATES, apply the known amplitude polynomial in CLOSED FORM in the loss; implemented as ARCHITECTURE on the EXISTING scattered samples (no extra sims, no structured extraction), amplitudes never enter the net -> exact, prior-width-independent generalization + FREE amplitude prior. Classes IAFactorGeometry / IATemplateMLP / IATemplateChi2 + nla_coeffs/tatt_coeffs. NLA validated NEUTRAL on the toy (0.234 == baseline; A1_1 is the easy direction on a narrow prior) -- the win SCALES WITH PRIOR WIDTH, so it is God-given for TATT's wide coupled amplitudes. WE FINISHED HERE."
metadata:
  node_type: memory
  type: project
---

Session 2026-06-26 built two emulator levers on pytorch1.ipynb
([[emulator-pipeline-and-goal]], [[emulator-sample-efficiency-is-the-goal]]).
The first (NPCE) was deprioritized; the second (IA template factoring) is the
PICKUP POINT for next session.

## NPCE (Neural-PCE, the user's arXiv 2404.12344v2 method) -- BUILT, DEPRIORITIZED

NPCE = a sparse-Legendre PCE makes the initial prediction, a NN refiner applies
a residual correction. Built: **PCEEmulator** (sparse Legendre PCE -- eqs 9-11
of the paper; map params->whitened dv via SVD MODES of the covariance-whitened
target = the eq-9 lambda_i; each mode amplitude a sparse Legendre polynomial;
candidate set = hyperbolic q-norm + max-interaction; selection = a self-contained
greedy LARS/OMP with a closed-form leave-one-out (PRESS) criterion -- no sklearn);
**PCEResidualChi2** (additive base: target = encode(dv) - PCE(theta), mirrors
ResidualBaseChi2); **PCERatioChi2** (multiplicative base: pred = PCE(theta)*(1+delta),
division-free, but it loses whitening + has zero-crossing leverage gaps and was a
step back).

**RESULT: PCE is NOT useful for cosmic-shear xi -- it only adds CAPACITY, it
cannot lower the data floor.** The deep reason (now a general principle):
- A smoothness prior only lowers a floor CAUSED by non-smoothness (overfitting).
  This floor is NOT: train==val means the saturated ResMLP is already a smooth
  interpolator -- no wiggle to suppress. The floor is COVERAGE + genuinely
  non-smooth HARDNESS (the As/Om h^2 nonlinear-structure direction), neither of
  which a smoothness prior can supply.
- The PCE's low-degree prior is right where the problem ISN'T and wrong where it
  IS: it nails mode 0 (the amplitude ~A_s/S_8, LOO 4.7e-3) -- the EASY part the
  ResMLP already learns -- but the SHAPE modes (1+) are NOT low-degree polynomial
  (LOO 0.3-0.6, need 150 wiggly terms). So it can only be a head start (capacity).
- Numbers: NPCE (PCE base + ResMLP width 128) = frac>0.2 0.208, vs bare ResMLP-128
  0.243 and ResMLP-256 0.212 -- i.e. ~= a width doubling = capacity-equivalent
  (the capacity-vs-bias confound; isolate at saturated width). Confirms
  [[emulator-sample-efficiency-is-the-goal]]: a reparam/representation change
  helps only if REPRESENTATION-limited, not DATA-limited.

**PCE build lessons (kept; reusable):** keep DEGREE LOW (p_max 3-6; degree 12
Runge-oscillated, every mode capped at max_terms, LOO 0.91, and the wiggly base
POISONED the refiner -- subtracting a wiggly base makes the smooth ResMLP's
residual HARDER not easier). Keep ONLY well-predicted modes (LOO < loo_max ~0.05);
a mediocre base mode injects more error than it removes. K-by-conservative-tail-
quantile was WRONG: keep FEW leading well-fit modes and let the refiner backstop
the rest (it corrects the full dv, not the K-mode subspace). Early-stop the mode
loop after max_fail consecutive gate misses (it was fitting 40 modes to keep 1).
The PCE FIT is CPU (numpy LARS, embarrassingly parallel over modes -> joblib +
BLAS-pinning if needed); the GPU is for the refiner. PCERatioChi2 recomputes the
frozen base every batch (slow) -> precompute at load + pack with target via a
new `target_dim` hook in _build_loaders_one (getattr(chi2fn,"target_dim",out_dim)).

## IA TEMPLATE FACTORING (GG + IA split) -- BUILT + validated on NLA. THE PICKUP POINT.

The exact-structure lever (the genuine version of "never emulate a dependence you
can write down", [[emulator-high-d-and-tatt-templates.md]]). The dv is EXACTLY a
polynomial in the IA AMPLITUDES with cosmology-only template coefficients:

    xi(cosmo, amps) = sum_t  c_t(amplitudes) * K_t(cosmo, photo-z, eta)

so the amplitudes factor OUT: emulate the cosmology-only TEMPLATES K_t, apply the
known closed-form polynomial c_t in the LOSS. The amplitudes never enter the net
-> EXACT generalization in them (free at inference, prior-width-independent), and
their prior costs ZERO training coverage.

**THE KEY DESIGN INSIGHT (resolves the "but you only get N/3 samples" fear):
implement the factoring as an ARCHITECTURE on the EXISTING scattered samples --
NOT as data preprocessing / structured 3-eval extraction.** The training samples
already have the amplitudes drawn from their prior (scattered). The model emits
the templates from the non-amplitude inputs; the loss reads each sample's OWN
amplitudes and applies the polynomial combine; chi2 on the combined xi. The
amplitudes varying across samples + the network's smoothness in cosmo IDENTIFY
the templates implicitly. So: same N sims, apples-to-apples vs baseline, no extra
data, no explicit template solve. (The explicit 3-eval-per-cosmology solve is only
worth it if GENERATING new data and you can afford 3x; for a fixed budget the
architecture is strictly better.) MUST exclude the amplitudes from the model INPUT
or the net absorbs their dependence and the exact guarantee is lost.

**The AMPLITUDE/POWER split (crucial, and the right TATT rehearsal):** amplitudes
that enter the IA field LINEARLY as coefficients factor out exactly; the
redshift-evolution POWERS (eta, entering as (1+z)^eta INSIDE the projection
integral) do NOT factor -- they stay emulated inputs. The notebook's LSST_A1_1 is
the NLA amplitude (factors out); LSST_A1_2 is the eta power (stays). TATT is the
same split: factor {a1, a2, b_TA}; keep {eta1, eta2} emulated.

**Template counts:** NLA = GG + GI(linear, 1) + II(quadratic, 1) = 3 templates,
coeffs [1, A1, A1^2]. TATT = GG + 3 GI (a1, a2, a1*b_TA) + 6 II (a1^2, a2^2,
(a1*b_TA)^2, a1*a2, a1^2*b_TA, a1*a2*b_TA) = 10 templates.

**Classes (general, IA-scoped naming -- the user's call: not TATT-specific since
they run NLA too; model-name lives on the coeff_fn):**
- `IAFactorGeometry` -- input whitening that DROPS the amplitude columns (whitens
  the rest from the covmat's sub-block) and APPENDS the raw amplitudes as the last
  n_amps columns. encode -> [whitened non-amp ; raw amps]; the model reads
  [:, :-n_amps], the loss reads [:, -n_amps:]. from_covmat(device, center,
  covmat_path, amp_names).
- `IATemplateMLP` -- ResMLP trunk on the non-amp input ([:, :-n_amps]); emits
  n_templates*n_keep, reshaped (B, n_templates, n_keep). (model spec carries
  n_amps, n_templates.)
- `IATemplateChi2(geom, coeff_fn, n_amps)` -- needs_params; _combine reads the
  last n_amps cols (physical amps), c = coeff_fn(amps) (B,n_templates), xi =
  einsum("bt,btk->bk", c, pred); chi2 = plain CosmolikeChi2 on the combined
  whitened xi. The combine is LINEAR so it COMMUTES with whitening; the training
  CENTER is absorbed into the GG (constant-coefficient) template automatically.
- `nla_coeffs` ((B,1)->(B,3) = [1, A1, A1^2]) and `tatt_coeffs` ((B,3)->(B,10)).
  Template ORDER is a coeff_fn convention (template 0 = GG carries the center).

**NLA RESULT on the toy = NEUTRAL (validates the machinery).** frac>0.2 0.234 ==
bare ResMLP-128 0.243 (within noise). EXPECTED: A1_1 is the easy/~0-hardness
direction on a NARROW prior, so factoring it gains little ON THE TOY. This is a
SUCCESS at validation (correct, regression-free, exact A1_1 generalization, one
fewer input dim). UNLIKE the PCE (a smoothness prior, can't help a data floor),
this is EXACT structure -- it just applies to a direction that wasn't binding.
**The benefit SCALES WITH the amplitude PRIOR WIDTH** (II ~ A1^2 swings hard over
a wide prior -> the baseline pays coverage, the factored model pays nothing).
Narrow NLA = neutral; wide coupled TATT = God-given (the joint amplitude coverage
cost -- exactly what makes TATT >> NLA -- goes to zero).

## NEXT SESSION -- WHERE WE FINISHED, WHAT TO PICK UP

1. **Create a HIGH-TEMPERATURE training set WITH TATT to stress-test the factoring
   code.** The toy is NLA-only (no a1/a2/b_TA/eta1/eta2 columns), so the TATT path
   (IATemplateChi2(geom, tatt_coeffs, n_amps=3), 10 templates) is BUILT but
   UNTESTED. Generate (cosmo + photo-z + eta1, eta2 + a1, a2, b_TA) -> xi at HIGH T
   with amplitudes sampled from their (wide) prior -- same structure as the NLA
   dump (amplitudes read per-sample, no extraction). Data generation is on the
   user's workstation via dataset_generator_lensing (see
   [[notebook-to-python-translation]]); cosmolike reviewed statically. First run
   NLA through the general classes (nla_coeffs, n_amps=1, n_templates=3) to confirm
   it reproduces 0.234 (certifies the generalization), then TATT.
2. Then begin the **notebook -> Python files translation**
   ([[notebook-to-python-translation]]).

**Why:** records that PCE/NPCE is a dead end for cosmic-shear xi (smoothness is
not the missing ingredient) and that the GG+IA template factoring (exact-structure,
implemented as architecture on existing samples) is the live forward lever, built
and NLA-validated, awaiting a high-T TATT dataset -- so next session goes straight
to stress-testing TATT, not re-deriving the design.
