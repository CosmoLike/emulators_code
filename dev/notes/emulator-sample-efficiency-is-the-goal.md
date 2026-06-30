---
name: emulator-sample-efficiency-is-the-goal
description: "The cosmic-shear emulator's REAL objective is SAMPLE EFFICIENCY -- the position of the learning curve f(delta-chi2>0.2) vs N_train -- not the asymptotic floor at unlimited data. The toy testbed (w0wa cosmic shear, ~82k vectors) is where methods are developed; the real target is T=512 + w0wa + TATT (~17+ params, hot prior, expensive cosmolike vectors) where N_train is the BINDING constraint and 'just add data' is infeasible. So every lever (analytic rescaling, architecture, physics input features, loss/conditioning, active sampling) is judged by whether it SHIFTS THE LEARNING CURVE LEFT (same accuracy, fewer samples), NOT by the floor at one fixed N_train. Corrects how rescaling etc. were judged this session."
metadata:
  node_type: memory
  type: project
---

The objective is **sample efficiency**, not the floor. The number that matters is
N_target = the smallest N_train at which a method reaches the target accuracy
(f(delta-chi2>0.2) < 0.10), or equivalently the accuracy at a fixed affordable
training budget. The deliverable is a method whose **learning curve** (f vs
N_train) is as far left / low as possible.

**SESSION RESULTS (2026-06-24) -- where this stands:**
- Plain ResMLP learning curve (val frac>0.2): 2k 0.51, 3.7k 0.33, 6.9k 0.245,
  10k 0.219, 46k 0.100 (goal), 82k ~0.06. Steep, still falling -> data-limited,
  NOT capacity ([[emulator-floor-is-data-coverage]]).
- TARGET RESCALING LOST the sample-efficiency test (narrow, one emulator): worse
  than plain at small N, converged by ~7k. Deprioritized. Mechanism = an
  UNCONFIRMED conditioning hypothesis, not a law -- see
  [[analytic-scaling-preprocessing]]. Do NOT over-generalize ("any reparam hurts"
  is false).
- UNTESTED live levers to try next (use the analytic WITHOUT touching the loss):
  (1) analytic dv as an INPUT feature / residual base; (2) PRETRAIN on cheap
  analytic dvs then fine-tune on cosmolike with the plain loss. Both keep the
  plain (well-conditioned) loss.
- Space-filling / maximin selection is WRONG here -- it pushes to the corners
  (the infeasible uniform) and fights the deliberate Gaussian-with-T_train,
  capped-correlation-0.25 sampling design. The sampling (T_train, cov shape) is
  the user's domain and is itself the volume-reduction strategy; do not replace it.
- CERTIFICATION discipline for the hard f<0.1 bar: f over a finite val set is
  noisy (binomial ~+/-0.015 at Nval~400) and best-epoch selection biases it LOW,
  so a single run at f=0.100 is NOT a certificate. Certify with a margin (e.g.
  target 0.085), several seeds (all must pass), and a bigger val set (sigma_f ~
  1/sqrt(Nval), free to enlarge).

**META-LESSON of this session (recorded against repetition): twice a sweeping
conclusion was drawn from one run -- "it's capacity" (refuted by the learning
curve) and "any target reparam hurts" (over-general from the rescaling). Size the
claim to the evidence: report the narrow empirical fact, flag the mechanism as a
hypothesis, and run the curve/experiment before writing a law.**

**SESSION 2026-06-25 -- where this stands (resume here).**
- OBJECTIVE SHARPENED (the user's framing, and it is the right one): the test is
  the CONVERGED PLATEAU of frac>0.2 at a FIXED useful N (~10k), NOT the small-N
  relative learning curve. A lower 10k plateau = reaching f<0.10 with fewer points.
  Small-N relative comparisons are useless if no method clears 0.10 up there.
- OUTPUT-REPARAMETRIZATION LEVERS ARE CLOSED. A (RescaledChi2) and B
  (ResidualBaseChi2) both came out == plain at a single converged ~10k run (~0.20),
  because they SHARE plain's optimum -- a converged single-N run CANNOT separate
  reparametrizations (conditioning shows only in the path / at small N, and even
  the paths matched). Stop running single-N A/B comparisons. See
  [[geometry-loss-composition]].
- THE PRINCIPLE for lowering a DATA-limited (train==val) plateau at fixed N: make
  the target EASIER TO GENERALIZE from few points -- reduce the effective DOF the
  net must learn, or inject structure. NOT reparametrization (A/B, same optimum),
  NOT capacity (data-limited), NOT regularizers (train==val -> variance already
  low -> neutral), NOT loss-shaping (exhausted; cannot narrow a spread).
- HARD CONSTRAINTS reaffirmed (do NOT propose these again): no changing the
  analytic R; no fine-tuning / pretrain (for now); no tighter cut -- the
  Gaussian-T sampling is fixed; sigma8 / early-universe combos are wrong lensing
  inputs and ParamGeometry already PCAs the inputs.
- A smoothness / anti-oscillation prior is OUT: the chi2 is a HIGH-PASS filter on
  the error (it already crushes oscillation; its blind spot is the smooth
  common-mode), so the penalty is redundant -- see the [[pytorch-teaching-style]]
  skill.
- PER-BIN DENSE-MLP SPLIT FAILED (2026-06-25): a per-bin ParallelResMLP came out
  WORSE than the single ResMLP (~0.27 vs ~0.24) -- it threw away the shared
  cosmology map and a dense MLP can't exploit output-axis structure anyway
  ([[per-bin-parallel-resmlp-plan]]).
- CNN HEAD CAME OUT NEUTRAL at T=16 (CORRECTED 2026-06-25b): the clean global
  ResCNN (appendix form, full whitening, fixed CNNBlock, gate) gives f(dchi2>0.2)
  ~=0.20 == plain ResMLP at 10k, gate near init. The earlier "0.24->0.18 worked"
  was CONFOUNDED (full->diagonal whitening + a broken CNNBlock). LEADING READ (the
  user's, and it is right): the T=16 toy is TOO EASY/narrow to show the
  theta-structured residual a conv fixes -- the value is at HIGH training
  TEMPERATURE (real target T=512), where the dv swings hard and the ResMLP leaves
  big axis-correlated residuals (the regime of the paper's 0.2->0.06). So the toy
  testbed may be the wrong instrument for output-side architecture levers. NEXT:
  rerun resmlp-vs-rescnn on a hotter training file ([[resmlp-cnn-perbin-architecture]]).
- LIVE LEVERS / NEXT (both keep the shared trunk): (a) the PER-BIN grouped-conv
  variant (ResCNNPerBin, built not run) -- removes the global conv's cross-bin
  smoothing; (b) the analytic INPUT feature (AugmentedParamGeometry, built not run,
  [[geometry-loss-composition]]). The NEXT experiment is the 3-model sweep
  (resmlp / rescnn / rescnn_pb) vs N_train -- judge by the SLOPE on log-log (a CNN
  curve sustaining its descent where ResMLP flattens = the win).
- ACTIVATION BAKE-OFF (2026-06-26, T=16, single seed, Nval~2000 so noise ~+/-0.01):
  H 0.244, Power 0.240, MultiGate K=3 0.227, Gated+Power K=3 0.235. Whole spread
  ~1.7sigma -> suggestive not proven. Pattern: the BULK-flexibility lever
  (multi-gate, a learnable slope-schedule activation) is the only one with a signal
  (~1.5-1.8sigma); the TAIL-power lever is idle at T=16 (features never reach their
  tails) and adding it to the multi-gate slightly HURT (0.227->0.235 = unused DOF =
  dead weight). Built three generalized activations of the paper's H(x) = gamma x +
  (1-gamma) Swish_beta(x): GatedActivation (K sigmoid gates, 3K+1 params/elem),
  PowerGatedActivation (H + bounded tail exponent p in [0.5,1.5] via a signed
  power psi_p(x)=sign(x)((1+|x|)^p-1)/p, 3 params/elem), GatedPowerActivation
  (both, 3K+2). All recover H at init. CONCLUSION: yet another model-side lever
  pinned to ~0.23-0.24 at fixed N -- reinforces the data-floor finding below.
  Carry MultiGate (only) to the high-T rerun (bulk flexibility should grow when the
  map gets hard); drop the tail power until high T exercises it.
- HARD-DIRECTION REGRESSION (2026-06-26) OVERTURNS the omega_b h^2 story. Fit
  log10 dchi2 ~ standardized ln(param/median). RESULT: ln(omega_b h^2) ALONE
  R^2 = 0.035 (NOT the hard direction -- it only looked hard via eyeball/Spearman
  because it shares H0). The real log-linear hardness direction is H0-led:
  univariate corr ln H0 +0.44, ln omegab -0.26 (NEGATIVE = more baryons -> EASIER),
  ln As +0.24, ln omegam +0.22, nuisances ~0. Joint coeffs ln H0 +0.28, ln As
  +0.17, ln omegam +0.16, ln omegab -0.08. Physically = "amount of NONLINEAR
  SMALL-SCALE STRUCTURE" (matter density Om h^2 x amplitude As up, baryon fraction
  down -> baryons suppress small scales) -- consistent with hardness=local curvature,
  and it is an AMPLITUDE/STRUCTURE direction, NOT a baryon-density one. BUT joint
  R^2 = 0.353: a log-linear param combo explains only ~35% of hardness; ~65% is NOT
  log-linear (nonlinear/interactions/coverage). CONSEQUENCES (revise the prior
  note): SCRAP the with-baryon analytic lever (omegab is anti-correlated, wrong
  target). The LOG-WHITENED ParamGeometry is now the right cheap test (it linearizes
  exactly the As/Om h^2 multiplicative direction -> should flatten the ~35%; cannot
  touch the 65%). Importance-sample HIGH-STRUCTURE regions (high Om h^2, high As),
  NOT high omega_b h^2. omega_b h^2 was the right CLIFF/CUT variable (catastrophic
  above 0.035) but NOT the within-region hardness gradient (that is H0/structure).
  META-LESSON: the eyeball omega_b h^2 scatter + single-param Spearman both pointed
  WRONG; the multivariate LOG-regression is the right localizer. Ran it before
  touching the baryon transfer -- good.
- WIDTH SWEEP at fixed N (2026-06-26, int_dim_res 32/64/128/256/512): f =
  0.583 / 0.322 / 0.243 / 0.212 / 0.211. Capacity-limited BELOW ~256 (steep drop),
  then SATURATES -- 256 vs 512 is 0.212 vs 0.211 despite 4x the params (0.71M vs
  2.48M) -> the ~0.21 floor is NOT capacity (third independent confirmation of the
  data floor, after the learning curve and omega_b h^2). IMPORTANT self-correction:
  width 128 -- the baseline for EVERY prior CNN/activation/loss experiment -- was
  slightly CAPACITY-LIMITED (0.243 vs 256's 0.212, a real ~2sigma gap). So those
  model-side "wins" at width 128 (MultiGate 0.227, CNN appendix ~0.235) were almost
  certainly CAPACITY, not better inductive bias: widening 128->256 buys 0.031,
  MORE than MultiGate's 0.017 nudge, so MultiGate ~ "plain at width ~180". At the
  SATURATED width 256 they would collapse back to ~0.21 = plain. ACTIONS: use width
  256 as the saturated baseline (~0.03 free over 128; 512 is wasted); the model-side
  search is now exhausted (even the activation/CNN nudges were capacity); levers
  stay data-side. Optional confirm: MultiGate K=3 at width 256 (bet: ~0.21 = plain).
- LOG-WHITENED INPUT REPARAM (LogParamGeometry, 2026-06-26) = NEUTRAL at T=16:
  linear inputs 0.200 vs log inputs 0.209 (within noise, same-seed width-256 A/B,
  both built from training samples so only the ln-transform differs). Logs As, H0,
  Omega_m, Omega_b; keeps n_s LINEAR (it is the exponent k^ns, not a multiplicative
  factor) and the DZ/A1 nuisances linear. WHY NEUTRAL (and a correction): the
  hardness regression found the hard DIRECTION is log-linear-predictable, but that
  does NOT imply log-INPUTS help -- a saturated-capacity net already represents the
  smooth As*(Om h^2)^ns dependence fine from linear inputs; the limit is DATA, not
  functional representation, so a coordinate change adds nothing. (Over-read the
  regression: "hard direction is log-linear in params" != "log inputs lower error".)
  Contributing: at the narrow T=16 prior, ln(param) ~ affine in param, so after
  whitening log ~ linear -- the transform is nearly a no-op (ln only bends over a
  WIDE range = high T).
- TOY EXHAUSTED (2026-06-26): EVERY model- and representation-side lever is NEUTRAL
  at T=16 -- architecture (CNN), activations (H + 3 generalizations), capacity
  (width, saturates 0.21), loss (A/B rescale, element-weight, focal, trim), input
  reparam (log) -- all for ONE reason: the toy is too narrow/easy to exercise them
  and the floor is DATA. Stop testing levers at T=16; they are all moot here. The
  ONLY remaining real experiment is to regenerate these comparisons at HIGH T
  (toward 512), where ln actually bends, the structure direction carries large
  errors, and N is the true binding constraint. Sampling note (also moot at T=16,
  do it when generating data): draw the SAME Gaussian-T from a low-discrepancy
  (scrambled Sobol -> probit -> Cholesky) sequence to kill i.i.d. clumping = free
  coverage; active learning is NOT worth it here (failures are DIFFUSE, no
  concentrated target -> degenerates to uniform), so when difficulty is diffuse the
  lever is REPRESENTATION not sampling.
- THE FLOOR IS A SAMPLE-COMPLEXITY (DATA/COVERAGE) FLOOR -- the robustness IS the
  proof. f(dchi2>0.2) ~= 0.20-0.24 at fixed N=10k is INVARIANT to architecture
  (CNN), activation (H + 3 generalizations), loss (A/B rescale, element-weight,
  focal, trim), and conditioning, but moves sharply with N_train (0.22@10k ->
  0.10@46k -> 0.06@82k). A metric invariant to every model-side change but moving
  with N is data-limited by definition. Mechanism = ESTIMATION error not
  approximation: train~=val + median 0.04 say the model fits fine; f>0.2 = the
  fraction of val cosmologies in parameter regions too SPARSELY covered by training
  points to pin the dv to dchi2<0.2 (nearest-neighbour distance ~ N^{-1/d}). You
  cannot narrow that broad log-normal error spread with model tweaks, only with
  denser data. RECONCILES the high-T hope: sample-complexity ~ function-complexity
  / inductive-bias-informativeness; at T=16 the map is so easy you are already at
  the coverage floor (no bias helps); at high T the map is hard so a good bias
  (CNN/activation/physics prior) lowers the data needed -> that is where the paper's
  architecture wins live. The two levers that move a data floor are both data-side:
  reduce effective DOF per point (physics priors / input features) and place points
  where needed (active/importance sampling). COVERAGE DIAGNOSTIC delivered (k-NN
  distance from each val point to the training cloud in whitened pgeom space vs
  dchi2; spearman>0 + bad-points-right-shifted-histogram + sparsest-decile frac >>
  densest-decile = coverage confirmed; then the Run-test-3 triangle says cluster
  (importance-sample) vs scattered (uniform N)).
- COVERAGE DIAGNOSTIC RESULT (2026-06-26, T=16) -- FINAL (numbers settle it; the
  read swung twice, land in the MIDDLE). The DECILES are decisive, not the
  histograms: frac>0.2 = 0.130 in the DENSEST decile vs 0.370 in the SPARSEST
  (2.8x), spearman(knn_dist, log dchi2) = +0.317. So COVERAGE IS A REAL, SUBSTANTIAL
  driver. The good/bad histograms look ~identical (and the median barely moves:
  good 1.746 vs bad 1.822) only because a 0.32 correlation genuinely overlaps
  heavily -- failure prob rises NON-LINEARLY with sparsity (flat in the bulk, steep
  in the far tail), so the extremes separate cleanly while the median does not.
  LESSON: trust the DECILES over eyeballing overlapping histograms; a marginal
  histogram + small median shift can still hide a decision-relevant correlation.
  (Earlier swing to "not coverage" was an over-correction.) BUT the densest decile
  STILL fails 13% -> a residual HARDNESS floor coverage cannot fix. So the floor
  DECOMPOSES, now sized: COVERAGE (sparse regions fail more, ~0.22->~0.13 headroom)
  + HARDNESS (residual ~0.13 even at max density). LEVERS: importance-sample toward
  LOW-DENSITY regions (high-leverage -- they carry the failures; reaches ~0.10 with
  fewer than 46k uniform) for the coverage part; physics prior / error-targeted
  sampling for the residual hardness part (need both to clear 0.10). Next
  diagnostic to size the hardness part: per val point, the variation of the TRUE dv
  among its nearest training neighbours (local curvature) -- does it separate the
  residual failures the densest decile still has?
- LOCAL-LINEAR "HARDNESS FLOOR" TEST FAILED as an instrument (2026-06-26) -- DO NOT
  reuse it. Idea was: fit a local linear params->target map over each val point's
  k=40 nearest TRAINING points, predict the val target, call its chi2 the
  "data-only floor", compare to the model. RESULT: floor median chi2 = 94,
  frac>0.2 = 1.000 (fails everywhere); model median 0.076, frac>0.2 0.235 -- the
  model BEATS local-linear by ~1000x (every point below the diagonal). The
  printed verdict ("at the DATA floor") is a MIS-FIRE: the ratio test
  f_model<=1.3*f_floor triggers trivially when f_floor=1.0; it lacked the case
  f_model<<f_floor = "baseline too weak to bound the model." LESSON: a local
  model-free interpolator CANNOT be a floor for a strong GLOBAL learner -- the
  ResMLP exploits global smoothness + physics-informed inductive bias to fit a map
  that is violently nonlinear over the inter-point scale (chi2-metric) but globally
  smooth, so it trounces any local method. REAL takeaways: (1) the model is a
  genuine global learner, not a local interpolator (good); (2) the right
  data-limit evidence stays the LEARNING CURVE (0.22->0.10->0.06 with N) +
  architecture/activation invariance at fixed N -- both already in hand and both
  say data-limited-at-fixed-N with no model-capacity headroom; (3) the physical
  address of the hardness -- CORRECTED below (it is NOT omega_b h^2).

**Why (the binding constraint at the real scale).** The notebook
([[emulator-pipeline-and-goal]]) is a TESTBED: w0wa cosmic shear, ~12 params,
~82k data vectors available -- here data is cheap, and the toy floor is
data-limited (f hits 0.10 at 46k of 82k, see [[emulator-floor-is-data-coverage]]),
so on the toy "more data" works. The REAL target is **T=512 + w0wa + TATT**: the
TATT intrinsic-alignment model adds ~5 params (a1, a2, b_ta + z-evolution) on top
of w0wa cosmology + photo-z + biases (~17-20 total), and T=512 is a very hot
sampling prior, so the parameter VOLUME explodes. Each cosmolike 3x2pt+TATT vector
is expensive to compute. So N_train is THE binding constraint -- dense coverage
needs a ridiculous number of sims, and "just add data" (the toy answer) is dead on
arrival. The asymptotic floor at unlimited data is irrelevant when you can never
afford unlimited data.

**The correction this forces (how to judge every lever).** A method must be judged
by whether it shifts the learning curve LEFT (same accuracy, fewer samples), not
by the floor at one N_train. This session judged the levers on the wrong axis:
the rescaling, the per-element weight, etc. were each tested at the ~10% subset
(10k) and called "neutral on the floor." That is the wrong test. A lever neutral
at the floor can still be a large sample-efficiency win -- which is precisely the
regime that matters when each datum is costly.

**The rescaling is the prime sample-efficiency lever.**
[[analytic-scaling-preprocessing]] removes ~79% of the broadband variance the net
would otherwise learn from data, so it should need fewer samples for a given
accuracy. Its benefit is LARGEST at small N_train (scarce data) and fades as N
grows (the net learns the variance itself). So the decisive test is the
rescaled-vs-plain LEARNING CURVE at SMALL N (2k-10k), where the curves separate --
not the 10k subset where they looked similar. Same logic for architecture /
inductive bias, physics input features / residual learning, and target
conditioning: all are candidate left-shifts of the curve.

**The experiment (the reference-figure structure).** The Dot/Linear/LSH/AFT/Latent
plot (f vs N_train/1000, log-y) is exactly a method comparison by learning curve --
the leftmost/lowest curve is the most data-efficient and is the one to scale to
T=512+TATT. Build the multi-method loop over LOG-spaced N_train including the small-N
regime; report N_target per method. Active / importance sampling (where to PLACE
the training points, not just how many) is the other axis at the real scale, since
uniform sampling of a hot high-D prior is hopeless.

**Why:** records that the goal is the learning curve's position (sample
efficiency), so future work compares methods by N-to-target on the toy testbed and
does not mistake "the toy floor is data-limited" for "the project is solved by more
data." The whole point is to make the model learn more from fewer expensive sims.
