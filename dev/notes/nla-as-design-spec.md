---
name: nla-as-design-spec
description: "IMPLEMENTED 2026-07-03 (verified end-to-end in the venv: carry widths, state round-trip, scaled-center encode, pure-Ats combine == manual, exact physical residual, backward; all four gotchas below handled: carry in AmplitudeFactorGeometry + state; encoded_dim property + run_emulator getattr injection; exp.model_name set by from_config; AsScaledNLAChi2 in IA/loss_functions). Run with train_args.model.name: nla_as. ORIGINAL SPEC (user-approved): model.name nla_as = the NLA factoring PLUS the linear-order As amplitude factored out. Coefficients [Ats, Ats*A1, Ats*A1^2] with Ats = As/as_ref (as_ref = training-mean As, an invisible constant the templates absorb; O(1) conditioning only). THREE templates, NO constant term -- the center stays but is SCALED PER SAMPLE: encode = whiten(squeeze(dv) - Ats*center) (the B-form trick), so dividing the matching condition by Ats gives templates = w(dv)/Ats - w(c), pure proportionality, O(1) targets at Ats~1. As is CARRIED not factored (user's caveat: halofit makes the dv nonlinear in As, so As stays a whitened model input; only the linear-order amplitude factors -- a strong prior, not an identity like A1). Geometry: AmplitudeFactorGeometry gains carry support (carried amps appended raw for the loss AND kept whitened in the input block; factored amps dropped). Appended order [As_1e9, LSST_A1_1] (As column name in the dumps is As_1e9). IMPLEMENTATION GOTCHAS FOUND IN ADVANCE: (1) encoded width != raw C width once an amp is carried (13 vs 12) but run_emulator injects input_dim = train_set['C'].shape[1] -> add an encoded_dim property to the geometry and make run_emulator use getattr(param_geometry, 'encoded_dim', C.shape[1]); (2) MODELS maps both 'nla' and 'nla_as' to TemplateMLP so `model_cls is TemplateMLP` cannot distinguish them -> store self.model_name on the experiment (from_config sets it from the YAML name) and branch on the name; (3) carry_idx must join state()/from_state (the AmplitudeFactorGeometry round-trip just added); (4) the loss class (AsScaledNLAChi2(TemplateFactoredChi2), n_amps=2, as_ref from train_set['C_mean'] at the As column) overrides encode/decode for the Ats-scaled center and defines the coeffs internally; conditioning caution: Ats multiplies the net output = the A-form gradient family that lost on the toy -- judge vs the nla baseline 0.1472 in one matched run."
metadata:
  node_type: memory
  type: project
---

User-approved design for `model.name: nla_as` (2026-07-03). NLA result it
builds on: nla beat resmlp 0.1472 vs 0.1558 (median 0.0467 vs 0.0531) at
matched everything, T=256, 250k.

**The math (agreed in conversation):**
- Limber: C_ell = As * G(shape) at linear order, so the linear amplitude
  factors exactly; halofit breaks global As-proportionality, so As STAYS
  a whitened input (user's caveat) -- the factoring is a strong prior,
  unlike the exact A1 polynomial.
- Coefficients [Ats, Ats*A1, Ats*A1^2], Ats = As/as_ref; as_ref =
  training-mean As, an invisible constant absorbed by the templates,
  kept only for O(1) output conditioning.
- NO constant template. The center stays, scaled per sample:
    encode(dv, params) = whiten(squeeze(dv) - Ats * center)
  so templates represent w(dv)/Ats - w(c): pure proportionality, O(1)
  targets near Ats = 1, chi2 exact (the offset cancels in the residual).
  This is the ResidualBaseChi2 "B-form" move applied to the center.
- The whitened-space combine stays exact: whitening is linear, scalars
  commute; only the (affine) center needed the special handling above.

**Wiring plan + pre-found gotchas:** see the description block (encoded_dim
injection, model_name-vs-class disambiguation, carry_idx in the state
round-trip, AsScaledNLAChi2 shape).

**Why:** the spec was negotiated in detail (constant-term inconsistency
caught by the user and resolved via the scaled center); this note lets the
implementation start cold without re-deriving any of it. Pairs with
[[npce-and-ia-template-factoring]] and [[omegam2h2-window-cut]] (OUTCOME
section carries the nla-vs-resmlp numbers).

**POST-RUN INSIGHT (nla diagnostics, 2026-07-03): LSST_A1_1 did NOT
vanish from the hardness ranking (still 5th, ~unchanged), and that is
CORRECT behavior, not a bug: the error is dxi = dK1 + A1 dK2 + A1^2 dK3,
so template errors are AMPLIFIED by |A1|. Factoring converts the A1
problem from axis COVERAGE (scales with prior width, the TATT killer)
to template-error AMPLIFICATION (shrinks with N). The right success
signatures are the metric gain (0.156 -> 0.147) and the sparse-decile
improvement (0.381 -> 0.346), not a zero A1 correlation. Expect the
same shape for TATT: amplitudes stay in the hardness ranking while
their prior stops costing coverage.**

**VERDICT (2026-07-03 run): ABANDONED, code kept.** nla_as frac>0.2 =
0.1559 (== resmlp 0.1558; nla alone 0.1472), median 0.0464. The Ats
scaling ERASED the nla gain, and hardness joint R^2 jumped 0.18 -> 0.42:
errors became strongly As-directional (dxi = Ats * dK amplification),
the A-form conditioning failure the loss-family history predicted. The
exact-A1 half works; the approximate-As half hurts.

**rescnn_nla BUILT (2026-07-03, awaiting first run).** TemplateResCNN in
emulator/IA/emulator_designs.py: TemplateMLP trunk emitting the 3
templates + ONE shared gated CNNBlock stack correcting each template in
theta order before the loss combines them (templates fold into the batch
axis, so the conv learns from 3x the examples; per-TEMPLATE gates,
(n_templates, 1), init 0.1 -- GG/GI/II have very different whitened
magnitudes). Amplitude polynomial untouched -> the correction inherits
the exact-A1 generalization. Basis handling = ResCNN's W_fd/W_df frozen
buffers (CUDA-graph safe); act_mid gotcha handled by CNNBlock itself.
Wiring: the isinstance checks in experiment.py became CAPABILITY FLAGS
on the model classes -- factored=True (TemplateMLP, TemplateResCNN)
picks AmplitudeFactorGeometry + the template-combining loss;
conv_head=True (ResCNN, TemplateResCNN) injects geom AND setdefaults
compile_mode="default" (so rescnn/rescnn_nla no longer crash under
reduce-overhead unless the YAML overrides). The rescale guard moved
ABOVE build_geometry's lazy cosmolike import (fail fast + testable
off-workstation). 19/19 venv tests pass (buffers invert, fold ==
per-template loop, gate=0 == trunk exactly, grads reach all gates,
state round-trip, from_config/build_specs injections, make_model
end-to-end, combine+backward). YAML: model.name: rescnn_nla + uncomment
the conv-head knobs (kernel_size 11 / channels 16 / n_blocks_cnn 1 /
gate_init 0.1). Judge vs nla 0.1472 at matched T=256/250k; the bet is
the dense-decile residual (0.122, untouched by nla) is theta-structured.
