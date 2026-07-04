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

**CONV HEAD REDESIGNED: BINS-AS-CHANNELS (2026-07-04d, user: "CNN is
too slow... remove channels key").** The channels knob, CNNBlock, and
TemplateMixCNNBlock are DELETED. ResCNN's head is now: theta-order dv
-> pad_idx scatter into the padded (n_bins, max_bin) layout (same
machinery as ResTRF; ResCNN/TemplateResCNN now carry needs_bins) ->
n_blocks_cnn x [ONE Conv1d(n_bins -> n_bins, kernel_size) + act] ->
gather -> W_df -> gate. TemplateResCNN: channels = the (template, bin)
pairs, Conv1d(T*G -> T*G, k) -- cross-bin AND cross-template in one
single kernel. Head hyperparameters are ONLY kernel_size +
n_blocks_cnn. Why it kills the speed problem BY CONSTRUCTION: the old
head expanded to C=16 filters ((3B, 16, 705) = 104 MB intermediates,
bandwidth-bound); the new head's tensors never exceed the padded dv
size ((B, 90, 26) = 7 MB at bs 768) -- no expansion exists to pay for.
Physics: channel mixing couples different bins at like angular scales
(up to per-bin mask offsets). Zero-init = the LAST conv (identity
start; plain ResCNN now has it too; set_train_phase / two-phase stays
on the Template variants only). Params per block: C^2*k + C (plain
G=30: ~9.9k; nla T*G=90: ~89k at k=11). Old test files
test_rescnn_nla/test_mix_and_phases superseded by test_rescnn_bins.

**ResTRF BUILT (2026-07-04c, user-commissioned; 32 venv checks).**
The bin-token transformer architecture, name: restrf (+ ia: nla ->
TemplateResTRF). Gated correction appendix like rescnn (user: "lets
try to maintain this... if it does not work we can think about trying
the other way" -- the paper's main-path form is the fallback). Design:
trunk -> W_fd theta order -> pad_idx scatter into the padded
(n_bins, max_bin) layout (bin_sizes via build_shear_angle_map; the
needs_bins capability flag makes build_geometry run it -- ini+n(z)
only, no cosmolike) -> per-bin UNIQUE embed (BinLinear) -> n_blocks_trf
x TRFBlock -> per-bin UNIQUE out (zero-init identity) -> gather ->
W_df -> gate. TRFBlock = pre-LN attention across bins (Q/K/V/O SHARED,
standard) + per-bin UNIQUE MLP stack (n_mlp_blocks deep) -- the user's
two deviations from the textbook block; unique weights replace the
positional encoding. ia: nla -> token features = the bin's segment
from ALL 3 templates concatenated (T*max_bin -> int_dim_trf), the TRF
analogue of templates-as-channels; A1 exactness untouched; per-template
gates; set_train_phase -> trunk_epochs two-phase works. Knobs:
int_dim_trf (divisible by n_heads), n_heads, n_blocks_trf,
n_mlp_blocks, gate_init. conv_head flag RENAMED needs_geom (+ new
needs_bins). PARAM NOTE: the head is ~200k at 30 bins/d32 (per-bin
unique weights x 30 dominate -- embed/out/MLPs), vs ~3k for the conv
head; compute still tiny (tokens (B,30,32), NOT bandwidth-bound).
GOTCHA (test trap): pre-LN LayerNorm is shift-invariant per token, so
a CONSTANT perturbation of one token is annihilated -- probe mixing
with a random vector.

**SIMPLIFICATIONS (2026-07-04b, user-driven):** (1) template_mix knob
DELETED -- templates-as-conv-channels is now THE TemplateResCNN head
(no fold path; user: "win-win", and the shared-kernel regularization
only matters at tiny N). Zero-init target is always the mix block's
collapse conv (the channels==1 Identity edge case is gone). A stale
`template_mix:` YAML key now raises TypeError (unexpected kwarg).
(2) head_lr_base REPLACED by a unified train_args.head block of
head-phase overrides: lr_base / loss_mode / trim / focus (trim+focus
are FULL replacement blocks incl. kappa, restarting at the head
phase's epoch 1; rationale: post-handoff there are few outliers, so
e.g. loss_mode chi2 + no trim). head: without trunk_epochs>0 raises
(silent-no-op trap). run_emulator signature: trunk_epochs, head_opts.

**CONFIG SCHEME REDESIGN (2026-07-04, user: "this naming is bad").**
train_args.model.name is now the ARCHITECTURE ONLY (resmlp | rescnn);
a separate model.ia key layers the factored IA design (absent/None =
plain; "nla"; "tatt" reserved). MODELS is keyed by (name, ia) tuples;
IA_DESIGNS = {"nla": {amp_names, coeff_fn, n_templates}} centralizes
the per-design data (tatt = one new entry when its dumps exist).
exp.model_name = composed display name ("rescnn_nla") -- run_tag
FILENAMES ARE UNCHANGED. exp.ia drives the design lookups; direct
construction infers ia from the factored flag. The old one-key names
(nla, rescnn_nla) now ERROR with a message teaching the split. YAML
`ia: none` (string) == absent. build_specs strips "ia" like
"name"/"activation". ALSO DELETED (same session, user: "I will never
do a ResMLP in parallel per redshift bin"): ParallelResMLP +
GroupedLinear/GroupedAffine/GroupedResBlock +
parallel/activations.py(GroupedActivation); parallel/ keeps ONLY
ParallelResCNN + GroupedCNNBlock (shared trunk, per-bin conv).

**CODE DELETED (2026-07-04, user: "we know this is a terrible case").**
Removed: the nla_as registry entry + NLA_AS_AMP_NAMES + wiring branches
(experiment.py), AsScaledNLAChi2 (IA/loss_functions.py), and the whole
carry_idx/carry_names mechanism in AmplitudeFactorGeometry (it existed
only for nla_as; names + encoded_dim + state round-trip KEPT -- they
serve nla/rescnn_nla and save_emulator; encoded_dim now always ==
n_param). Do NOT reference nla_as code paths -- only this note's
physics lesson survives. Old .h5 saves with a stale carry_idx key
still load: from_state reads only its named keys, so the extra key is
simply never touched (verified: encode/decode/state round-trip green
post-removal).

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
off-workstation). YAML: model.name: rescnn_nla + uncomment the
conv-head knobs. Judge vs nla 0.1472 at matched T=256/250k; the bet is
the dense-decile residual (0.122, untouched by nla) is theta-structured.

**SPEED DIAGNOSIS (2026-07-04 runs on the 3060): the head is
MEMORY-BANDWIDTH-bound, not FLOP- or launch-bound.** Fold mode's
intermediates are (3*bs, channels, n_keep) -- ~104 MB each at bs 768 /
ch 16 / K 705 -- and each CNN block moves ~1.4 GB/step fwd+bwd; the
3060's ~330 GB/s makes head s/epoch scale as
(3 if fold else 1) * channels * n_blocks_cnn. Observed: 128w/ch8/1blk
default = 3.5 s/epoch; 96w/ch16/2blk reduce-overhead = 5.8 -- the jump
is the 4x head, NOT reduce-overhead failing (it no longer crashes on
this torch; keep it). nla baseline 0.8 s/epoch.

**FIXES BUILT (2026-07-04, all tested, 52 venv checks):**
(1) model.template_mix: true -- templates become the conv's input
CHANNELS (Conv1d 3->C->3 on (B,3,K)) instead of folding into batch:
identical FLOPs (3x moved from rows into kernel depth), 1/3 the
traffic (16 JOINT feature maps vs 48 per-template ones -- sharing
weights never shrinks activations, so this is the only structural way
down), cross-template features; A1 exactness untouched (it lives in
the loss combine). Trade: drops the one-shared-kernel inductive bias.
(2) Zero-init identity head (unconditional): the LAST cnn block's
output layer is zeroed, H(0)=0 -> corr==0 at init, model == trunk
exactly; gradient wake-up chain: collapse live at step 1 (through
gate!=0), conv+gate wake at step 2. Supersedes the gate-must-not-be-0
concern.
(3) train_args.trunk_epochs: N -- the user's two-phase schedule:
phase 1 (1..N) head BYPASSED entirely (set_train_phase("trunk"), pure
nla cost ~0.8 s/epoch); phase 2 trunk frozen AND under no_grad (no
trunk backward), head-only training from the identity start ->
loss-continuous handoff. run_emulator orchestrates as TWO
training_loop_batched calls (each restores its best + own
warmup/opt/sched/trim/focus cycle; phase 2 starts from phase 1's BEST
trunk automatically); histories concatenate. set_train_phase is
duck-typed (hasattr through the compile wrapper); guards: trunk_epochs
< nepochs (fails at top), model must define set_train_phase (fails
after make_model). Banner prints "(two-phase: N trunk + M head)".
