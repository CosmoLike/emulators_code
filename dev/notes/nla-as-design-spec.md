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

**NESTED MODEL-BLOCK SCHEMA (2026-07-04e, user: "I dont like all
hyperparameters on the same level").** The YAML model block now nests
one sub-block per component, so keys need no suffixes:
  model: {name, ia, mlp: {width, n_blocks}, activation: {type,
  n_gates} (bare string accepted as type shorthand), cnn:
  {kernel_size, n_blocks, gate_init}, trf: {width, n_blocks,
  n_mlp_blocks, gate_init}, compile_mode}.
Constructors KEEP flat kwargs (internal API); build_specs translates
via MODEL_BLOCK_KEYS tables (mlp.width -> int_dim_res, cnn.n_blocks ->
n_blocks_cnn, trf.width -> int_dim_trf, ...). Validation: unknown
top-level keys, unknown keys inside the ACTIVE blocks, old FLAT keys,
and a missing mlp block raise loudly -- but the INACTIVE head's block
is silently IGNORED (contents not even validated), per the user:
keeping cnn: and trf: both configured lets a run switch architectures
by changing name: alone (ARCH_HEAD: resmlp->None, rescnn->cnn,
restrf->trf; exp.arch set by from_config, None = direct construction
= every present block translated). n_heads IS a documented trf: hyperparameter (user
explicitly corrected an earlier removal: "dont hardwire - make it a
hyperparameter"; default 4). The multi-head lesson stands: H is free
(the H-fold pattern count is exactly canceled by d/H-narrow dot
products; only the (B,H,G,G) score tensor grows, megabytes) -- it
tunes inductive bias (how many parallel in-what-respect attention
patterns), not capacity. gate_init lesson (same session): with the zero-init
identity head, 0.1 is a SOFT-START BRAKE only (under Adam the
correction ramps ~10x slower in output units; gate=0 is a true
deadlock -- gate grad prop to corr=0, head grad prop to gate); for
short phase-2 runs gate_init 1.0 is the better default to try.

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

**ResTRF ADAPTERS DELETED (2026-07-04h, user: the paper's embed/output
layers were its parameter-heaviest pieces and existed only because its
TRF lived on a SYNTHETIC latent sequence -- ours is physical).** No
embedding in, no projection out: tokens are the raw padded bin
segments at NATURAL width max_bin (plain: 30 bin tokens; nla: the
(template, bin) pairs = 90 tokens, mirroring the conv's
pairs-as-channels). The width knob is GONE (trf: has only n_heads /
n_blocks / n_mlp_blocks / gate_init; a stale trf.width errors).
n_heads must divide max_bin (26 -> 1|2|13; default 2). Identity start
moved INTO TRFBlock: both branch outputs (wo + last MLP BinLinear)
are zero-initialized so every block == identity at init, and the
models define corr = blocks(h) - h == 0 at epoch 1 (no out layer to
host the zero-init). Head params HALVED: plain 119k -> ~45k; nla 221k
-> ~110k at real scale (90 tokens x width 24; MLPs dominate, 2x90x
(24^2+24)). BinLinear/TRFBlock params renamed n_bins -> n_tokens.
trf.shared_mlp flag (2026-07-04i, user request): true = ONE textbook
MLP shared by every token (plain nn.Linear position-wise) -- the
ablation isolating the unique-MLP deviation; params drop by another
factor n_tokens on the MLPs. CAVEAT (documented): with shared MLP +
shared attention the head is permutation-equivariant over tokens (the
unique weights WERE the positional encoding); token identity then
comes only from segment content. Default false (unique).

**GENERIC HYPERPARAM SWEEP + GPU PACKING + PARALLEL OPTUNA
(2026-07-04y; user: "drivers to go over things like batch size or
activation functions... a YAML file to decide which hyperparameter"
+ "make sure all these drivers work on multiple GPUs" + the H200
multi-training-per-GPU question + "is the Optuna driver ready for
multiGPU? IF not - please make it").**
(1) NEW DRIVER sweep_hyperparam_emulator_cosmic_shear.py: sweeps
exactly ONE train_args leaf named by a dotted path in a YAML
`sweep:` block (parameter + values; block-style example commented
in the train YAML). Any leaf: bs, lr.lr_base, trim.start,
model.cnn.kernel_size, model.cnn.film (bools), head.lr_base
(missing blocks created; the trunk_epochs guard still fires).
Special case model.activation(.type) sets exp.activation per value
(build_specs reads the family off the experiment, NOT train_args
-- a train_args copy would silently no-op); model.name/ia REFUSED
(class change); unknown first segments REFUSED (exp.train .gets
top keys, so a typo'd path would silently train the same config N
times -- SWEEPABLE_TOP_KEYS guards). set_by_path = deep-copy +
dotted set. Data staged ONCE per worker (unlike ntrain). Outputs:
save_sweep_table (results.py; numeric = value/frac columns,
categorical/bool = index column + "# values: 0=H, 1=power" map,
np.loadtxt-safe either way) + plot_sweep_curve (plotting.py;
numeric line, log-x when positive span > 20x; categorical labeled
markers; log-y only when all fracs > 0).
(2) MULTI-GPU MACHINERY FACTORED into scheduling.py: even_assign
(round-robin, equal-cost) + run_gpu_pool (spawn, one process per
(GPU, lane), per-GPU job queues, setup_fn-once/job_fn-per-job,
parent drains with timeout + liveness -> a dead worker raises
instead of hanging; "__lane_failed__" names setup errors). BOTH
sweep drivers now run through it (sweep_ntrain rewired:
_sweep_worker/_run_parallel -> _sweep_setup/_sweep_job + pool).
PYTHON 3.14 TRAP found + fixed: Process releases its args after
start(), so loop-created Queues/Locks/Semaphores got GC'd -> named
OS semaphores unlinked -> children died in SemLock._rebuild
(FileNotFoundError); the pool keeps a keepalive list until join.
(3) --gpu-pack (OFF by default, both sweep drivers; user's rule
verbatim): estimate each training's VRAM fraction
(estimate_train_vram_fraction: 2*N*dv_width*4 bytes -- targets +
pre-shuffle transient, dv_width the on-disk width = upper bound --
+ 2 GiB fixed overhead) -> vram_tokens: <=20% -> 1 token (4 share
a GPU), <=40% -> 2, else 4 = exclusive, of GPU_TOKENS=4 per card.
Token grabs serialized by a per-GPU lock (multi-token deadlock
impossible: releases need no lock). Engages even on ONE visible
GPU (the lone-H200-allocation case). Numbers: 250k x 1560-wide dv
~ 3.5% of an H200 (packs 4-way) but ~41% of a 3060 (exclusive) --
matching the user's don't-pack-on-amypond requirement even if the
flag is accidentally on. Underestimates degrade to streaming (the
loaders size against real mem_get_info), not crashes; co-located
timing numbers are NOT comparable to exclusive runs (documented).
(4) tune_single MULTI-GPU: --n-gpus + --journal; parent creates
ONE study in an optuna JournalStorage file under --fileroot
(journal_storage() shims the 3.x/4.x backend rename), enqueues the
warm-start only when the study is fresh, splits --n-trials, spawns
one _tune_worker per GPU (own exp + staging, TPESampler(seed=
gpu_id) -- same seed would duplicate proposals), all cooperating
through the shared journal; parent reloads and prints the best.
Same journal name RESUMES the study; serial path unchanged
(in-memory, no file). ram_frac divided by workers (each stages the
SAME subset -> P private copies, the parallel shared-budget trap).
(5) TESTS: test_gpu_pool_pack.py (21 checks: token thresholds,
estimate math + H200-vs-3060 sanity, even/lpt_assign, set_by_path
+ read_sweep_block validation incl. refusals, pool on CPU with
timing-proven token exclusivity (exclusive job overlap -0.00s,
1-token pair +1.00s) + loud setup failure, numeric/categorical
table + figure round-trips, and 2 spawned processes sharing one
journal study -- 8/8 trials, warm start honored). Suite total 17.
README (tree, orchestration graph, driver tables, examples,
appendices) + example YAML sweep: block updated. DOCS PASS (user:
"write documentation on these new settings. And create example
YAML"): README section 6 grew two anchored subsections -- "The
sweep: block" (rules table + outputs) and "Multi-GPU execution and
packing" (driver x split table, the token ladder, when to use
--gpu-pack and when not, the journal resume semantics, the MPS
note); NEW example_yamls/sweep_hyperparam_emulator_cosmic_shear.yaml
(active lr sweep + five commented swap-in sweeps: bs, activation
family, film on/off, cnn depth, head lr; compact train_args
pointing at the train YAML for full key docs); tune YAML header
now documents --n-gpus/--journal + resume. GPU-side verification
(real co-located trainings, journal study on 2 cards) happens on
amypond -- the Mac has no CUDA.**

**DRIVER PARITY + SUBPACKAGE CURRENCY PASS (2026-07-04x,
post-compact; user: "make sure sweep/tune are as updated as
train_single" + "go over all files... formal and explicative and
with plenty of graphs" + "PCE needs update... make sure both
parallel/ and PCE/ work with all updates").** (1) The startup
banner moved into EmulatorExperiment.print_design() (model spec /
run line + two-phase split / guards clip+rewind / every sub-block
incl. trunk:/head: / cuts; quiet-gated) and ALL THREE drivers call
it -- a stale YAML now dies at launch of a sweep or study, not one
training in. train_single's own header was itself stale (missing
tatt, the 5 cnn knobs, trf film, clip/rewind) -- fixed as the
reference; tune's header carried the PRE-NESTED flat schema example
(model: int_dim_res -- would ERROR if copied) -> rewritten to
model.mlp.width, plus a note that search ranges nest at any depth
(model.cnn.kernel_size, head.lr_base; _walk_train_args recurses, so
this always worked -- only the docs lagged). experiment.py __init__
/ from_config docstrings caught up too (tatt, clip/rewind,
new cnn/trf keys). (2) parallel/ + PCE/ brought to current
machinery: ParallelResCNN gained the needs_geom + needs_bins
capability flags (it predates the isinstance->flag refactor) and
now THREADS block_opts["act"] into GroupedCNNBlock (same silent-H
bug ResCNN got fixed for on 06-30); everything else verified
compatible rather than changed -- the PCE losses already ride the
static-shape _reduce by delegation, batching.py already honors
target_dim + the needs_params encode, and the compiled
fwd_loss/fwd_chi2 twins already branch on needs_p. The recorded PCE
to-dos (pack-base-at-load via target_dim) were ALREADY BUILT.
(3) Doc pass, house style (formal + shape-flow graphs with
legends): parallel/ fully rewritten (module prose + ParallelResCNN
forward graph + GroupedCNNBlock graph + full Arguments blocks --
the user's flagged example); PCE/ module docstrings expanded
(verdict + build lessons + PS jargon), PCEEmulator forward +
from_training fit-pipeline graphs, PCEResidual encode graph,
PCERatio pack/unpack graphs; batching.py regime-ladder graph +
its 3 public sizing fns converted comment-headers -> formal
docstrings; data_staging.py staging-pipeline graph + param_stats
docstring; plotting.py plot_xi got a full Arguments docstring
(body stays byte-faithful). AST-minus-docstrings diff proves
doc-only files code-identical to HEAD; only
parallel/emulator_designs.py carries code changes (the flags +
act threading). NEW TESTS: test_print_design.py (banner content +
quiet gating), test_pce_parallel.py (19 checks: PCE fit exact on
a polynomial map 7.6e-14; residual loss == plain on base+pred
under tensor trim/focus/kappa; ratio pack/unpack == direct
residual; fullgraph compile + no recompile across annealed
values; BOTH PCE losses train through training_loop_batched incl.
the 2*n_keep target; ParallelResCNN flags/shape/act-threading;
grouped conv leaks exactly 0 across bins). FULL BATTERY: 16/16
suites green.

**TRANSFER-LEARNING CURRICULUM IDEA (2026-07-04w, user, "for
later" -- banked, NOT built).** Train the TRUNK on a restricted
parameter space (fixed photo-z errors; later: LCDM only) using a
cheap/large dump, then train the HEAD -- frozen trunk, fresh
optimizer, the existing two-phase machinery -- on the FULL space
(photo-z free; later: w0wa) with an independent, possibly much
smaller dump. Motivation: extension training data is expensive;
the head has 10-100x fewer params and learns only the residual, so
its sample complexity should be far lower (multi-fidelity /
residual-learning argument). KEY STRUCTURAL FACT: without film the
head CANNOT carry new-parameter dependence at all (it is a fixed
map of the trunk output); FiLM conditioning is the enabler, and
the conditioning vector must include the head-only parameters --
which the trunk should NOT see (exclude them from the trunk input
rather than train-on-a-slice-and-extrapolate). The existing
factored-amplitude machinery (append columns past n_in that the
trunk drops and a downstream consumer reads) is the exact pattern
to generalize: amplitudes -> loss; head-only params -> FiLM.
RISKS: per-token affine FiLM may be too rigid for full photo-z
freedom (may need more blocks / capacity -- measure); residual
still needs coverage in the new directions; trunk quality caps the
scheme (its systematic error must be head-correctable). PRACTICE
EXPERIMENT (cheap, decisive, same dump): phase 1 trunk on N rows,
phase 2 head on an INDEPENDENT N/2 subset (disjoint rows, fresh
from the dump), vs baselines (same-set head; full-N head) -- if
head-on-N/2-fresh matches, the head's sample appetite is small and
the premise holds. Mechanism needed: per-phase training subsets in
run_emulator (a head: n_train / fraction knob slicing disjoint
rows from tidx; loaders already index globally). The full version
(different dumps per phase) comes with the fixed-photo-z dump the
user can generate. Extension ladder: fixed-photoz -> free-photoz
head; LCDM trunk -> w0wa head.

**FIRST FILM EVIDENCE (04w addendum, same day).** Truncated-trunk
run (150 trunk epochs, best epoch 98 at frac 0.5000 -- deliberately
immature) + film head (cnn: k11 rescale -> k5, groups 6, 3 blocks,
separable, film, gate 1): head took frac 0.5000 -> 0.2766 BY EPOCH
22 and val med 0.20 -> 0.091. Compare the film-less 150-trunk smoke
(baseline 0.5586 -> plateaued ~0.37): the film head kept descending
where the fixed head stalled -- consistent with the mechanism (an
immature trunk's residuals are strongly parameter-dependent, which
only film can express). NOT yet proof of the cross-space transfer
premise (same parameter space both phases; the independent-subset
practice run remains the discriminating test), but a necessary
condition passed emphatically. Post-22 behavior: hard plateau, lr
cuts rewinding to 22 repeatedly (harmless by design; the schedule
positions advance through rewinds -- trim/focus are epoch-indexed
and do not rewind). The RISING train loss (0.92 -> 1.6 over epochs
11-52) is the trim shrinking (0.03 -> 0.01) + focus ramping
re-weighting the objective, NOT divergence. Practical: head
converges in ~25-150 epochs at this trunk quality -- use short
head phases for iteration, not 2350 epochs. User next: more
n_blocks (cheap: at this config each block ~ 4.2k params =
separable conv ~1,980 + film 2,160 + act 52; rescale keeps the
view fixed). WARNING TO INVESTIGATE: "skipping cudagraphs due to
cpu device (primals_90)" at head-phase compile -- a stray CPU
tensor in the traced fwd_loss/fwd_chi2 disabled CUDA-graph replay
(compiled kernels still run, but no single-replay steps; part of
the 5.6s/epoch). DEVICE AUDIT BUILT (same
day): audit_devices(model, lossfn, device) in training.py walks
every model parameter/buffer + every tensor attribute of lossfn
and lossfn.geom, flagging type mismatches AND cuda-index
mismatches (cuda:0 vs cuda:1 on the two-GPU boxes); run_emulator
calls it after build_loaders and prints "device audit: <owner>:
<device>" per offender (non-silent runs). test_device_audit.py.
AUDIT CAME BACK CLEAN (user rerun) -- primals_90 is NOT a stored
tensor. Narrowed suspects: (a) a Python float dynamo lifted to a
0-dim CPU tensor input on the head-phase RECOMPILE (automatic
dynamic promotes changed/specialized scalars; kappa is the prime
candidate), or (b) a tensor created at trace time inside a
forward. NEXT DIAGNOSTIC (no code change): rerun with env
TORCH_LOGS="cudagraphs" -- the skip message then carries "Found
from <file:line>" naming the producing code. The doubled skip
message = forward AND backward graphs both falling back. Fix
pattern once named: pass the offender as a 0-dim DEVICE tensor
like trim_t/focus_t. Also noted: the truncated-trunk film result
REPRODUCED (0.5000 -> 0.284 by head epoch 5, matching the first
run) -- the 04w-addendum finding is stable across runs.
RESOLVED (high confidence, fix shipped): TORCH_LOGS carried no
"Found from" on their torch build, so the primals table was
RECONSTRUCTED LOCALLY by tracing the exact production config with
a capture backend (find_primals_90*.py, scratchpad): head-phase
fwd_loss = 89 accounted inputs ending in trim_t/focus_t;
production's 90th is the ONE remaining Python float in the traced
step -- KAPPA (focus_scale). Some torch builds lift closure floats
as unspecialized 0-dim CPU tensor inputs, AND the lift happens on
RECOMPILE -- exactly why the trunk phase (first compile,
specialized constant) recorded graphs while the head phase
(recompile of the same code object) skipped. FIX: kappa_t =
torch.as_tensor(float(kappa), device=device), once per pass,
passed as focus_scale (_reduce already accepts tensor scalars;
bit-exact equivalence verified). No float remains in the traced
closure. CONFIRMED IN PRODUCTION (user: "success"): skip line GONE, head
epochs 5.4 -> 5.1s with graph replay restored (modest on the
small-SM card because the head phase carries real arithmetic; the
payoff scales on H200/under contention where launch overhead
dominates). The kappa-lift diagnosis stands proven end to end:
audit -> eval-graph-clean -> local primals reconstruction ->
tensor promotion -> skip eliminated.

**TRF FILM + TATT ACTIVATED (2026-07-04v; user: "so no film To
TRF? Once you do all that you need to implement the TATT version
for ia").** (A) model.trf.film bool: ResTRF + TemplateResTRF get
one identity-initialized FiLMGenerator per TRF block, applied to
the TOKEN STREAM after each block (t = gamma(z)*t + beta(z),
per-token, broadcast over max_bin; the stream carries the
correction corr = stream - t0, so identity init keeps corr = 0
exactly -- tested bit-exact). Tokens play the conv head's channel
role; conditioning = full input (plain) / x[:, :n_in] (factored,
amplitude-blind -- tested with randomized wo + last-MLP + generator
weights). Generators freeze/train with the head in set_train_phase.
(B) ia: tatt is LIVE: TATT_AMP_NAMES = [LSST_A1_1, LSST_A2_1,
LSST_BTA_1] (tatt_coeffs order a1, a2, b_TA; the eta powers
LSST_A1_2/LSST_A2_2 stay emulated inputs); IA_DESIGNS["tatt"] =
{3 amps, tatt_coeffs, 10 templates}; MODELS routes (resmlp|rescnn|
restrf, "tatt") to the SAME Template* classes -- zero new code
paths, exactly as the registry was designed (everything downstream
reads IA_DESIGNS: AmplitudeFactorGeometry columns,
TemplateFactoredChi2 n_amps, model n_amps/n_templates, conv groups
-> 1|10|20 automatic). Verified at tatt dims: ch = 300 conv head
with groups=20 + separable + film builds and is amplitude-blind in
ALL THREE amplitudes; 300-token TRF builds; tatt_coeffs == hand
formula; loss combine == manual einsum; groups=6 rejected. The
"tatt reserved" comments retired (experiment.py, YAML, README).
GATING ITEM: training dumps holding the 10 TATT templates do not
exist yet -- the code path waits on them. test_trf_film_tatt.py
(15 checks); full 12-suite battery green.

**FILM RE-INJECTION BUILT (2026-07-04u; user: "ok lets implement
it", after recovering the months-old claude.ai FiLM guide --
recovered design + full mapping in [[film-conditioning]]).** New
model.cnn.film bool (default off, house rule): one
identity-initialized FiLMGenerator per conv block
(Linear(n_cond, 2*C), zero weight, bias gamma=1/beta=0; new
building block with full docstring+graph) predicting a
per-channel affine applied conv -> gamma(z)*c + beta(z) -> act.
Conditioning z: ResCNN = the full input; TemplateResCNN =
x[:, :n_in] ONLY (the non-amplitude slice the trunk consumes) --
amplitude-blindness preserved BY CONSTRUCTION, tested with
randomized head weights (vary the amplitude column -> output
bit-identical; vary a cosmology column -> the correction moves).
Identity start untouched (bit-exact model == trunk at init, tested
both classes); generators are head params (set_train_phase freezes
them with the head; grads flow in head phase). Cost at nla dims:
2,160/block (= one separable block, coincidentally); current smoke
head 4,427 -> 8,747 with 2 blocks. Composes with groups +
separable + rescale_kernel (tested) and the compiled fwd_loss
(static shapes). ONE boolean only -- no d_latent knob, the
conditioning source is fixed as the trunk's own input. FiLM
strictly generalizes the per-template gate (gate = constant
per-template outer valve, kept; FiLM = per-channel function of
cosmology inside blocks). Ablation ladder next: film_grouped
(per-channel generators, same count) and conditional LayerNorm in
TRFBlock -- see [[film-conditioning]]. test_film.py (12 checks);
full 11-suite battery green. YAML/README/tune docs updated.

**LAUNCH-BOUND FIX: COMPILED FWD+LOSS + PRE-SHUFFLE (2026-07-04t;
user approved option 5 after the MCMC-contention incident showed
epochs are launch-bound, +50% under CPU load).** Diagnosis recap: a
tiny model's step is ~40 CPU-launched micro-kernels (model fwd
replay + ~30 eager loss fwd/bwd kernels + fused opt + gathers);
epoch time = CPU dispatch, not GPU compute; a faster GPU (user's
production H200) makes this WORSE relatively. Three changes: (1)
CosmolikeChi2 reduction rewritten STATIC-SHAPE and factored into
_reduce (sort + zero-weight prefix mask == topk exactly -- same
kept set, weighted mean permutation-invariant, grads identical;
topk's k anneals per epoch and would recompile/recapture); trim /
focus now accept 0-dim tensors (float guards recompile per value;
k computed in float64 so round parity with python holds);
ElementWeightedChi2 deduped onto _reduce; ALL other losses (IA,
Rescaled, PCE) delegate to the base -> inherit free. (2)
training_loop_batched builds _fwd_loss(xb, yb, trim_t, focus_t) =
autocast(model) + lossfn.loss, compiled with the SAME mode
make_model used (stashed as model.emul_compile_mode attr; off-CUDA
= same function uncompiled -- one code path); per-epoch anneal
fill_()s into 0-dim device tensors. (3) pre-shuffle per chunk (Cc =
Cc[bp] once) -> every batch is a contiguous slice view: kills 3
gather launches/step (factored path gathered Cc TWICE), one
transient chunk copy (~780MB @ 250k; bounded by chunk not N_train
-- at 10M dvs the pool streams in budget-sized chunks and the
transient is unchanged; what scales is H2D traffic -> the 10M-era
optimization is prefetch overlap, not graphs). ESTIMATES (accounting
not measurement): ~40 -> ~8 launches/step, quiet trunk epochs 0.75
-> ~0.4-0.55s, contention sensitivity /3-5; full-step CUDA graph
(option 4, NOT built) would buy ~15-20% more but needs capturable
optimizer + rewind copy_ semantics -- revisit after measuring.
test_fwdloss_compile.py (8 checks: old==new reduction to 2e-6,
tensor==float exact, k-parity sweep, fullgraph compile with
error_on_recompile proving one graph across annealed values, grads
flow, loop integration incl. needs_params). Full battery green.
NOTE fused AdamW was ALREADY auto-injected on CUDA in
make_optimizer (my tier-1 suggestion was pre-built); YAML fused:
false is silently overridden (setdefault would fix if ever wanted).
PRODUCTION VERIFIED (07-04, MCMC still running on amypond): trunk
epochs 2.2-2.4s contended -> 1.5s ("good - it did reduce"); epoch 1
= 22.8s one-time compile (loss now inside the trace). Quiet-machine
number still to be observed (~0.4-0.55s expected).

EVAL PASS STREAMLINED (04t follow-up; user: "if that is easy to
fix - why not fix it now?", + the amypond eGPU-over-TB4 argument:
~4 GB/s host link makes copies/latency precious). eval_val
rewritten consume-don't-stash: per batch, fwd_chi2(xb, yb) = model
forward + per-sample chi2 in ONE compiled graph (the eval twin of
_fwd_loss, built once per phase in training_loop_batched next to
its sibling and passed in; eager fallback built inside eval_val for
direct callers). Kills: the per-batch (bs, out_dim) prediction
clone, the ~1 GB/epoch stash+cat VRAM churn (factored heads), and
~half the eval launch count. The ragged final batch now pads BOTH
xb and yb (chi2 runs per batch; pad rows = duplicated row 0, real
chi2 values sliced off). D2H unchanged: one 4-bytes-per-val-point
transfer + the .item() reads (TB4-relevant: transfers already
minimal, kept that way). Numerically identical by construction
(per-sample chi2 is row-independent). TB4 traffic clarification
banked: the stash/cat churn was VRAM-internal, not host-link; what
crosses TB4 is loader streaming (zero if train+val resident, which
the budget planner targets: val planned against budget - train).
test_fwdloss_compile.py extended to 11 checks (eval == direct
full-batch chi2 with ragged batches, needs_params padded-params
threading, explicit-twin spy proving every batch runs at bs).
Full battery green.

**HEAD FOCUS ENABLED (2026-07-04s; user: "why CNN has no focus? by
the handoff there are very few outliers -- focus could help where
chi2 is ~1 to 10").** The all-zero head focus block was a
HISTORICAL safety default from the 04l post-mortem, not a
principled choice. User's physics is right: after a mature trunk,
the batch is ~85% solved bulk and the head's whole job is the
0.2-10 band, but a plain mean aims most gradient at the bulk by
headcount; focal weighting re-aims it. Structurally safer than the
04l chi2 failure: weights bounded (h <= 1), detached (ungameable),
monsters excluded by the trim floor, rewind as net. kappa selects
the band (h = c/(c+kappa) crosses 1/2 at c = kappa): kappa ~0.2 =
metric-aligned (everything above the frac>0.2 threshold; bulk gets
~15x less at gamma 2); kappa ~1 = the literal 1-10 band BUT
down-weights the counted 0.2-1 points to ~3% -- rejected for the
frac metric. RECOMMENDED head focus: start 0, end 2.0, kappa 0.2,
hold 15; anneal: user halved my symmetry-derived 100 to ~50 --
CORRECT, the ramp only needs to outlast the HEAD's fast-learning
window (head best ~epoch 90 historically; full gamma at ~65 lands
when easy gains are exhausted). Too-short symptom: flat first ~30
head epochs (focus on the tail before the broad correction is
learned) -> lengthen the HOLD, not the anneal. User's current
production template: sqrt both phases; trunk trim 0.1 -> 0.01
(hold 50, anneal 400); head lr_base 0.001, trim 0.03 -> 0.01
(hold 15, anneal 100); trim floors 0.01 everywhere = the 04l
vaccine. YAML STYLE (banked in auto-memory too): block style, one
key per line, never inline {...} -- example YAMLs converted.
README = the doc surface that repeatedly went stale (head knobs,
two-phase keys, guards all missing until audited 07-04); treat it
as first-class in every feature's doc pass.

**SEPARABLE CONV FLAG (2026-07-04r; user approved after a
weighted-sum design review: "yes lets do that").** New
model.cnn.separable bool (default false): each head block factors
into depthwise Conv1d(C->C, k, groups=C) (per-channel k-tap theta
filter, C*k weights) + pointwise Conv1d(C->C, 1, groups=groups)
(channel mix, C*(C/groups) weights), NO activation between -- the
pair composes into one constrained conv w[o,c,t] =
pointwise[o,c]*depthwise[c,t], i.e. a low-rank factorization of the
same weighted sum (~k/2 fewer weights), not a different operation.
Added assumption: a channel's theta-smoothing profile is
independent of which channel it mixes into (plausible for
covariance-driven leakage; MobileNet/Xception trade). Zero-init
identity start moves to the last block's pointwise (zeroing the
depthwise too would stall the wake-up). Composes with groups (on
the pointwise) and rescale_kernel (on the depthwise taps; RF math
unchanged, pointwise k=1 adds no reach). At the smoke config (C=90,
k=7, groups=6, 2 blocks): conv weights 19,080 -> 4,320 (head total
~4.4k vs trunk 76k). DESIGN REVIEW banked with it: weighted sum
kept as the base op (residuals small/signed/covariance-coupled ->
additive linear mixing + H is the matching bias; stacked blocks
already give nonlinear interactions); GLU-style multiplicative
mixing REJECTED for now (2x params, no evidence of
calibration-like multiplicative residuals -- revisit if
diagnostics show whole-bin scale errors); max/pooling rejected
(discards sign); attention = restrf, not a CNN flag.
test_separable.py (10 checks) + full battery green. PRODUCTION
TIMING (07-04 smoke run): param print verified to the digit (head
4,427 vs trunk 76,048), but head epochs got SLOWER, 3.9s vs ~2.9s
plain -- params and wall time are decoupled on the small-SM GPU
(same lesson as the conv-as-matmul revert): the head phase is
W_fd/W_df-dominated, so cutting 80% of conv weight saves ~nothing
while the two-kernels-per-block structure (depthwise =
bandwidth-bound, low arithmetic intensity) adds overhead. separable
buys PARAMETER economy (sample efficiency / regularization -- the
currency of the TATT+w0wa comparison), not speed; use plain conv
when wall time matters more.

**CONV GROUPS: PHYSICAL CHANNEL CUTS (2026-07-04p; user: "I want
groups=3 ... groups=2 (xi+ never mixes with xi-) ... and groups=6
where GG GI II dont talk AND xi+ dont talk to xi-").** New
model.cnn.groups key (default 1 = dense), passed to nn.Conv1d groups
= consecutive channel blocks that never mix; conv params divide by
groups (nla k=5: 40,590 / 13,590 / 6,840 at g = 1/3/6). Channel
order makes the cuts physical -- verified in
build_shear_angle_map's source: dv layout is pm outer (xi+ block
then xi-), pairs middle, theta inner, so plain-head channels =
xi+ pairs then xi- pairs, and the factored head is template-major
with that same bin order inside each template. Allowed values are
ENFORCED (anything else = loud error with the reason): plain rescnn
1 | 2 (2 = xi branch cut); rescnn+nla 1 | n_templates |
2*n_templates (3 = GG/GI/II isolated; 6 = that AND the xi cut).
GOTCHA the implementation guards: bin_sizes drops fully-masked
bins, so a wholly-masked bin on one branch would silently shift the
xi boundary -- the branch cut therefore VALIDATES per-bin pm
(geom.pm_kept run starts) at build and fails loudly if the halves
are not clean xi+/xi-. Physics framing: dense mixing is the
hypothesis that GG/GI/II residuals share structure; groups=3/6 are
its ablations (amplitude exactness untouched either way -- the head
acts before the loss combine). test_groups.py (12 checks: param
scaling, perturbation isolation incl. dense control, loud
rejections, shifted-boundary rejection, YAML mapping); full battery
green.

**TRUNK-VS-HEAD PARAM PRINT (2026-07-04o, user request).** Below the
"trainable parameters: N (M excluding pure linear transformations)"
banner line, run_emulator now prints "  trunk X vs head Y (excluding
pure linear transformations)" -- both numbers in the
excluding-linear convention, X + Y == M exactly. Trunk found by the
duck-type convention .mlp (ResCNN/ResTRF) else .model (ResMLP +
factored trunks); head = the complement (convs / TRF blocks /
gates); line omitted for pure trunks (nothing to split).
test_param_split.py (5 checks: reconciliation + manual counts + the
no-head case).

**KERNEL RESCALING FLAG (2026-07-04n; user: "kernel_size number =
the optimal one for 1 block, and another flag that when set
rescales it when I increase the number of CNN blocks").** New
model.cnn.rescale_kernel bool (default false). Semantics:
kernel_size is tuned AS IF n_blocks were 1, i.e. it states the
head's target receptive field; with the flag on, the per-block
kernel shrinks with depth so the stack keeps that view: RF of n
stacked same-padded convs = n*(k-1)+1, solve >= kernel_size for the
smallest odd k_n = odd-up(ceil((kernel_size-1)/n) + 1). Table at
kernel_size 27 (one bin + margin at max_bin 26): n = 1/2/3/4/5 ->
k = 27/15/11/9/7 (n=3 -> 11 reproduces the current production
config). Properties (tested): identity at n=1, always odd, RF never
undershoots, monotone non-increasing, head params ~ C^2 *
(kernel_size - 1 + 2n) nearly flat in depth (depth buys
nonlinearity, not size). NOT a replacement of kernel_size by a
string -- user explicitly rejected kernel_size: "auto" mid-build
("I am confused here") in favor of number + flag. Implementation:
rescale_kernel_size in building blocks; ResCNN + TemplateResCNN
take rescale_kernel and store the resolved self.kernel_size;
MODEL_BLOCK_KEYS cnn block maps it; example YAML documents it.
test_rescale_kernel.py (13 checks) + full battery green.

**CLIP + REWIND STABILITY GUARDS (2026-07-04m; user: "how to
implement both").** Two new YAML knobs, top-level with symmetric
trunk:/head: per-phase overrides (both default off): clip = per-step
gradient-norm ceiling (nn.utils.clip_grad_norm_ over
model.parameters() between backward and step; frozen params have
grad None and are skipped) -- kills the single-batch kicks (train
10517 epochs) any loss mode can produce on the fat tail; rewind = on
every ReduceLROnPlateau lr cut, model.load_state_dict(best_state) +
optimizer.load_state_dict(best-epoch snapshot) then REAPPLY the new
reduced lrs (load_state_dict would bring back the old lr). Optimizer
snapshot (copy.deepcopy, only when rewind on) is taken wherever
best_state is -- baseline seed + every new best -- because Adam
moments from a bad basin would kick restored weights right back out.
Rewind is plateau-scheduler-only BY DESIGN (a cosine/step scheduler
changes lr every epoch; rewinding on each change would pin the run
to its best forever). At a true plateau rewind is a no-op (best ~=
current); after an excursion it bounds the damage to `patience`
epochs -- the exact fix for 04l's 1200 frozen epochs. Threading:
training_loop_batched(clip, rewind) <- run_emulator(clip, rewind +
phase_opts.get) <- exp.train (train_args.get) <- YAML top level or
trunk:/head: blocks; the phase banner notes overrides. Tests:
test_clip_rewind.py (7 checks: spy-optimizer norm bound, rewind
restores best weights at reduced lr, no-rewind control, per-phase
threading). Advice for the next chi2-head run: head {trim end 0.01,
clip 1.0} + rewind true. VERIFIED IN PRODUCTION (07-04 smoke run,
trunk phase, user: "worked nicely"): epoch 1096 plateau cut ->
"rewound to best epoch 1094 (frac>0.2 0.1484), resuming at lr
6.39e-05"; the two drifted epochs were erased and the next epochs
immediately matched the best -- the healthy-plateau case is a
near-no-op exactly as designed.

**CONV-AS-MATMUL REVERTED (2026-07-04m addendum to 04k).** On the
production GPU the matmul path changed nothing (head epochs 2.9 ->
3.0s; epoch-1 compile grew 4.3 -> 15.8s -- the unfold graph compiles
slower). Post-mortem: the ~1%-of-matmul conv pathology was CPU-EAGER
only; on the GPU cuDNN/inductor handle the shape fine, and the
already-observed head/trunk ratio 3.9x < arithmetic ratio ~10x had
said so. The head phase there is at its arithmetic floor (~6.4
TFLOP/epoch at 3.0s ~ 2.1 TFLOP/s sustained on a small-SM card --
the log's "Not enough SMs" warning). User: "I prefer to use pytorch
cnn own functions" -> conv1d_as_matmul DELETED from building blocks,
both forwards back to self.convs[i](c) (resurrect from git history
if a CPU/MPS path ever matters). Lesson banked: benchmark
conclusions do not transfer across devices; the honest floor
argument (head does ~10x trunk MACs) was the real story all along.
Speed levers that DO exist: smaller kernel_size / fewer conv blocks,
larger bs.

**FIRST FULL TWO-PHASE PRODUCTION RUN: 0.1105 (2026-07-04l).** rescnn
+nla, T=256 N_train=250k, 1500 trunk + 1500 head (bs 768, head chi2
lr_base 1e-3, head trim 0.05 -> 0.0 hold 20 anneal 100): BEST frac>0.2
= 0.1105 at HEAD EPOCH 92 (val med 0.035) -- scoreboard nla-trunk-only
was 0.1472, so the conv head cut misses ~25%; goal 0.10 close.
Best-restore returned epoch 92, so the deliverable survived what came
next. COLLAPSE POST-MORTEM (user: "stuck on frac ~0.3 -- how to
avoid"): head trim hit 0.0 at epoch 120; with trim 0 the train MEAN
chi2 was ~79 vs MEDIAN ~0.05 -- the quadratic loss was ~entirely a few
monster outliers. Spikes began (169: train 10517; 225: 2323), a big
excursion at head epochs ~272-300 knocked it into a tail-fitting
basin, and from there train loss kept IMPROVING (79 -> 52) while val
med rose 0.035 -> 0.142 and frac froze at ~0.305 for 1200 epochs: the
untrimmed-chi2 objective genuinely prefers sacrificing the bulk to
shave monsters -- it anti-correlates with frac>0.2. ReduceLROnPlateau
then decayed lr to 5e-8 while stuck, freezing the bad basin (no
recovery mechanism). FIXES: (1) YAML, primary: head trim needs a FLOOR
(end: 0.01, never 0.0) -- the objective, not the optimizer, was wrong;
(2) candidate loop upgrades (not yet implemented): grad-norm clip
knob; rewind-to-best-weights whenever the plateau scheduler cuts lr
(bounds any excursion to `patience` epochs). Diagnostics PDF: bad
points (dchi2>0.2) skew to LARGER mean-dist-to-8-nearest-train-pts --
the residual 11% lives in sparse train regions, so N_train sweep is
the likely path below 0.10. Head params at this config (k=11, 3
blocks, ch=90): convs 267,570 + acts 156 + gates 3 = 267,729 (trunk
304,182; total 571,911). NOTE this run still used the OLD Conv1d path
(2.9s/epoch) -- re-sync for conv-as-matmul before the next one.

**CONV RUNS AS A MATMUL (2026-07-04k; user: head epochs 2.9s vs trunk
0.7s, "please investigate").** Root cause was NOT compile/no_grad/
scatter (all measured innocent, <2ms each): nn.Conv1d(90->90, k=11)
over LENGTH 26 sits outside every fast conv path and ran at ~1% of
matmul throughput (CPU probe: 48.7ms for 2.3M MACs/sample vs 0.48ms
for the same-size W_fd matmul). Honest floor first: head-phase steps
do ~10x the trunk phase's arithmetic anyway (12.9M vs 1.3M
MACs/sample; W_fd/W_df 3x780x780 both ways ~5.5M + conv fwd+bwd
~6.9M), so head epochs can never be trunk-cheap -- the fix targets
only the ~100x conv inefficiency on top. Fix: conv1d_as_matmul in
emulator_designs_building_blocks.py -- pad, unfold(2, K, 1) window
view, one (B*L, C*K) @ (C*K, C_out) GEMM against the SAME nn.Conv1d
weights (state_dict/checkpoints/optimizer groups unchanged; returns
contiguous so downstream .view works). Used in both ResCNN and
TemplateResCNN conv loops. Verified (test_conv_as_matmul.py, all
suites re-run green): forward AND all grads match the native path to
1e-5; CPU head step 80 -> 14.4ms (5.5x); expected on-GPU head epoch
~1.2-1.5s from 2.9s (GPU convs less pathological than CPU, factor
must be measured there). GOTCHA: takes effect only after re-syncing
dev -> the training machine; the user's first rerun predated the
patch landing, hence "didn't make a difference".

**HANDOFF-JUMP BUG FIXED (2026-07-04j; user's 300+700 rescnn+nla test
showed phase-2 epoch 1 exploding: val 0.34 -> 4.96, train loss 54891,
frac>0.2 -> 1.000).** ROOT CAUSE 1 (real bug): the warmup lr ramp was
applied AFTER each epoch trained -- epoch 1 of EVERY pass trained at
the FULL base lr while printing the ramped value it had just set for
epoch 2. Harmless at a random init; at the two-phase handoff the
full-strength first epoch wrecked the identity start. FIXED: warmup
set at the TOP of the epoch (epoch e of W trains at base*e/W; the
scheduler still only steps after warmup). ROOT CAUSE 2 (no safety
net): best-tracking never evaluated the INCOMING model, so the
handoff-quality weights were never snapshotted -- phase 2 could end
worse than phase 1. FIXED: baseline epoch-0 eval seeds
best_state/best_frac before the loop (printed as "epoch 0 baseline");
a pass can now never END worse than it STARTED. AMPLIFIERS in the
user's config (advice, not bugs): head loss_mode chi2 + trim 0.0 on a
still-fat tail (trunk_epochs 300 is early; mean-chi2 directions get
outlier-dominated) and gate_init 1.0 (full-scale corr). REVISED
ADVICE: for phased runs keep gate_init 0.1 and give the head phase a
small annealed trim (e.g. start 0.05 -> 0) or sqrt loss until the
trunk is mature; the chi2/no-trim head objective is for a
well-converged trunk. Loop-level tests: test_warmup_baseline.py
(lr ramp captured during training via a forward probe; destructive
pass returns incoming weights bit-exact).

**TO DISCUSS (user, 2026-07-04, deferred -- "I am tired"): POSITIONAL
ENCODING in the TRF, and whether position matters more generally in
our designs.** User's instinct: shared_mlp probably NEEDS a positional
encoding ("yes we need to add some positional encoding no?"). Context
for the discussion: (a) shared_mlp mode is the ONLY position-blind
piece in the package -- unique MLPs (default) encode position by
having per-token weights; the conv head's bins-as-channels is
position-aware (each channel pair has its own kernel slice); the
dense trunk is position-aware by construction; (b) the cheap standard
fix = a learned additive per-token embedding, a (n_tokens, max_bin)
parameter added to the tokens before the blocks (~2.3k params at
90x26 -- negligible), which would make shared_mlp a CLEAN
specialization-only ablation (position kept, specialization removed)
instead of the current combined ablation; (c) fixed sinusoidal
encodings make little sense here (tokens have physical identities,
not sequence order); (d) open general question: does the theta axis
WITHIN a token also want position info (the conv gets it from kernel
locality; attention sees theta only through the token features).

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
(2) head_lr_base REPLACED by per-phase override blocks, made SYMMETRIC
2026-07-04f (user: "this should be symmetric"): the top-level
loss_mode / lr / trim / focus are the SHARED DEFAULTS; train_args
trunk: (phase 1) and head: (phase 2) each override lr_base /
loss_mode / trim / focus for their own pass (trim+focus are FULL
replacement blocks incl. kappa, restarting at the pass's epoch 1;
rationale for head: post-handoff there are few outliers, so e.g.
loss_mode chi2 + no trim). Either block without trunk_epochs>0 raises
(silent-no-op trap). run_emulator signature: trunk_epochs, trunk_opts,
head_opts.

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
