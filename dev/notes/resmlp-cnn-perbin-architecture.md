---
name: resmlp-cnn-perbin-architecture
description: "ResMLP + 1D-CNN correction head for the cosmic-shear emulator, grounded in the user's PUBLISHED CMB design (ResMLP -> embedding -> 1D CNN/TRF -> output). CORRECTED 2026-06-25b: the clean GLOBAL ResCNN is NEUTRAL at T=16 (f(dchi2>0.2) ~= 0.20 == plain ResMLP, gate near init); the earlier '0.24->0.18 worked' was CONFOUNDED (it changed full->diagonal whitening AND used a broken mid-nonlinearity-less CNNBlock). Final architecture = APPENDIX ResCNN: pure-ResMLP trunk in FULL whitening + a GATED conv correction that converts to theta order internally via fixed buffers W_fd/W_df (CUDA-graph-safe; live geometry calls + reduce-overhead crash); fixed CNNBlock has act_mid; needs compile_mode='default' (reduce-overhead trips CUDA-graph-trees on the skip-add). LEADING READ: the T=16 toy is too easy/narrow to exhibit the theta-structured residual a conv fixes; the CNN's value is at HIGH training temperature (real target T=512), per the paper -- test by swapping a hotter training file. Classes: DiagonalGeometry, CNNBlock, ResCNN(appendix), GroupedCNNBlock, ResCNNPerBin. Gotchas: conv-collapse-without-mid-nonlinearity; geometry-call-in-forward breaks CUDA graphs (use buffers); reduce-overhead fragile. PORT FIX 2026-06-30: ResCNN threads --activation into the CNN head (block_opts['act'], fallback activation_fcn); was silently H regardless of the trunk."
metadata:
  node_type: memory
  type: project
---

**STATUS (CORRECTED 2026-06-25b): the clean global ResCNN is NEUTRAL at T=16 --
the earlier "0.24->0.18 WORKED" was CONFOUNDED, walk it back.** That first run
changed the WHITENING (full->diagonal) AND used a BROKEN CNNBlock (no
mid-nonlinearity, so 16 channels folded to 1), so the drop was the whitening
change, not the CNN. The corrected architecture (APPENDIX ResCNN: a pure-ResMLP
trunk in FULL whitening + a gated 1D-CNN correction that converts to theta order
internally; fixed CNNBlock with act_mid; CUDA-graph-safe buffers) comes out
f(dchi2>0.2) ~= 0.20 at 10k == plain ResMLP (within noise). The learned gate sits
near its init -> the optimizer found no use for the conv.

**LEADING EXPLANATION (the user's call, and it is right): the T=16 toy is TOO EASY
to discriminate an output-axis head.** At T=16 the training prior is narrow, the dv
barely varies, the ResMLP already nails the bulk (median ~0.04), and the residual
it leaves is the scattered ~20% tail, NOT a smooth theta-structured error -- so a
conv has nothing to correct. The published CNN/TRF win (0.2->0.06) lived in the
HARD, HIGH-TEMPERATURE regime (broad prior -> large dynamic range, structured
nonlinear theta-variation -> big axis-correlated ResMLP residuals = the conv's home
turf). The user trains production emulators at much higher T (real target T=512).
So this is "the testbed is too easy," not "the CNN does not work." Caveat: high T
also worsens input COVERAGE (same N over more volume), which a head cannot fix --
but the paper is the existence proof the output-structure win can dominate at high
T. NEXT: rerun the SAME resmlp-vs-rescnn comparison on a hotter training file
(cs_64/256/512 from cosmolike -- the user's workstation; code is ready, only the
data T changes); expect the gate to grow and the rescnn curve to pull below resmlp.
Free pre-check without new data: the per-element residual diagnostic on the T=16
ResMLP -- flat/tiny along theta confirms "nothing to fix." The PER-BIN variant is
built but moot until a regime where the global CNN bites.

**T=16 ResCNN DATA (2026-06-25b), and the refinement it forces:** learned gate
went 0.1 -> 0.168 (GREW, did not collapse -> the CNN is NOT idle; it found a
loss-reducing direction), yet frac>0.2 ~= 0.20 (flat vs plain). The per-element
shoulder residual is CONCENTRATED, not flat (top 5% of elements carry 42% of the
marginal mis-fit), with a clear theta-trend: worst block = small theta (8-17') +
the two highest source-z bins (0.97, 1.34), rms ~0.19 sigma, bias ~0 (scatter).
THE TRAP: that small-theta/high-z block is the MARGINAL-LENS GHOST the
[[emulator-floor-is-data-coverage]] / [[pytorch-teaching-style]] notes already
flagged -- strongly-correlated neighbours share a smooth common-mode the chi2
(high-pass filter) barely charges for, so reducing its marginal residual does NOT
move frac>0.2. Net read: the CNN found real but small theta-structure (gate up,
bulk loss down) that sits in the chi2's BLIND SPOT -> active but metric-neutral.
Deeper point: the CNN's smoothness inductive bias OVERLAPS the chi2's blind spot,
so on an easy regime it corrects what the metric forgives. This refines "too easy"
to "the structure exists but is small and chi2-tolerated at T=16"; the bet is that
at high T the same small-theta/high-z structure grows large enough to leave the
common-mode and become chi2-penalized signal the conv can convert to a metric gain
(the paper's 0.2->0.06 regime). Stop dissecting the T=16 ghost; the hot-T run is
the next real experiment.

(Original "it worked" text below is RETAINED for the design rationale only; the
empirical claim is superseded by the corrected status above.)

**IMPLEMENTATION (all classes in the notebook; plain CosmolikeChi2 loss
throughout -- only the MODEL and the GEOMETRY change):**
- **DiagonalGeometry** (DataVectorGeometry subclass) -- REQUIRED for any CNN. The
  default full whitening rotates the dv into the covariance EIGENBASIS, which
  scrambles the theta axis, so a conv over it would convolve eigenmodes (useless).
  DiagonalGeometry whitens by SCALING each kept element by its marginal
  sigma=sqrt(diag cov), NO rotation -> the dv stays in THETA ORDER so the conv sees
  the real axis. Overrides whiten (x/sigma) + unwhiten (x*sigma); encode/decode/
  chi2 inherit; the chi2 still uses the FULL Cinv_sq (so MSE != chi2 -- keep the
  explicit contraction). Verified: a delta stays a delta after whiten (==1 nonzero
  -> diagonal, not rotating), round-trip exact, chi2 == full-whitening == direct.
- **CNNBlock** (global head): conv_in(1->channels, k, same-pad) -> ACTIVATION ->
  conv_out(channels->1) -> activation. The MIDDLE activation is ESSENTIAL: two
  stacked LINEAR convs collapse to a single 1->1 kernel, so without it >1 channel
  is wasted (the first version lacked it; channels=1 hid the bug). channels default
  ~16 (the CMB students went up to ~72).
- **ResCNN** (global): Linear(in->int_dim) -> n_blocks ResBlock -> Linear(int_dim->
  cnn_dim) -> CNNBlock(cnn_dim) -> Linear(cnn_dim->out) -> Affine. cnn_dim defaults
  to output_dim (the kept-dv length = geom.dest_idx.numel(), from cosmolike --
  NEVER hardcode 705; it comes from the geometry). The ResMLP+embedding (cnn_dim ~
  dv size, as in the CMB 512->5120 ~ 4998) produces a DECENT dv -- it does the
  MAPPING; the CNN CORRECTS the axis-structured residuals it leaves.
- **GroupedCNNBlock + ResCNNPerBin** (per-bin): grouped Conv1d (groups=n_bins)
  convolves each bin INDEPENDENTLY (no smoothing across the bin-boundary jumps the
  global conv blurs). Bins differ in length (24-26), so PAD each to max_bin:
  cnn_dim = n_bins*max_bin, reshape (B, n_bins, max_bin), grouped conv, final
  Linear maps the padded layout to the real dv. Needs DiagonalGeometry +
  build_shear_angle_map (bin_sizes). NOT yet run.
- **The sweep compares MODELS, not losses**: build_method returns (chi2fn,
  model_opts) -- resmlp -> DataVectorGeometry+ResMLP; rescnn -> DiagonalGeometry+
  ResCNN; rescnn_pb -> DiagonalGeometry+build_shear_angle_map+ResCNNPerBin (spec
  built INLINE since it carries geom_n). Read the SLOPE of f vs N_train on log-log:
  the win is a CNN curve sustaining its descent where ResMLP flattens (the CMB
  TRF-vs-ResMLP divergence); rescnn_pb minus rescnn = the cost of the global conv's
  boundary smoothing.

**PACKAGE PORT FIX (2026-06-30): ResCNN threads the run's activation into the CNN
head.** In the ported emulator/ package, EmulatorExperiment injects the
--activation choice (make_activation) into model_opts["block_opts"]["act"], which
flowed to the trunk ResBlocks but NOT to CNNBlock: ResCNN built the head with
CNNBlock's default act=activation_fcn (the paper's H). So a non-default
--activation (power / multigate / gated_power) applied to the TRUNK only, while
the conv head (act_mid + the output act) silently stayed on H. Fixed:
ResCNN.__init__ now passes act=block_opts.get("act", activation_fcn) to each
CNNBlock, so head and trunk share one activation family; default H is unchanged.
Any earlier rescnn run with a non-H activation had an H head, so re-run if that
mattered. All four make_activation factories take a single size arg, which is
exactly what CNNBlock calls (act(dim)).

**The architecture (shared trunk, per-bin CNN head):**
1. **Shared ResMLP trunk**: params -> latent (e.g. 512). KEEPS the cosmology-map
   parameter sharing -- learned ONCE for all bins.
2. **Embedding layer**: latent -> per-bin feature maps (channels x n_theta), the
   way the CMB paper's embedding (512 -> 5120) reshaped the latent into a 1D
   feature map for the conv.
3. **Per-bin 1D CNN**: ONE independent 1D CNN per tomographic bin (xi+/-, source
   pair). Implement as `nn.Conv1d(..., groups=n_bins)` -- one kernel does all bins
   in PARALLEL (avoids the slow 30-module loop the per-bin ResMLP hit, same
   grouped-batching idea as GroupedLinear/einsum "gbi,gio->gbo").
4. **Per-bin linear output** -> that bin's values; concatenate -> full dv.

**Why it is right (it fixes BOTH per-bin ResMLP failures, see
[[per-bin-parallel-resmlp-plan]]):**
- The pure per-bin ResMLP split came out WORSE than the single ResMLP (~0.27 vs
  ~0.24 frac>0.2 at matched params) because (a) it threw away the parameter
  sharing -- 30 heads each re-learned the SHARED cosmology->dv map alone, from the
  same 12 params, with no cross-bin borrowing, so the hard part was replicated 30x
  not divided; and (b) a dense MLP is OUTPUT-PERMUTATION-INVARIANT, so it never
  exploited the within-bin theta-smoothness anyway (it predicts each output via
  its own readout row; reorder the outputs + targets + Cinv and it trains
  identically). The dv's 1D layout (smooth bins, sawtooth theta recycling
  large->small, jumps at bin boundaries, xi+/xi- amplitude split) is real but
  INVISIBLE to a dense MLP.
- This architecture fixes both: the SHARED trunk keeps the sharing, and the 1D CNN
  is an AXIS-AWARE head -- it sees theta as an axis (local weight-sharing kernel),
  so it genuinely exploits the smoothness, which is itself an effective-DOF
  reduction (learn one theta-kernel, apply everywhere). The per-bin split is now
  on the CHEAP refinement (the CNN), not the expensive shared trunk.

**BARE OUTPUT FIRST (the user's call).** First test = shared trunk + embedding +
per-bin 1D CNN + per-bin linear output (raw bin values). The smooth-basis output
(CNN -> ~5 Chebyshev/spline coefficients or xi-vs-theta PCA modes per bin; dv =
Phi.c, smooth by construction, ~150 vs 705 DOF) is a LATER refinement -- the CNN's
weight-sharing already supplies the smoothness inductive bias, so bare-first is
right; add the basis only if the bare CNN underwhelms. (Note: a smoothness
PENALTY is redundant with the high-pass chi2; a smooth-basis OUTPUT is a real DOF
reduction -- different mechanisms, see [[pytorch-teaching-style]].)

**Expect a LARGE gain (published precedent), not modest.** CORRECTION (the
assistant first called this "modest" -- WRONG, refuted by the user's paper): in
the user's CMB emulator the post-ResMLP axis-aware head dropped f(dchi2>0.2) from
~0.2 (bare ResMLP) to ~0.06 (ResMLP+TRF) -- a factor ~3 -- and a 1D CNN tracked
the transformer almost exactly. The reason: the latent->dv OUTPUT mapping is a
LARGE share of the difficulty (not cheap refinement). The bare ResMLP's single
dense output projection (512 -> n_dv) treats every output as an INDEPENDENT
readout, so it needs lots of data to pin them all down; an axis-aware head (CNN/
TRF) shares weights along the ell/theta axis and learns the output structure with
far FEWER effective parameters -> big sample-efficiency win. This is the
reduce-effective-DOF lever applied to the OUTPUT side (the part underweighted
earlier). So this is the MOST PROMISING lever, with published precedent for a
large drop. Only open caveat: the paper's curves are >=100k; cosmic shear here is
~10k, so the magnitude at our N is to be measured. Judge by the converged
frac>0.2 plateau at fixed N (~10k) vs the single-ResMLP ~0.22-0.24 baseline.

**Open design choices at build time:** independent per-bin CNN (specialization,
groups=n_bins, more params) vs ONE shared CNN applied to every bin (fewest params,
exploits that all bins are correlation functions with similar theta-shape -- the
paper's global-CNN spirit); embedding shape (channels x n_theta per bin); CNN
depth/kernel/channels. Reuse the grouped-Conv1d batching so it stays fast on CUDA.

**Why:** the architecturally-correct realization of the user's theta-structure
intuition and their published CMB design, replacing the failed per-bin ResMLP
split. Pairs with [[geometry-loss-composition]] (still plain CosmolikeChi2 loss;
the model changes, not the geometry/loss) and the analytic INPUT feature
([[geometry-loss-composition]], AugmentedParamGeometry) -- the two live
sharing-preserving levers (input side vs output side).
