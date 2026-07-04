# FiLM conditioning (recovered from a months-old claude.ai chat)

Source: `~/Downloads/film_conditioning_guide.pdf` (18-page lecture
note, "FiLM Conditioning in a 1D CNN Emulator", re-shared
2026-07-04; extracted text in the session scratchpad). Based on
Dumoulin et al. 2018 (Feature-wise transformations, Distill) and
Zhu et al. 2025 (arXiv:2505.22574, the Part III attention-emulator
paper -- the same series whose adapter design the ResTRF build
rejected).

## What the guide says (faithful summary)

Problem it solves: in an MLP -> CNN emulator the cosmological
parameters enter once, at the front; each CNN block dilutes that
information ("the student loses the problem summary during the
exam"). FiLM re-injects it at every block via a per-channel affine:

    FiLM(h; z) = gamma(z) * h + beta(z)

    theta_cosmo (B, n_param)
       |  MLP
       v
    z (B, d_latent)  ----------------------------+
       |  project + reshape                      |  FiLM generator
       v                                         |  per block:
    h (B, C, N_theta)                            |  Linear(d_latent,
       |  n_blocks x [Conv1d -> (BN) ->          |         2*C)
       |     FiLM(gamma_l, beta_l) -> act]  <----+  -> gamma, beta
       v                                            (B, C) each
    head -> xi

Key mechanics:
- gamma, beta are per CHANNEL, broadcast over theta (unsqueeze(-1)):
  cosmology sets a global property of each angular piece, not a
  per-theta local correction. Different channels get different
  modulation = cosmology chooses which pieces to amplify.
- The generator is one Linear(d_latent, 2C) per block (independent
  per block: early blocks condition broad shape, late blocks fine
  structure; cost 2*C*d_latent each -- tiny).
- Identity init: gamma = 1, beta = 0 (zero weight, bias [1, 0]) so
  FiLM starts as a no-op -- the guide's version of our zero-init
  identity-start philosophy.
- Placement: after the norm (BN or LayerNorm), before the
  activation; equivalent to Conditional BatchNorm.
- Channel-grouped variant: one tiny Linear(d_latent, 2) per channel
  -- same parameter count, inductive bias aligned with "each
  channel is a physical piece".
- Interpretability payoff: after training, plot gamma_c vs sigma8 /
  Omega_m per channel -- a direct probe of which angular pieces the
  network thinks are cosmology-sensitive; gamma ~ 1 channels need no
  reinjection.

## Mapping onto OUR architecture (the important part)

The guide's baseline is the latent-reshape design (a synthetic
feature sequence projected from z -- the same family whose
adapters the ResTRF build deleted). Ours keeps the physical dv as
the features; FiLM transfers cleanly because it only needs (a)
channels and (b) a conditioning vector.

What it would add that the current heads LACK: the correction head
(conv or TRF) never sees the cosmological parameters -- it is one
fixed map applied to whatever templates arrive. FiLM makes the
correction parameter-dependent, per (template, bin) channel:
gamma, beta shape (B, T*n_bins) from z. This matches the
diagnostics story (residual misses concentrated in sparse / edge
parameter regions -- exactly where a fixed map cannot adapt).

Relations to existing pieces:
- STRICTLY GENERALIZES THE GATE: our gate = per-template constant
  scaling the correction; FiLM gamma = per-channel FUNCTION of
  cosmology inside the blocks. (gate could stay as the outer
  identity-start valve; FiLM modulates within.)
- Identity init (gamma=1, beta=0) composes with the zero-init last
  conv: the head still equals the trunk exactly at epoch 1 and at
  the two-phase handoff.
- Orthogonal to the H activation: H has learnable PER-POSITION
  (theta) parameters, fixed across cosmology; FiLM is PER-CHANNEL,
  cosmology-dependent. Two different axes.
- The channel-grouped FiLM variant is the same physics instinct as
  our groups=3/6 cuts.
- CRITICAL factored-design constraint the guide does not know: the
  conditioning vector z MUST exclude the IA amplitudes
  ([:, :-n_amps], the trunk's own input) or the head stops being
  amplitude-blind and the closed-form amplitude exactness dies.
- Cost from raw whitened params (11-dim): Linear(11, 2*90) ~ 2.2k
  params per block -- noise next to even the separable head.
- Compile/CUDA-graph friendly: static shapes, no data-dependent
  anything; works inside the compiled fwd_loss unchanged.
- We have no BatchNorm (deliberate); guide confirms FiLM is valid
  without it (apply after conv, before act). TRF head: same idea
  with gamma per token (and TRFBlock already has LayerNorms to
  condition, the classic conditional-LN placement).

## Status

Recovered + banked 2026-07-04; rung (1) IMPLEMENTED the same day
(user: "ok lets implement it"): model.cnn.film bool, FiLMGenerator
building block, both conv heads, identity init, amplitude-blind
conditioning, two-phase-aware; test_film.py, 12 checks green. See
nla-as-design-spec 04u for the implementation record. Remaining
ablation ladder: (2) channel-grouped generator (film_grouped, same
parameter count, tighter physics); (3) conditional LayerNorm in
TRFBlock (the TRF head's version). No production run yet -- the
film: false -> true one-key ablation is ready whenever a smoke run
slot opens.
