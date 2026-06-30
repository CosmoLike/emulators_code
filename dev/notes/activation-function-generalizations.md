---
name: activation-function-generalizations
description: "The paper's gated activation H(x) and three built generalizations in pytorch1.ipynb (2026-06-26), all NEUTRAL at T=16 but reusable and worth re-testing at high T. CHARACTERIZATION: H(x) = (gamma + (1-gamma) sigmoid(beta x)) x = gamma*x + (1-gamma)*Swish_beta(x) -- a per-element LEARNABLE INTERPOLATION between the identity and Swish (gamma=0 -> Swish; beta=1 -> SiLU; beta->inf -> ReLU / LeakyReLU(gamma)); a smooth learnable Leaky-ReLU that beat tanh because it is NON-SATURATING (linear tails). GENERALIZATIONS (each a strict superset, init recovers H): GatedActivation(n_gates=K) = multi-gate bulk slope-schedule (3K+1 params); PowerGatedActivation = H times a BOUNDED tail-power transform psi_p (p in [0.5,1.5], safe superlinearity, 3 params); GatedPowerActivation = both (3K+2). Plus the conv-collapse gotcha (an expand-then-collapse conv needs a mid-nonlinearity or the channels fold to one kernel). Bake-off T=16: H 0.244, Power 0.240, MultiGate 0.227, Gated+Power 0.235 -- all within ~1.7sigma, and the width sweep later showed the small MultiGate edge was CAPACITY not bias."
metadata:
  node_type: memory
  type: project
---

Built 2026-06-26 in pytorch1.ipynb: three generalizations of the paper's gated
activation. All NEUTRAL at T=16 (bake-off below) -- kept because they are reusable
and are the natural thing to re-test at high T, where the function is hard enough
for a matched bias to bite ([[emulator-sample-efficiency-is-the-goal]]).

**The paper's H(x), characterized.** H(x) = (gamma + (1-gamma) sigmoid(beta x)) x =
gamma*x + (1-gamma)*Swish_beta(x), where Swish_beta(x) = x*sigmoid(beta x). So H is
a per-element LEARNABLE INTERPOLATION between the identity and Swish. gamma=0 ->
Swish (beta=1 -> SiLU); gamma=0, beta->inf -> ReLU; gamma=alpha, beta->inf ->
LeakyReLU(alpha). Geometrically a smooth learnable Leaky-ReLU: negative-tail slope
gamma, positive-tail slope 1 (pinned), kink sharpness beta, kink at 0. It beat tanh
in the user's CMB paper because it is NON-SATURATING (asymptotically linear tails);
tanh saturates (slope -> 0) and kills gradient + dynamic range.

**Three generalizations (classes in the notebook; each a strict superset, init
recovers H exactly):**
- GatedActivation(dim, n_gates=K): multi-gate BULK. gate = a0 + sum_k w_k
  sigmoid(beta_k (x - mu_k)); out = gate*x. K sigmoids shape the slope-vs-x schedule
  in the bulk. 3K+1 params/elem. K=1 also frees the positive slope (a0+w) and the
  kink center (mu).
- PowerGatedActivation(dim, p_min=0.5, p_max=1.5): H's gate TIMES a bounded tail-
  power transform psi_p(x) = sign(x)((1+|x|)^p - 1)/p, with p = p_min +
  (p_max-p_min) sigmoid(rho) CONFINED to [0.5,1.5] ("between sqrt(x) and x^1.5").
  psi_p has slope 1 at x=0 for ANY p (the /p normalizes), so p reshapes ONLY the
  TAIL; base 1+|x| >= 1 makes any p finite (no NaN). SAFE superlinearity, unlike a
  raw x^n. 3 params/elem (gamma, beta, rho).
- GatedPowerActivation(dim, n_gates=K): both -- multi-gate bulk TIMES psi_p tail.
  3K+2 params/elem.

**Why a raw x^n tail is wrong (and the linear tail is the sweet spot).** x^n blows
up (x=10 -> x^2=100, x^3=1000) = gradient explosion (n x^{n-1}) + extrapolation
catastrophe. It is the OPPOSITE failure from tanh: tanh saturates (slope -> 0), x^n
over-grows (slope -> inf); the LINEAR tail (slope 1) is the sweet spot between them,
which is exactly why H beat tanh. The bounded psi_p (p<=1.5, over a narrow prior)
gets "a bit more than linear" safely.

**Conv-collapse gotcha (same principle, reinforced).** An expand-then-collapse conv
block (conv 1->C filters, then 1x1 collapse C->1) needs a NONLINEARITY BETWEEN them
(act_mid) or the two LINEAR convs fold to a single 1->1 kernel and the C channels
are wasted -- same fact that depth/width only adds capacity ACROSS a nonlinearity.
(CNNBlock had this bug; fixed with act_mid.) Real ML channels are never wasted
because there is always a nonlinearity between conv layers; the collapse only
happens in a contiguous LINEAR run.

**Bake-off (T=16, single seed, Nval~2000 so noise ~+/-0.01, width 128): H 0.244,
Power 0.240, MultiGate K=3 0.227, Gated+Power K=3 0.235.** Whole spread ~1.7sigma ->
suggestive not proven. Pattern: bulk-flexibility (multi-gate) is the only one with a
(weak) signal; tail-power is IDLE at T=16 (features never reach their tails) and
adding it to the multi-gate slightly HURT (unused DOF = dead weight). LATER UNDERCUT
by the width sweep ([[emulator-sample-efficiency-is-the-goal]]): width 128 was
capacity-under-saturated (256 -> 0.212), so the MultiGate "win" was likely CAPACITY,
not bias -- at saturated width it would collapse to plain. So NEUTRAL at T=16, like
every other model-side lever.

**Wiring / caveat.** Factory pattern: block_opts = {"act": lambda s:
GatedPowerActivation(s, n_gates=3)}. Weight-decay caveat: w/beta/mu are 2D (K,dim),
so make_optimizer's ndim>=2 rule would DECAY them (they are activation shape params,
not weight matrices) -- keep weight_decay=0 (the runs do) or exclude them; gamma/
beta/a0/rho are 1D and fine.

**Why:** reusable activation machinery + the clean characterization (H = identity
<-> Swish interpolation) + the safe bounded-tail-power construction + the conv-
collapse gotcha; all neutral at T=16 but the right thing to re-test at high T.
