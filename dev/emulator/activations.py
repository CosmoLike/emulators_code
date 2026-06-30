"""Scalar learnable activation functions for the ResBlock `act` slot.

Each class is an nn.Module for one elementwise activation with learnable
per-feature shape parameters; a ResBlock takes an `act` factory (a
callable act(dim) -> module) and builds one per layer. activation_fcn is
the paper's H(x) = (gamma + (1-gamma) sigmoid(beta x)) x, a learnable
identity<->Swish interpolation; GatedActivation (K gates),
PowerGatedActivation (a bounded power tail), and GatedPowerActivation
(both) generalize it. make_activation maps a short name ("H", "power",
"multigate", "gated_power") to the matching factory, so the activation
can be chosen by string from a driver or YAML.
"""

import torch
import torch.nn as nn


class activation_fcn(nn.Module):
    """
    The paper's learnable activation H(x): a per-element interpolation
    between the identity and a Swish-like gate.

      H(x) = (gamma + (1 - gamma) * sigmoid(beta * x)) * x

    Each feature has its own learnable gamma and beta (length-`dim`
    vectors). The gate gamma + (1 - gamma) sigmoid(beta x) runs from gamma
    (x -> -inf) to 1 (x -> +inf), making H asymptotically linear at both
    tails (slope gamma left, 1 right) -- non-saturating, hence better than
    tanh here. gamma = beta = 0 at init, so H starts as 0.5 * x
    (sigmoid(0) = 0.5); training then shapes each feature's curve. The
    Gated/Power/GatedPower variants generalize this same gate.

    Arguments:
      dim = feature width (one independent gamma / beta per feature).
    """
    def __init__(self, dim):
        super(activation_fcn, self).__init__()
        self.dim   = dim
        self.gamma = nn.Parameter(torch.zeros((dim)))
        self.beta  = nn.Parameter(torch.zeros((dim)))
    def forward(self,x):
        # H(x) = (gamma + (1 - gamma) sigmoid(beta x)) * x, elementwise.
        exp = torch.mul(self.beta,x)            # beta * x
        inv = torch.special.expit(exp)          # sigmoid(beta x)
        fac_2 = 1-self.gamma                     # (1 - gamma) weight
        out = torch.mul(self.gamma + torch.mul(inv,fac_2), x)
        return out


class GatedActivation(nn.Module):
  """
  Generalized H(x): x times a learnable gate of K sigmoids.

    gate(x) = a0 + sum_k w_k * sigmoid(beta_k * (x - mu_k))
    out     = gate(x) * x

  Every term is a bounded sigmoid times x, keeping the output
  asymptotically linear (slope a0 as x->-inf, a0+sum_k w_k as
  x->+inf) -- non-saturating like H, never blows up.

  H = (gamma + (1-gamma) sigmoid(beta x)) x is the K=1 case
  (a0=gamma, w=1-gamma, mu=0); the general form also frees the
  positive-side slope (a0+w) and the kink center mu, and K>1 adds
  gates (a learned slope-vs-x schedule). All parameters are
  per-element vectors of length `dim`, one activation shape per
  feature (as gamma/beta were).

  Arguments:
    dim     = feature width (gamma/beta were this shape too).
    n_gates = number of sigmoid components K (default 1).
  """
  def __init__(self, dim, n_gates=1):
    super().__init__()
    K = n_gates
    # a0 = negative-tail slope (gate value as x -> -inf).
    self.a0 = nn.Parameter(torch.zeros(dim))
    # per-gate weight / sharpness / center, each (K, dim). Init
    # reproduces H's start: gate 0 (w=1, beta=0, mu=0) -> 0.5;
    # extra gates inactive (w=0) but beta=1, spread mu, ready to
    # specialize once training turns them on.
    w0    = torch.zeros(K, dim)
    beta0 = torch.zeros(K, dim)
    mu0   = torch.zeros(K, dim)
    w0[0] = 1.0
    if K > 1:
      beta0[1:] = 1.0
      mu0[1:] = torch.linspace(-1.5, 1.5, K)[1:, None]
    self.w    = nn.Parameter(w0)
    self.beta = nn.Parameter(beta0)
    self.mu   = nn.Parameter(mu0)

  def forward(self, x):
    # unsqueeze(-2) adds a size-1 axis before the last:
    # (..., dim) -> (..., 1, dim), which broadcasts against the K
    # gate parameters (shape (K, dim)) -> (..., K, dim), matching
    # each input value against all K gates at once.
    xx = x.unsqueeze(-2)                            # (...,1,dim)
    s  = torch.sigmoid(self.beta * (xx - self.mu))  # (...,K,dim)
    gate = self.a0 + (self.w * s).sum(-2)          # (..., dim)
    return gate * x


class PowerGatedActivation(nn.Module):
  """
  H(x) with a learnable, bounded power tail. Same leaky/Swish gate
  as the paper's H, but the multiplied x becomes a signed power
  transform psi_p: linear near 0, ~|x|^p in the tail, with p
  learnable per element and confined to [p_min, p_max] (default
  [0.5, 1.5], between sqrt(x) and x^1.5). p = 1 recovers H.

    gate(x) = gamma + (1 - gamma) * sigmoid(beta * x)
    psi_p(x) = sign(x) * ((1 + |x|)^p - 1) / p
    H(x)     = gate(x) * psi_p(x)

  psi_p has slope 1 at x=0 for any p (the /p normalizes it), so p
  reshapes only the tail, not the behavior near 0. The base
  1+|x| >= 1 keeps any real p finite (no NaN), and the sigmoid box
  blocks a blow-up power -- safe on a narrow prior, unlike a raw
  x^n. rho=0 at init -> p=1 -> starts as H.

  Arguments:
    dim   = feature width (per-element gamma/beta/rho vectors).
    p_min = smallest tail exponent (default 0.5, sqrt-like).
    p_max = largest tail exponent (default 1.5, mildly super-
            linear). p ranges in (p_min, p_max) via a sigmoid.
  """
  def __init__(self, dim, p_min=0.5, p_max=1.5):
    super().__init__()
    self.gamma = nn.Parameter(torch.zeros(dim))
    self.beta  = nn.Parameter(torch.zeros(dim))
    # rho sets the exponent: p = p_min + (p_max-p_min)*sig(rho).
    # rho=0 -> midpoint p=1 for [0.5,1.5] -> identity tail.
    self.rho   = nn.Parameter(torch.zeros(dim))
    self.p_min = p_min
    self.p_max = p_max

  def forward(self, x):
    # bounded learnable exponent in (p_min, p_max), per element.
    p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(
      self.rho)
    # signed power: linear (slope 1) near 0, ~|x|^p in the tail;
    # base 1+|x| >= 1 keeps any p finite (no NaN).
    ax  = x.abs()
    psi = torch.sign(x) * ((1.0 + ax) ** p - 1.0) / p
    # leaky/Swish gate (your H), applied to the power transform.
    g = self.gamma + (1.0 - self.gamma) * torch.sigmoid(
      self.beta * x)
    return g * psi


class GatedPowerActivation(nn.Module):
  """
  The full activation: a K-component multi-gate (bulk slope
  schedule) times a bounded power-tail transform. Merges
  GatedActivation (K gates) and PowerGatedActivation (tail
  exponent) -- the two orthogonal generalizations of H(x).

    gate(x) = a0 + sum_k w_k * sigmoid(beta_k * (x - mu_k))
    psi_p(x) = sign(x) * ((1 + |x|)^p - 1) / p
    H(x)     = gate(x) * psi_p(x)
    p        = p_min + (p_max - p_min) * sigmoid(rho)

  The K sigmoids shape the slope vs x in the bulk; psi_p reshapes
  only the tail (slope 1 at x=0 for any p), with p boxed into
  [p_min, p_max] so it cannot blow up. Every term is a bounded
  sigmoid times a mild power, keeping the output finite.

  Recovers H at K=1 and the default init: gate 0 (w=1, beta=0,
  mu=0) -> 0.5, and rho=0 -> p=1 -> psi=x, so H = 0.5 x at init.
  Extra gates start inactive (w=0).

  Per-element parameters: a0 (1) + {w,beta,mu} x K (3K) + rho (1)
  = 3K + 2 vectors of length `dim`.

  Arguments:
    dim     = feature width (per-element parameter vectors).
    n_gates = number of bulk sigmoid gates K (default 1).
    p_min   = smallest tail exponent (default 0.5, sqrt-like).
    p_max   = largest  tail exponent (default 1.5, super-linear).
  """
  def __init__(self, dim, n_gates=1, p_min=0.5, p_max=1.5):
    super().__init__()
    K = n_gates
    # --- multi-gate (bulk slope schedule) ---
    self.a0 = nn.Parameter(torch.zeros(dim))   # neg-tail slope
    w0    = torch.zeros(K, dim)
    beta0 = torch.zeros(K, dim)
    mu0   = torch.zeros(K, dim)
    w0[0] = 1.0                                # gate 0 -> H init
    if K > 1:
      # extra gates: active (beta=1), spread centers, but w=0
      # (inactive) until training engages them.
      beta0[1:] = 1.0
      mu0[1:] = torch.linspace(-1.5, 1.5, K)[1:, None]
    self.w    = nn.Parameter(w0)
    self.beta = nn.Parameter(beta0)
    self.mu   = nn.Parameter(mu0)
    # --- bounded tail exponent ---
    self.rho   = nn.Parameter(torch.zeros(dim))  # rho=0 -> p=1
    self.p_min = p_min
    self.p_max = p_max

  def forward(self, x):
    # bulk gate: a0 + sum_k w_k sigmoid(beta_k (x - mu_k)).
    # unsqueeze(-2) adds a size-1 axis before the last
    # ((..., dim) -> (..., 1, dim)), broadcasting x against the
    # K gates (shape (K, dim)) -> (..., K, dim).
    xx   = x.unsqueeze(-2)               # (..., 1, dim)
    s    = torch.sigmoid(self.beta * (xx - self.mu))
    gate = self.a0 + (self.w * s).sum(-2)   # (..., dim)
    # bounded learnable tail exponent in (p_min, p_max).
    p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(
      self.rho)
    # signed power: linear (slope 1) near 0, ~|x|^p in the tail.
    ax  = x.abs()
    psi = torch.sign(x) * ((1.0 + ax) ** p - 1.0) / p
    return gate * psi


def make_activation(name, n_gates=3):
  """
  Activation factory by name, for a ResBlock's `act` slot.

  Maps a short name to a factory callable act(dim) -> module -- the
  contract ResBlock's `act` expects (it calls act(size) once per
  layer) -- letting a driver or YAML pick the activation by string
  rather than importing a class. The gated families use
  K = n_gates gates.

  Arguments:
    name    = one of:
                "H"           -> activation_fcn, the paper's H
                                 (also the ResBlock default).
                "power"       -> PowerGatedActivation (bounded
                                 learnable tail exponent).
                "multigate"   -> GatedActivation (K = n_gates).
                "gated_power" -> GatedPowerActivation (K gates plus
                                 the tail exponent).
    n_gates = number of gates K for the multi-gate families
              (default 3); ignored by "H" and "power".

  Returns:
    a factory act(dim) -> nn.Module.
  """
  if name == "H":
    return activation_fcn
  if name == "power":
    return lambda dim: PowerGatedActivation(dim)
  if name == "multigate":
    return lambda dim: GatedActivation(dim, n_gates=n_gates)
  if name == "gated_power":
    return lambda dim: GatedPowerActivation(dim, n_gates=n_gates)
  raise ValueError(f"unknown activation: {name!r}")
