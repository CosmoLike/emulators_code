"""Shared nn building blocks (Affine, ResBlock, rescale_kernel_size,
FiLMGenerator, BinLinear, TRFBlock).

The small nn.Modules the emulator models (emulator_designs.py) are
assembled from. Where each piece sits:

  ResMLP = Linear -> n_blocks x ResBlock -> Linear -> Affine
  ResCNN = ResMLP trunk + conv correction head (bare nn.Conv1d
             layers, needing no block here; the rescale_kernel flag
             resolves the width through rescale_kernel_size)
  ResTRF = ResMLP trunk + TRFBlock correction head
             (per-token unique MLPs = BinLinear)

Affine is a learnable scalar scale and shift (the default ResBlock
"norm" and the models' final layer). ResBlock is a width-preserving
residual block (n dense layers, each with a norm and activation
factory, skip added before the last). rescale_kernel_size shrinks
the conv heads' kernel as their depth grows, preserving a single
block's receptive field. FiLMGenerator predicts the conv heads'
optional per-channel, cosmology-dependent modulation (the film
flag). BinLinear and TRFBlock are the ResTRF
head's pieces: per-token unique linears and a transformer block
whose tokens are the tomographic bins. Grouped / per-bin conv twins
live in parallel/.
"""

import torch
import torch.nn as nn

from .activations import activation_fcn


class Affine(nn.Module):
    """
    A learnable scalar scale and shift: out = x * gain + bias.

    gain and bias are single scalars (shape (1,)) broadcast over
    every element of x: one global scale and shift, not a
    per-feature transform. gain inits to 1, bias to 0, so at init it
    is the identity. Used as the ResBlock default "norm" factory and
    the final layer of ResMLP / ResCNN.

    Both are nn.Parameter, registered and trained. Weight decay is
    kept off both (make_optimizer decays only ndim >= 2 weight
    matrices): decaying gain toward 0 would attenuate the signal,
    and decaying a bias has no principled meaning.
    """
    def __init__(self):
        super(Affine, self).__init__()
        # one learnable scale (gain, init 1) and shift (bias, init
        # 0), each a scalar broadcast over all of the input.
        self.gain = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        # elementwise: every entry scaled by gain, shifted by bias
        # (both broadcast from their size-1 shape).
        return x * self.gain + self.bias


class ResBlock(nn.Module):
  """
  Width-preserving residual block: n_layers dense layers between
  two skip points, the input added back to the last layer's output
  before its norm and activation. Input and output share one width
  by design, so the skip connection is the identity (no projection
  layer needed):

    x ─┬─ Linear ─ norm ─ act ─ ... ─ Linear ─(+)─ norm ─ act ─> out
       └─────────────── identity skip ──────────┘

  Arguments:
    size     = feature width, shared by input and output.
    n_layers = number of dense layers between two skip points.
    norm     = normalization factory, invoked as norm(size).
    act      = activation factory, invoked as act(size).

  norm and act are factories, not ready-made modules: each is
  invoked once per dense layer so every layer holds an independent
  module. A shared instance would couple the layers' learnable
  normalization parameters.

  Factory examples:
    norm = nn.BatchNorm1d       (accepts size)
    norm = lambda s: Affine()   (Affine accepts no size)
    act  = activation_fcn       (accepts size)
    act  = lambda s: nn.Tanh()  (Tanh accepts no size)
  """
  def __init__(self,
               size,
               n_layers = 2,
               norm = lambda s: Affine(),
               act = activation_fcn):
    super().__init__()
    self.skip = nn.Identity()

    # Sublayers go in nn.ModuleList, not a plain list or numbered
    # attributes: ModuleList registers each submodule with the
    # parent, so its parameters appear in .parameters(), transfer
    # under .to(device), and are saved in the state_dict. Build the
    # n_layers dense layers, norms, and activations in one loop;
    # each is its own module (fresh norm / act per layer, never
    # shared).
    layers, norms, acts = [], [], []
    for _ in range(n_layers):
      layers.append(nn.Linear(in_features=size, out_features=size))
      norms.append(norm(size))
      acts.append(act(size))
    self.layers = nn.ModuleList(layers)
    self.norms  = nn.ModuleList(norms)
    self.acts   = nn.ModuleList(acts)

  def forward(self, x):
    xskip = self.skip(x)
    out = x
    n = len(self.layers)
    for i in range(n):
      out = self.layers[i](out)
      # Skip added to the final linear layer's output, before its
      # norm and activation (a pre-activation residual addition).
      if i == n - 1:
        out = out + xskip
      out = self.acts[i](self.norms[i](out))
    return out


def rescale_kernel_size(kernel_size, n_blocks_cnn):
  """
  Per-block kernel width that preserves a single block's view as
  the conv head deepens.

  kernel_size is read as the width one block alone would use, so it
  states the head's target total view: a single same-padded conv of
  width k sees k positions. n stacked same-padded convs of width
  k_n see

    RF = n * (k_n - 1) + 1

  Why: write r = (k_n - 1)/2 for the kernel radius (odd kernel,
  same-padded). One conv's output at position p is a weighted sum
  of the k_n inputs p-r .. p+r -- one layer sees k_n positions.
  Stack a second conv: its output at p reads the k_n layer-1
  positions p-r .. p+r, and each of those sees its own k_n-wide
  input window. Consecutive layer-1 positions hold windows shifted
  by exactly one column (stride 1), so the union of the k_n
  windows is one contiguous window, k_n - 1 columns wider than
  each. Drawn at k_n = 3 (r = 1):

    y[p]                          layer-2 output: taps
     │         │         │        h[p-1], h[p], h[p+1]
    h[p-1]    h[p]     h[p+1]     layer 1: three k_n-wide input
     │         │         │        windows, each shifted by one
     ▼         ▼         ▼        column
    x[p-2..p] x[p-1..p+1] x[p..p+2]
                                  union = x[p-2 .. p+2]:
                                  5 = 2*(k_n - 1) + 1 positions

  Every extra layer therefore adds the same r of reach per side --
  the new kernel's outermost tap already sits r columns out, and
  looks through a window extending r further. The growth is
  additive, never multiplicative:

    RF_1 = k_n;   RF_(i+1) = RF_i + (k_n - 1)
    =>  RF_n = k_n + (n - 1)(k_n - 1) = n*(k_n - 1) + 1

               y[p]               layer n
              ╱    ╲
          ..........              the cone widens by r per side,
         ╱          ╲             per layer
    x[p - n*r] .. x[p + n*r]      input window = 2*n*r + 1
                                               = n*(k_n - 1) + 1

  (Same-padding does not change the count: where the cone hangs
  past the signal's edge it reads padded zeros, not extra data.)

  Without rescaling, deepening the stack over-grows the view (3
  blocks of k = 11, r = 5, reach 3*5 = 15 per side: RF = 31, wider
  than a whole 26-point bin). This helper instead solves RF >=
  kernel_size for the smallest odd k_n (same-padding needs odd):

    k_n = ceil((kernel_size - 1) / n_blocks_cnn) + 1, then odd-up.

  Extra depth then buys nonlinearity at a fixed total view. It also
  keeps the head size nearly flat: per-block conv parameters scale
  with k_n (C_in*C_out*k_n + C_out), so the head total ~
  C^2 * n * k_n ~ C^2 * (kernel_size - 1 + 2n). At kernel_size 27
  (one bin + margin at the LSST-Y1 run's max_bin = 26):

    n_blocks_cnn : 1    2    3    4    5
    k_n          : 27   15   11   9    7
    RF           : 27   29   31   33   31

  (RF overshoots kernel_size where odd-up rounds; it never
  undershoots.)

  Arguments:
    kernel_size  = the single-block kernel width = the target
                   receptive field (odd).
    n_blocks_cnn = number of stacked conv+activation blocks.

  Returns:
    the per-block kernel width k_n (odd int; = kernel_size when
    n_blocks_cnn is 1).
  """
  # ceil((kernel_size-1)/n) via negated floor division (pure ints,
  # no float rounding), + 1 = the smallest k_n with RF >=
  # kernel_size.
  k = -(-(kernel_size - 1) // n_blocks_cnn) + 1
  # odd-up: same-padding needs an odd kernel to keep the length.
  if k % 2 == 0:
    k += 1
  return k


class FiLMGenerator(nn.Module):
  """
  Predicts a per-channel affine modulation (gamma, beta) from a
  conditioning vector -- the generator half of FiLM (Feature-wise
  Linear Modulation; Dumoulin et al. 2018, and the recovered
  design note notes/film-conditioning.md).

    z  (B, n_cond)            conditioning vector (here: the
       │                      NON-amplitude whitened parameters)
       │  Linear(n_cond, 2*C)
       ▼
    out (B, 2*C)
       │  split at C
       ▼
    gamma (B, C), beta (B, C)     one scale + one shift PER
                                  CHANNEL, per sample

  The caller applies them to a feature map h of shape (B, C, L) as

    gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)

  broadcasting over the length axis: the modulation depends on the
  cosmology and the channel, never on the position -- cosmology
  sets a global property of each channel's piece of the data
  vector, not a per-theta local correction. This is what
  re-injects parameter information into a correction head that
  otherwise only ever sees the trunk's output: without FiLM the
  head is one fixed map applied identically at every point of
  parameter space; with it, the cosmology chooses which channels
  to amplify or suppress, and by how much.

  Identity at init: the weight is zeroed and the bias set to
  gamma = 1, beta = 0, so FiLM starts as a no-op for every input
  -- the same identity-start convention as the zero-init conv and
  the TRFBlock branches (the model still equals its trunk exactly
  at epoch 1 and at a two-phase handoff). Gradients reach the
  zeroed weight through the inputs, so it wakes as soon as a
  cosmology-dependent modulation helps.

  (legend: B = batch rows; n_cond = conditioning width -- the
  factored heads pass the non-amplitude parameter slice, keeping
  the head amplitude-blind so the closed-form amplitude exactness
  survives; C = number of channels to modulate; L = the broadcast
  length axis, max_bin here.)

  Arguments:
    n_cond     = conditioning-vector width.
    n_channels = number of channels C to modulate.
  """
  def __init__(self, n_cond, n_channels):
    super().__init__()
    self.n_channels = n_channels
    # one linear producing both halves at once: columns [:C] are
    # gamma, [C:] are beta.
    self.linear = nn.Linear(in_features=n_cond,
                            out_features=2 * n_channels)
    # identity init: zero weight kills the z-dependence, the bias
    # supplies gamma = 1 / beta = 0 (init fns run under no_grad;
    # the slices are views into the one bias parameter).
    nn.init.zeros_(self.linear.weight)
    nn.init.ones_(self.linear.bias[:n_channels])
    nn.init.zeros_(self.linear.bias[n_channels:])

  def forward(self, z):
    # z: (B, n_cond) -> (B, 2C), split into the two halves.
    out = self.linear(z)
    gamma = out[:, :self.n_channels]    # (B, C)
    beta  = out[:, self.n_channels:]    # (B, C)
    return gamma, beta


class BinLinear(nn.Module):
  """
  G independent Linear(in_features, out_features) layers -- one per
  token -- run as a single batched einsum instead of a Python loop
  over G modules. The weights stack into (G, in, out), the biases
  into (G, out); token g's rows only ever meet weight[g].

  This is the "unique per token" piece of the ResTRF head: a
  standard transformer applies one shared MLP to every token,
  whereas here each token gets its own weights. The tokens are
  physically distinct -- a tomographic bin (plain ResTRF) or a
  (template, bin) pair (the factored version) -- and the unique
  weights also make them distinguishable to the model, doing the
  job a positional encoding does in a standard transformer, so
  ResTRF needs none.

  These per-token layers live in the correction head, after
  attention has shared information across tokens -- the trunk's
  parameter sharing (the expensive cosmology map, learned once) is
  untouched.

  Arguments:
    n_tokens     = number of independent tokens G.
    in_features  = input width per token.
    out_features = output width per token.
  """
  def __init__(self, n_tokens, in_features, out_features):
    super().__init__()
    # build G ordinary nn.Linear layers just to borrow their init,
    # then stack their weights/biases and discard them. l.weight is
    # (out, in); .t() -> (in, out) for the einsum; stack adds the
    # token axis.
    lins = []
    for _ in range(n_tokens):
      lins.append(nn.Linear(in_features=in_features,
                            out_features=out_features))
    weights, biases = [], []
    for l in lins:
      weights.append(l.weight.detach().t())
      biases.append(l.bias.detach())
    self.weight = nn.Parameter(torch.stack(weights))   # (G, in, out)
    self.bias   = nn.Parameter(torch.stack(biases))    # (G, out)

  def forward(self, x):
    # x: (B, G, in). einsum("bgi,gio->bgo", x, weight): g appears in
    # both operands and the output, so it is a batch axis -- token g
    # uses weight[g] only, all G in one batched matmul; i appears in
    # both inputs but not the output, so einsum sums over it (the
    # matmul contraction); b and o are kept.
    y = torch.einsum("bgi,gio->bgo", x, self.weight)
    # bias (G, out) broadcasts over the B axis: every sample's token
    # g gets token g's bias.
    return y + self.bias


class TRFBlock(nn.Module):
  """
  One transformer block over tokens at their natural width: no
  embedding in, no projection out -- the tokens are the (padded)
  physical bin segments themselves, so dim = max_bin, the padded
  bin length. (A learned embedding is what a transformer needs when
  its sequence is synthetic -- a flat latent vector split into
  tokens; here the sequence structure is physical, so the adapter
  layers and their parameters are simply not needed.)
  Self-attention across the G tokens, then a per-token MLP branch
  -- both pre-norm residual branches, as in a standard pre-LN
  transformer:

    x  (B, G, dim)             G tokens (bins) of width dim
       │  LayerNorm; wq / wk / wv        (shared across tokens)
       ▼
    q, k, v  (B, G, H, d_head)           H heads, d_head = dim/H
       │  scores = q.k / sqrt(d_head); softmax over the key axis
       ▼
    att  (B, H, G, G)          per head: each query bin's weights
       │                       over all key bins
       │  att @ v; merge heads; wo       (wo zero-initialized)
       ▼
    x + attention branch
       │  LayerNorm; n_mlp_blocks x [BinLinear + act]
       │                                 (last layer zero-init)
       ▼
    x + MLP branch             the block's output (= x at init)

  (legend: B = batch rows; G = n_tokens, the number of tokens; dim
  = the per-token width; H = n_heads; d_head = dim/H, the feature
  slice each head works in.)

  Two deliberate deviations from the textbook block:
  - the tokens are physical: a tomographic bin's theta segment
    (plain ResTRF) or a (template, bin) pair's (the factored
    version), so attention shares information across bins (the
    cross-bin correlations a within-bin conv cannot see);
  - the position-wise MLP is not shared (by default): each token
    has its own n_mlp_blocks-deep stack (BinLinear), where a
    standard transformer applies one shared MLP to every token.
    The unique weights specialize each token's correction and
    stand in for the positional encoding (see BinLinear).
    shared_mlp=True restores the textbook shared MLP -- the
    ablation baseline isolating that deviation. Caveat: with the
    MLP shared (and the attention maps always shared), nothing in
    the block tells the tokens apart structurally -- the head
    becomes permutation-equivariant over tokens, with no
    positional encoding; token identity then comes only from the
    segments' content.

  The attention projections (wq / wk / wv / wo) are shared across
  tokens, as in any transformer -- shared maps are what let every
  token attend to every other with one set of weights; the
  per-token specialization lives in the MLPs.

  The block is exactly the identity at init: both branch outputs
  (wo and the last MLP layer) are zero-initialized, so x passes
  through untouched. A stack of these blocks therefore satisfies
  blocks(x) == x at init, which is what lets the ResTRF head
  define its correction as blocks(h) - h == 0 -- the zero-init
  identity start, with no output projection to host it. Gradients
  still reach the zeroed layers (a layer's weight gradient depends
  on its inputs, not on its own weights); the layers behind them
  wake one step later.

  LayerNorm (not the package's Affine) opens both branches: the
  softmax's saturation depends on the score scale, so attention
  wants its inputs actively normalized, and pre-LN is the
  stable-training default for transformers.

  Arguments:
    dim          = token width = max_bin, the padded bin length
                   (must be divisible by n_heads; the LSST-Y1
                   cosmic-shear run keeps max_bin = 26 theta
                   points per bin, allowing n_heads = 1, 2, or
                   13).
    n_tokens     = number of tokens G.
    n_heads      = attention heads (each head attends over all G
                   tokens with dim/n_heads of the features).
    n_mlp_blocks = depth of each token's private MLP stack.
    act          = activation factory act(dim) -> module for the
                   MLP layers (the run's activation; defaults to
                   activation_fcn, the paper's H).
    shared_mlp   = False (default): per-token unique MLPs
                   (BinLinear). True: one MLP shared by every
                   token (plain nn.Linear applied position-wise)
                   -- the textbook block, see the caveat above.
  """
  def __init__(self, dim, n_tokens, n_heads=2, n_mlp_blocks=2,
               act=activation_fcn, shared_mlp=False):
    super().__init__()
    assert dim % n_heads == 0, (
      f"the token width ({dim} = the padded bin length) must be "
      f"divisible by n_heads ({n_heads})")
    self.n_heads = n_heads
    self.d_head  = dim // n_heads

    # attention branch: pre-norm, shared Q/K/V/output projections.
    self.ln_att = nn.LayerNorm(dim)
    self.wq = nn.Linear(in_features=dim, out_features=dim)
    self.wk = nn.Linear(in_features=dim, out_features=dim)
    self.wv = nn.Linear(in_features=dim, out_features=dim)
    self.wo = nn.Linear(in_features=dim, out_features=dim)

    # MLP branch: pre-norm, n_mlp_blocks layers, each its own
    # activation instance. Per-token unique (BinLinear) by default;
    # with shared_mlp one nn.Linear serves every token (a Linear on
    # a (B, G, dim) tensor applies position-wise to the last axis,
    # which is exactly the textbook transformer FFN).
    self.ln_mlp = nn.LayerNorm(dim)
    lins, acts = [], []
    for _ in range(n_mlp_blocks):
      if shared_mlp:
        lins.append(nn.Linear(in_features=dim, out_features=dim))
      else:
        lins.append(BinLinear(n_tokens, dim, dim))
      acts.append(act(dim))
    self.mlp_lins = nn.ModuleList(lins)
    self.mlp_acts = nn.ModuleList(acts)

    # identity at init: zero both branch outputs (see docstring).
    # The final MLP activation maps 0 -> 0 (H(x) = gate(x)*x), so a
    # zeroed last layer silences the whole branch.
    nn.init.zeros_(self.wo.weight)
    nn.init.zeros_(self.wo.bias)
    nn.init.zeros_(self.mlp_lins[-1].weight)
    nn.init.zeros_(self.mlp_lins[-1].bias)

  def forward(self, x):
    # x: (B, G, dim) -- G tokens of width dim.
    B, G, _ = x.shape

    # --- attention branch (pre-LN residual) ---
    h = self.ln_att(x)
    # split the feature axis into heads: (B, G, dim) -> (B, G, H,
    # d_head); view is free (the Linear output is contiguous).
    q = self.wq(h).view(B, G, self.n_heads, self.d_head)
    k = self.wk(h).view(B, G, self.n_heads, self.d_head)
    v = self.wv(h).view(B, G, self.n_heads, self.d_head)
    # attention scores, einsum("bghd,bkhd->bhgk"): d is contracted
    # (the query-key dot product), b and h are batch axes, and the
    # kept g (query bin) x k (key bin) pair is the GxG attention
    # matrix per head. Divided by sqrt(d_head) so the dot products
    # stay O(1) and the softmax does not saturate at init.
    att = torch.einsum("bghd,bkhd->bhgk", q, k) / self.d_head ** 0.5
    # softmax over the key axis: each query bin's weights over all
    # bins sum to 1.
    att = torch.softmax(att, dim=-1)
    # weighted sum of the value tokens, einsum("bhgk,bkhd->bghd"):
    # k is contracted against each query's attention row; the
    # result is one mixed d_head vector per (query bin, head).
    out = torch.einsum("bhgk,bkhd->bghd", att, v)
    # merge the heads back: (B, G, H, d_head) -> (B, G, dim).
    # reshape (not view): the einsum output need not be contiguous.
    out = out.reshape(B, G, self.n_heads * self.d_head)
    x = x + self.wo(out)

    # --- per-bin MLP branch (pre-LN residual) ---
    h = self.ln_mlp(x)
    n = len(self.mlp_lins)
    for i in range(n):
      h = self.mlp_acts[i](self.mlp_lins[i](h))
    return x + h
