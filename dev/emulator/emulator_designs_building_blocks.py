"""Shared nn building blocks (Affine, ResBlock, BinLinear, TRFBlock).

The small nn.Modules the emulator models are assembled from. Affine is a
learnable per-output scale and shift (the default ResBlock "norm" and the
models' final layer). ResBlock is a width-preserving residual block (n
dense layers, each with a norm and activation factory, skip added before
the last). BinLinear and TRFBlock are the ResTRF head's pieces: per-bin
unique linears and a transformer block whose tokens are the tomographic
bins. conv1d_as_matmul runs the ResCNN heads' bare nn.Conv1d layers as
a single matmul (same parameters, same output; the heads' conv shape is
pathologically slow on the native conv kernels). Grouped / per-bin conv
twins live in parallel/.
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
  # Residual block. Input and output share one width by design, so
  # the skip connection is the identity.
  #
  # Arguments:
  #   size = feature width, shared by input and output
  #   n_layers = number of dense layers between two skip points
  #   norm = normalization factory, invoked as norm(size)
  #   act = activation factory, invoked as act(size)
  #
  # norm and act are factories, not ready-made modules: each is
  # invoked once per dense layer so every layer holds an independent
  # module. A shared instance would couple the layers' learnable
  # normalization parameters.
  #
  # Factory examples:
  #   norm = nn.BatchNorm1d       (accepts size)
  #   norm = lambda s: Affine()   (Affine accepts no size)
  #   act = activation_fcn        (accepts size)
  #   act = lambda s: nn.Tanh()   (Tanh accepts no size)
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


def conv1d_as_matmul(conv, x):
  """
  Run an nn.Conv1d as one matmul -- same parameters, same output,
  matmul-shaped compute.

  The correction heads' conv shape (moderate channels over a tiny
  length: 90 -> 90 channels, 26 theta positions) sits outside every
  fast conv path and benchmarks at ~1% of matmul throughput -- the
  conv alone was most of a head-phase epoch. A convolution IS a
  linear map, so express it as one: pad, slide a kernel-wide window
  along theta, and contract each position's (C_in x K) receptive
  field against the flattened conv weight in a single
  (B*L, C_in*K) @ (C_in*K, C_out) matmul. Output identical to
  conv(x) to float precision; ~25x faster forward at the head shape
  on CPU (~5x on the whole head-phase training step).

  The parameters still live in the nn.Conv1d passed in -- same
  state_dict, same checkpoints, same optimizer groups; only the
  compute path changes.

  Arguments:
    conv = an nn.Conv1d with stride/dilation/groups = 1 and
           symmetric same-padding (what the ResCNN heads build).
    x    = (B, C_in, L) input.

  Returns:
    (B, C_out, L) tensor, contiguous, equal to conv(x).
  """
  K   = conv.kernel_size[0]
  pad = conv.padding[0]
  B, C, L = x.shape
  xp = nn.functional.pad(x, (pad, pad))     # (B, C, L + 2*pad)
  # unfold along the length axis: a strided VIEW (no copy) of shape
  # (B, C, L, K) whose [b, c, l] slot is the K-window xp[b, c, l:l+K]
  # -- position l's receptive field in channel c.
  xu = xp.unfold(2, K, 1)
  # gather each position's full (C, K) receptive field into one row:
  # (B, L, C, K) -> (B*L, C*K). The reshape after the permute is
  # where the one real copy happens.
  m = xu.permute(0, 2, 1, 3).reshape(B * L, C * K)
  # conv.weight is (C_out, C_in, K); flattened to (C_out, C*K) its
  # rows match m's columns, so one GEMM computes every output
  # position and channel at once.
  y = m @ conv.weight.reshape(conv.out_channels, C * K).t()
  y = y + conv.bias
  # (B*L, C_out) -> (B, C_out, L); contiguous so downstream .view
  # calls work.
  return y.view(B, L, conv.out_channels).permute(0, 2, 1).contiguous()


class BinLinear(nn.Module):
  """
  G independent Linear(in_features, out_features) layers -- one per
  token -- run as a single batched einsum instead of a Python loop
  over G modules. The weights stack into (G, in, out), the biases
  into (G, out); token g's rows only ever meet weight[g].

  This is the "unique per token" piece of the ResTRF head: a
  standard transformer applies ONE shared MLP to every token,
  whereas here each token gets its own weights. The tokens are
  physically distinct -- a tomographic bin (plain ResTRF) or a
  (template, bin) pair (the factored version) -- and the unique
  weights also make them distinguishable to the model, doing the
  job a positional encoding does in a standard transformer, so
  ResTRF needs none.

  These per-token layers live in the correction HEAD, after
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
  One transformer block over tokens at their NATURAL width: no
  embedding in, no projection out -- the tokens are the (padded)
  physical bin segments themselves, so dim = the bin length. (A
  learned embedding is what a transformer needs when its sequence
  is synthetic -- a latent split into tokens; here the sequence
  structure is physical, so the adapter layers and their
  parameters are simply not needed.) Self-attention across the G
  tokens, then a per-token MLP branch -- both pre-norm residual
  branches, as in a standard pre-LN transformer.

  Two deliberate deviations from the textbook block:
  - the TOKENS are physical: a tomographic bin's theta segment
    (plain ResTRF) or a (template, bin) pair's (the factored
    version), so attention shares information across bins (the
    cross-bin correlations a within-bin conv cannot see);
  - the position-wise MLP is NOT shared (by default): each token
    has its own n_mlp_blocks-deep stack (BinLinear), where a
    standard transformer applies one shared MLP to every token.
    The unique weights specialize each token's correction and
    stand in for the positional encoding (see BinLinear).
    shared_mlp=True restores the textbook shared MLP -- the
    ablation baseline isolating that deviation. Caveat: with the
    MLP shared (and the attention maps always shared), NOTHING in
    the block tells the tokens apart structurally -- the head
    becomes permutation-equivariant over tokens, with no
    positional encoding; token identity then comes only from the
    segments' content.

  The attention projections (wq / wk / wv / wo) ARE shared across
  tokens, as in any transformer -- shared maps are what let every
  token attend to every other with one set of weights; the
  per-token specialization lives in the MLPs.

  The block is EXACTLY the identity at init: both branch outputs
  (wo and the last MLP layer) are zero-initialized, so x passes
  through untouched. A stack of these blocks therefore satisfies
  blocks(x) == x at init, which is what lets the ResTRF head
  define its correction as blocks(h) - h == 0 -- the zero-init
  identity start, with no output projection to host it. Gradients
  still reach the zeroed layers (their grads depend on their
  INPUTS, not their weights); the layers behind them wake one
  step later.

  LayerNorm (not the package's Affine) opens both branches: the
  softmax's saturation depends on the score scale, so attention
  wants its inputs actively normalized, and pre-LN is the
  stable-training default for transformers.

  Arguments:
    dim          = token width = the padded bin length (must divide
                   by n_heads; a bin length of 26 allows 1 / 2 /
                   13).
    n_tokens     = number of tokens G.
    n_heads      = attention heads (each head attends over all G
                   tokens with dim/n_heads of the features).
    n_mlp_blocks = depth of each token's private MLP stack.
    act          = activation factory act(dim) -> module for the MLP
                   layers (the run's activation; defaults to
                   activation_fcn, the paper's H).
    shared_mlp   = False (default): per-token unique MLPs
                   (BinLinear). True: ONE MLP shared by every token
                   (plain nn.Linear applied position-wise) -- the
                   textbook block, see the caveat above.
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

    # identity at init: zero both branch OUTPUTS (see docstring).
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
    # softmax over the KEY axis: each query bin's weights over all
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
