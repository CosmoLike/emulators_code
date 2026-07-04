"""Shared nn building blocks (Affine, ResBlock, BinLinear, TRFBlock).

The small nn.Modules the emulator models are assembled from. Affine is a
learnable per-output scale and shift (the default ResBlock "norm" and the
models' final layer). ResBlock is a width-preserving residual block (n
dense layers, each with a norm and activation factory, skip added before
the last). BinLinear and TRFBlock are the ResTRF head's pieces: per-bin
unique linears and a transformer block whose tokens are the tomographic
bins. (ResCNN's bins-as-channels conv head is a bare nn.Conv1d, needing
no block here.) Grouped / per-bin conv twins live in parallel/.
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


class BinLinear(nn.Module):
  """
  G independent Linear(in_features, out_features) layers -- one per
  tomographic bin -- run as a single batched einsum instead of a
  Python loop over G modules. The weights stack into (G, in, out),
  the biases into (G, out); bin g's rows only ever meet weight[g].

  This is the "unique per bin" piece of the ResTRF head: a standard
  transformer applies ONE shared MLP to every token, whereas here
  each bin (token) gets its own weights. The bins are physically
  distinct (different source pairs and xi+/- branches), and the
  unique weights also make them distinguishable to the model --
  doing the job a positional encoding does in a standard
  transformer, so ResTRF needs none.

  These per-bin layers live in the correction HEAD, after attention
  has shared information across bins -- the trunk's parameter
  sharing (the expensive cosmology map, learned once) is untouched.

  Arguments:
    n_bins       = number of independent bins G (= tokens).
    in_features  = input width per bin.
    out_features = output width per bin.
  """
  def __init__(self, n_bins, in_features, out_features):
    super().__init__()
    # build G ordinary nn.Linear layers just to borrow their init,
    # then stack their weights/biases and discard them. l.weight is
    # (out, in); .t() -> (in, out) for the einsum; stack adds the
    # bin axis.
    lins = []
    for _ in range(n_bins):
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
    # both operands and the output, so it is a batch axis -- bin g
    # uses weight[g] only, all G in one batched matmul; i appears in
    # both inputs but not the output, so einsum sums over it (the
    # matmul contraction); b and o are kept.
    y = torch.einsum("bgi,gio->bgo", x, self.weight)
    # bias (G, out) broadcasts over the B axis: every sample's bin g
    # gets bin g's bias.
    return y + self.bias


class TRFBlock(nn.Module):
  """
  One transformer block over bin tokens: self-attention across the
  G bins, then a per-bin MLP branch -- both pre-norm residual
  branches, as in a standard pre-LN transformer.

  Two deliberate deviations from the textbook block:
  - the TOKENS are the tomographic bins: each (xi+/-, source-pair)
    bin's theta segment is one token, so attention shares
    information across bins (the cross-bin correlations a per-bin
    conv cannot see);
  - the position-wise MLP is NOT shared: each bin has its own
    n_mlp_blocks-deep stack (BinLinear), where a standard
    transformer applies one shared MLP to every token. The unique
    weights specialize each bin's correction and stand in for the
    positional encoding (see BinLinear).

  The attention projections (wq / wk / wv / wo) ARE shared across
  bins, as in any transformer -- shared maps are what let every bin
  attend to every other with one set of weights; the per-bin
  specialization lives in the MLPs.

  LayerNorm (not the package's Affine) opens both branches: the
  softmax's saturation depends on the score scale, so attention
  wants its inputs actively normalized, and pre-LN is the
  stable-training default for transformers.

  Arguments:
    dim          = token embedding width (must divide by n_heads).
    n_bins       = number of bin tokens G.
    n_heads      = attention heads (each head attends over all G
                   bins with dim/n_heads of the features).
    n_mlp_blocks = depth of each bin's private MLP stack.
    act          = activation factory act(dim) -> module for the MLP
                   layers (the run's activation; defaults to
                   activation_fcn, the paper's H).
  """
  def __init__(self, dim, n_bins, n_heads=4, n_mlp_blocks=2,
               act=activation_fcn):
    super().__init__()
    assert dim % n_heads == 0, (
      f"int_dim_trf ({dim}) must be divisible by n_heads ({n_heads})")
    self.n_heads = n_heads
    self.d_head  = dim // n_heads

    # attention branch: pre-norm, shared Q/K/V/output projections.
    self.ln_att = nn.LayerNorm(dim)
    self.wq = nn.Linear(in_features=dim, out_features=dim)
    self.wk = nn.Linear(in_features=dim, out_features=dim)
    self.wv = nn.Linear(in_features=dim, out_features=dim)
    self.wo = nn.Linear(in_features=dim, out_features=dim)

    # per-bin MLP branch: pre-norm, n_mlp_blocks unique layers per
    # bin, each its own activation instance.
    self.ln_mlp = nn.LayerNorm(dim)
    lins, acts = [], []
    for _ in range(n_mlp_blocks):
      lins.append(BinLinear(n_bins, dim, dim))
      acts.append(act(dim))
    self.mlp_lins = nn.ModuleList(lins)
    self.mlp_acts = nn.ModuleList(acts)

  def forward(self, x):
    # x: (B, G, dim) -- G bin tokens of width dim.
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
