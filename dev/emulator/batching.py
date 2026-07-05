"""Memory sizing and the regime-aware data loaders.

Decides where each source's data lives and hands the training loop two
closures (rows -> whitened param inputs, rows -> encoded targets) that
hide it. compute_batch_size_bytes, compute_model_size_bytes, and
batches_per_load estimate per-batch and resident memory.
_build_loaders_one picks one of three regimes against a VRAM budget
(pre-encode the target set on the GPU, stream from RAM, or stream from a
disk memmap) and reports the bytes it made resident. build_loaders runs it
per source (train, then val against the reduced budget) and returns the
data dict the loop consumes.

The regime ladder, per source:

    dv rows (disk memmap or RAM)      raw params C
       │                                 │  param_geometry.encode
       │                                 ▼
       │                              C_used (n_used, Ncosmo)
       │                              resident on the GPU
       │
       │  enc_dvs + resident < 0.8 * budget ?
       │
       ├─ yes -> regime 1 (resident): encode every target once
       │         (chunked), hold (n_used, tgt_dim) on the GPU;
       │         load_dv(rows) is a pure index, no per-epoch I/O.
       ├─ no, dv is a RAM ndarray -> regime 2 (RAM stream):
       │         chunks RAM -> GPU each epoch, encoded on the
       │         fly (pinned host memory on CUDA).
       └─ no, dv is a disk memmap -> regime 3 (disk stream):
                 the same chunk path, reads hit the disk.

    (legend: n_used = distinct rows this source loads; Ncosmo =
     parameter count; enc_dvs = bytes of the encoded target set;
     resident = model + Cinv + encoded params, bytes pinned on
     the GPU for the whole run; tgt_dim = target width, out_dim
     unless the loss stages a wider one via target_dim; budget =
     the VRAM bytes this source may plan against.)

PS: a loader is a closure load(rows) -> tensor mapping global row indices
to a ready-to-train batch on the compute device. It hides where the data
lives (resident on the GPU, streamed from RAM, or read from a disk
memmap), so the training loop just asks for rows and gets device tensors
back, the same in every regime. Two loaders per source: load_C for
whitened param inputs, load_dv for encoded targets. whitened = rotated
into the covariance eigenbasis and scaled to unit variance, so the
components are decorrelated (the form the network sees). encoded = a data
vector through the geometry's encode (keep unmasked entries, subtract the
training mean, whiten), the form trained against. resident = held in GPU
memory for the whole run, not re-loaded per batch.
"""

import numpy as np
import torch


def compute_batch_size_bytes(model, bs, sample_dims, dv_len=3000):
  """
  Estimate the GPU bytes one training batch costs.

  Measures the autograd-saved activations of a real forward pass
  (a spy pair of saved-tensor hooks, see the body) and adds the
  batch input/output buffers and the chi2's per-batch float64
  scratch. Only shapes matter, so the probe runs on zeros.

  Arguments:
    model       = the network; probed with one dummy forward.
    bs          = minibatch size the estimate is for.
    sample_dims = shape of one model input, no batch axis (the
                  cosmo param vector: (Ncosmo,)).
    dv_len      = full dv length the chi2 un-squeezes to (~3000
                  is conservative; avoids a cosmolike query).

  Returns:
    estimated bytes per batch (saved activations + I/O buffers +
    chi2 scratch).
  """
  # model.parameters() iterates the weight tensors; next() grabs
  # the first. Its .device is where it lives; put x there too.
  dev = next(model.parameters()).device

  # dummy input batch (bs, *sample_dims). Only shapes matter for
  # memory, not values -> zeros.
  x = torch.zeros(bs, *sample_dims, device=dev)

  total = 0  # running byte count of saved tensors
  def pack(t):
    # pack: first callback. autograd calls it the moment a
    # tensor is saved during forward and stores what pack
    # returns. Record the size, return t unchanged. (+= alone
    # would rebind total as a local; nonlocal points it at the
    # outer total.)
    nonlocal total
    total += t.numel() * t.element_size()
    return t
  def unpack(stored):
    # unpack: second callback, called in backward to rebuild the
    # saved tensor. We never call backward, but
    # saved_tensors_hooks requires it, so hand stored back.
    return stored

  # saved_tensors_hooks customizes how activations are stored
  # between forward and backward (to save memory): pack
  # transforms each on the way in (e.g. compress), unpack
  # reverses it out. The pair must round-trip: unpack(pack(t))
  # == t. Here the no-op pair just spies on the sizes. See the
  # PyTorch saved-tensor-hooks tutorial (URL split to fit the
  # width; rejoin with no space):
  #   https://docs.pytorch.org/tutorials/intermediate/
  #   autograd_saved_tensors_hooks_tutorial.html
  hooks = torch.autograd.graph.saved_tensors_hooks  # alias
  with hooks(pack, unpack):
    # saving happens during forward, so total is complete the
    # instant model(x) returns.
    out = model(x)

  # device buffers tied to this batch: input x, model output,
  # and the target (matches the output's shape/dtype, so another
  # out_bytes). element_size() = bytes per element (float32 -> 4,
  # float64 -> 8).
  in_bytes  = x.numel() * x.element_size()
  out_bytes = out.numel() * out.element_size()
  io = in_bytes + 2 * out_bytes

  # the chi2 runs outside model(x), so the hook never sees it.
  # Per batch it builds a few full-length float64 buffers (the
  # unsqueezed residual, the r @ Cinv product, the copy autograd
  # saves for backward) -- budget three (bs, dv_len) doubles.
  chi2 = 3 * bs * dv_len * 8

  return total + io + chi2


def compute_model_size_bytes(model):
  """
  Bytes the model keeps resident for the whole run.

  Counts weights, gradients, and the optimizer's per-parameter
  state, budgeted at the worst typical case. opt_state = state
  tensors the optimizer keeps per param (each param-sized):

      SGD (plain)                    0
      SGD+momentum, Adagrad,         1
        RMSprop (default)
      Adam, AdamW, Adamax, NAdam     2
      Adam(amsgrad), RMSprop         3   <- worst typical
        (centered + momentum)

  Arguments:
    model = the network whose parameters are counted.

  Returns:
    bytes = n_params * element_size * (2 + opt_state), i.e.
    weights(1) + grads(1) + opt_state buffers.
  """
  opt_state = 3
  # total parameter elements across all weight tensors.
  p = 0
  for t in model.parameters():
    p += t.numel()
  esize = next(model.parameters()).element_size()  # bytes
  # weights(1) + grads(1) + opt_state buffers
  return p * esize * (2 + opt_state)

def batches_per_load(model,
                     bs,
                     sample_shape,
                     budget,
                     dv_len=3000):
  """
  Batches per streamed chunk that fit the VRAM budget.

  resident = model (weights + grads + optimizer state) + the
  chi2 precision matrix Cinv; the chunk gets what is left of
  0.8 * budget, divided by one batch's cost.

  Arguments:
    model        = the network (sizes the resident + probe cost).
    bs           = minibatch size.
    sample_shape = shape of one model input, no batch axis.
    budget       = VRAM bytes to plan against; explicit (real
                   free VRAM in research, emulated GPU_MEM in
                   class).
    dv_len       = full dv length (sizes Cinv and the chi2
                   scratch).

  Returns:
    number of bs-row batches per streamed chunk (at least 1).
  """
  cinv     = dv_len * dv_len * 8
  resident = compute_model_size_bytes(model) + cinv
  free = 0.8 * budget - resident
  per  = compute_batch_size_bytes(model=model,
                                  bs=bs,
                                  sample_dims=sample_shape,
                                  dv_len=dv_len)
  return max(1, int(free // per))


def _build_loaders_one(device, C, dv, idx,
                       param_geometry, chi2fn,
                       model, bs, budget,
                       dv_len=3000, CHUNK=1000):
  """
  Build the two data loaders for one source -- a train or val
  file -- and decide where that source's data lives. Both take
  global row indices into the full C/dv dump; a `slots` helper
  maps those to local positions in the compact resident subset
  (see below), so the rest of the pipeline is identical wherever
  the data ended up (see Returns for the four outputs). The
  params are always encoded once and kept on the GPU (tiny,
  n_used x Ncosmo), so only the data vectors (the large array)
  change placement, by a memory ladder against `budget`:
    Regime 1 (resident gather): the encoded set fits, so
      pre-encode it once; a batch is pure on-device indexing.
    Regime 2 (RAM stream): does not fit but the dvs are an in-RAM
      ndarray; stream RAM->GPU a chunk at a time, encode on the
      fly (pinned memory on CUDA).
    Regime 3 (disk stream): the dvs exceed RAM (a np.memmap), so
      the same per-chunk path reads from disk.
  Resident memory = model + the chi2's Cinv + the encoded
  params; the dvs get what is left (0.8 * budget - resident), and
  `fits` decides regime 1 versus streaming. The orchestrator
  subtracts the returned `used` before sizing the next source,
  so two sequential builds share one GPU without overrunning it.
  Works for the plain CosmolikeChi2 (encode takes the dv alone)
  and the param-aware losses (RescaledChi2 / ResidualBase /
  PCEResidualChi2 / PCERatioChi2), whose encode also takes this
  block's whitened params (the resident C_used rows) to build R
  or the PCE base; the `rescaled` flag branches encode. A loss
  may also stage a wider target via a target_dim attribute (see
  tgt_dim below).

  Arguments:
    device     = target device for the staged tensors.
    C          = full param dump, (N, Ncosmo).
    dv         = full dv dump, (N, Ndv); ndarray -> regime 2,
                 np.memmap -> regime 3.
    idx        = global row indices into C/dv to make loadable.
    param_geometry = ParamGeometry; .encode whitens raw params.
    chi2fn     = CosmolikeChi2 or RescaledChi2 (output geom).
    model      = network; read only to size resident memory.
    bs         = minibatch size; the chunk is a multiple of it.
    budget     = VRAM bytes to plan against.
    dv_len     = full dv length the chi2 unsqueezes to.
    CHUNK      = rows per block when pre-encoding (regime 1).
  Returns:
    load_C  = callable: global rows -> whitened inputs.
    load_dv = callable: global rows -> whitened targets.
    load    = rows per chunk chosen for this regime.
    used    = GPU bytes this source made resident.
  """
  ncosmo    = C.shape[1]
  # out_dim = model output width = the unmasked dv entries the
  # network predicts (dest_idx holds the kept positions;
  # .numel() counts them).
  out_dim   = chi2fn.dest_idx.numel()

  # tgt_dim = width of the target tensor this loader stages per
  # row. Normally just the encoded truth, one value per kept
  # entry, so tgt_dim == out_dim. One loss needs more room:
  # PCERatioChi2 forms pred = base * (1 + net_output), where
  # net_output is the model's fractional correction and base a
  # fixed reference dv (the frozen PCE). Rather than recompute
  # base every batch, it precomputes it once here and stages
  # [base ; truth] as one 2*n_keep-wide target, unpacked inside
  # the chi2.
  #
  # A loss requests that wider target via a `target_dim`
  # attribute. getattr(obj, "name", default) returns obj.name if
  # present, else `default` (never raises), so a loss without
  # target_dim falls back to out_dim and stages the plain truth
  # -- the same opt-in pattern as needs_params below.
  tgt_dim   = getattr(chi2fn, "target_dim", out_dim)

  # used_rows = the distinct rows this source loads. np.unique
  # drops duplicates (idx may name a row twice) and returns them
  # sorted, so each row is staged once and in file order: what
  # slots() assumes, and what makes a memmap read sequential.
  # n_used counts them.
  used_rows = np.unique(idx)
  n_used    = len(used_rows)

  assert dv.shape[1] == chi2fn.total_size, (
    f"dv width {dv.shape[1]} != "
    f"total_size {chi2fn.total_size}")

  # encode the params once, resident on the GPU. For the
  # rescaled geometry these whitened params also let encode build
  # R.
  C_used = param_geometry.encode(
    torch.from_numpy(C[used_rows]).float().to(device))

  rescaled = getattr(chi2fn, "needs_params", False)

  model_bytes = compute_model_size_bytes(model)
  cinv        = dv_len * dv_len * 8
  enc_params  = n_used * ncosmo * 4
  resident    = (model_bytes + cinv + enc_params)

  def slots(rows):
    # Translate global row numbers into local positions in the
    # compact resident subset. This run trains on only the
    # N_train subset of the on-disk dump; the loaders staged
    # those used rows into C_used / dv_used in sorted order, so a
    # row's global index (position in the full dump) differs from
    # its local index (position in the resident subset). used_rows
    # is the sorted kept global indices, so a row's local index is
    # where it sits inside used_rows. np.searchsorted gives each
    # query's insertion index into the sorted array; since every
    # query row is itself in used_rows, that index is its row
    # index in C_used/dv_used.
    local_pos = np.searchsorted(used_rows, rows)
    return torch.from_numpy(local_pos).to(device)

  def load_C(rows):
    return C_used[slots(rows)]

  enc_dvs = n_used * tgt_dim * 4
  fits    = enc_dvs + resident < 0.8 * budget

  if fits:
    # Regime 1: pre-encode every target, hold it on the GPU.
    dv_used = torch.empty(n_used, tgt_dim, device=device)
    for start in range(0, n_used, CHUNK):
      block = used_rows[start:start + CHUNK]
      # raw dvs for this block, on the device.
      dv_t = torch.from_numpy(dv[block]).float().to(device)
      if rescaled:
        # rescaled target: encode also needs this block's params.
        # C_used is in used_rows order, so the block is the local
        # slice start : start + len(block).
        params = C_used[start:start + len(block)]
        enc = chi2fn.encode(dv=dv_t, params_whitened=params)
      else:
        enc = chi2fn.encode(dv_t)
      dv_used[start:start + len(block)] = enc

    def load_dv(rows):
      return dv_used[slots(rows)]

    bytes_per_row = (tgt_dim + ncosmo) * 4
    vram_left     = 0.8 * budget - resident - enc_dvs
    fit_rows = max(bs, int(vram_left // bytes_per_row))
    load = min(len(idx), fit_rows)

  elif not isinstance(dv, np.memmap):
    # Regime 2: dvs live in CPU RAM.
    def load_dv(rows):
      cpu = torch.from_numpy(dv[rows]).float()
      if device.type == "cuda":
        cpu = cpu.pin_memory()
      gpu = cpu.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len)
  else:
    # Regime 3: dvs exceed RAM, read from the memmap.
    def load_dv(rows):
      host = torch.from_numpy(dv[rows]).float()
      gpu  = host.to(device)
      if rescaled:
        return chi2fn.encode(dv=gpu, params_whitened=load_C(rows))
      return chi2fn.encode(gpu)

    load = bs * batches_per_load(model=model,
                                 bs=bs,
                                 sample_shape=C.shape[1:],
                                 budget=budget,
                                 dv_len=dv_len)

  used = enc_params + (enc_dvs if fits else 0)
  return load_C, load_dv, load, used


def build_loaders(device, train_set, val_set, param_geometry, 
                  chi2fn, model, bs, budget,
                  dv_len=3000, CHUNK=1000):
  """
  Build the train and val loaders, return the data dict the
  training loop and eval_val consume. Train and val live in
  separate files (T and T/2); each is passed as a source dict
  and gets its own loaders via _build_loaders_one (no shared
  rows, no leakage). The same training-built param_geometry /
  chi2fn whiten both sources.

  Arguments:
    device     = target device.
    train_set  = training source dict:
                   "C"   full param dump (T file),
                   "dv"  full dv dump (T file),
                   "idx" global rows to train on (into C/dv).
    val_set    = validation source dict, same three keys.
    param_geometry, chi2fn, model, bs, budget, dv_len, CHUNK
               = forwarded to _build_loaders_one (see there);
                 the same geometry for both sources.
  Returns:
    data = nested dict, one sub-dict per source, both with
      the same keys:
        data["train"] = {load_C, load_dv, idx, load}
        data["val"]   = {load_C, load_dv, idx, load}
  """
  (load_C, load_dv, load,
   used_tr) = _build_loaders_one(device=device, 
                              C=train_set["C"], 
                              dv=train_set["dv"], 
                              idx=train_set["idx"],
                              param_geometry=param_geometry, 
                              chi2fn=chi2fn, 
                              model=model, 
                              bs=bs, 
                              budget=budget, 
                              dv_len=dv_len, 
                              CHUNK=CHUNK)

  # train is now resident on the GPU, so the val call plans
  # against a budget reduced by what train took. (model + Cinv
  # are shared, counted by each call.)
  (load_C_val, load_dv_val, load_val, 
   _) = _build_loaders_one(device=device, 
                           C=val_set["C"], 
                           dv=val_set["dv"], 
                           idx=val_set["idx"],
                           param_geometry=param_geometry, 
                           chi2fn=chi2fn,
                           model=model, 
                           bs=bs, 
                           budget=budget - used_tr,
                           dv_len=dv_len, 
                           CHUNK=CHUNK)

  return {
    "train": {
      "load_C": load_C,
      "load_dv": load_dv,
      "idx": train_set["idx"],
      "load": load,
    },
    "val": {
      "load_C": load_C_val,
      "load_dv": load_dv_val,
      "idx": val_set["idx"],
      "load": load_val,
    }
  }
