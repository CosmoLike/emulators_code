"""Device selection, construction factories, evaluation, and training.

The run layer tying the package together. pick_device and make_logger are
setup helpers. make_model, make_optimizer, and make_scheduler each build one
component from a {cls, **kwargs} spec dict; build_run_specs assembles the six
spec dicts from a config (with the default / suggest / search resolvers for the
[default, min, max, kind] hyperparameter ranges). eval_val and eval_source_chi2
score the model, training_loop_batched is the per-epoch loop (trim / focus
annealing, best-epoch tracking), and run_emulator orchestrates: builds
everything, trains, and returns the model plus the per-epoch histories.

PS: a loader is a closure load(rows) -> tensor mapping global row indices to a
ready-to-train batch already on the compute device, hiding where the data lives
(resident on the GPU, streamed from RAM, or a disk memmap). build_loaders
(batching.py) makes one loader for the whitened parameters (load_C) and one for
the encoded targets (load_dv) per source; the loop here just asks for the rows
it wants. whitened = rotated into a covariance eigenbasis and scaled to unit
variance, decorrelating the components (the form the model sees, input and
target).
"""

import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from .batching import build_loaders
from .emulator_designs import ResMLP
from .loss_functions import anneal_value


def pick_device(name=None):
  """
  Choose the compute device: an explicit name, else auto-detect.

  Arguments:
    name = force a device by string ("cuda" / "mps" / "cpu"); None
           (default) auto-detects CUDA, else Apple MPS, else CPU.

  Returns:
    a torch.device.
  """
  if name is not None:
    return torch.device(name)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def make_logger(quiet=False):
  """
  Build a print function gated by a quiet flag.

  Returns a `log(*args, **kwargs)` callable that forwards to the
  builtin print when `quiet` is False and is a no-op when True --
  the standard "--quiet" stdout gate a CLI driver wraps its prints
  in. `quiet` is captured once, at build time.

  Arguments:
    quiet = if True, the returned logger swallows every call (prints
            nothing); if False (default), it forwards to print.

  Returns:
    log = a function with print's signature (*args, **kwargs) that
          prints unless quiet.
  """
  def log(*args, **kwargs):
    if not quiet:
      print(*args, **kwargs)
  return log


def make_model(model_opts, input_dim, output_dim, device):
  """
  Build the network from a spec dict.

  Mirrors make_optimizer: the model class is a value in the spec
  dict, its settings the other keys -- swapping architectures is
  a one-dict change.

  Arguments:
    model_opts = model spec dict. "cls" is the model class (e.g.
                 ResMLP), stored as a value (same factory trick as
                 the optimizer). "compile_mode" (optional) sets the
                 CUDA torch.compile mode (see below). Every other
                 key forwards to the constructor (int_dim_res,
                 n_blocks, block_opts, ...).
    input_dim  = number of input features (the cosmological
                 parameter count); injected, not in the dict.
    output_dim = number of outputs (unmasked dv length); injected.
    device     = device to build on. On CUDA the model is
                 torch.compile'd per compile_mode; eager on MPS/CPU.

  compile_mode (CUDA only; default "reduce-overhead"):
    "reduce-overhead" = inductor + CUDA graphs; fastest but fragile
      -- large constant buffers or a skip-add of the trunk output
      can trip CUDA-graph-trees bookkeeping (internal
      AssertionError during warmup).
    "default" = inductor kernel fusion, no CUDA graphs (robust).
    None      = plain eager (no compile).
  Returns:
    the model on `device`, compiled per compile_mode on CUDA.
  """
  cls = model_opts["cls"]
  compile_mode = model_opts.get(
    "compile_mode", "reduce-overhead")
  # forward every key except cls / compile_mode to the constructor.
  extra = {}
  for k, v in model_opts.items():
    if k not in ("cls", "compile_mode"):
      extra[k] = v
  model = cls(input_dim=input_dim,
              output_dim=output_dim, **extra).to(device)
  if device.type == "cuda" and compile_mode is not None:
    model = torch.compile(model, mode=compile_mode)
  return model


def make_optimizer(model, opt_opts, lr, device):
  """
  Build the optimizer from a spec dict.

  Mirrors make_model / make_scheduler: the optimizer class is a
  value in the dict, its settings the other keys. Parameters split
  into two groups so weight decay falls only on the weight matrices
  (ndim>=2), never on the 1D params below -- decaying those would
  pull a unit-init gain toward 0 and attenuate signal.

  Arguments:
    model    = network whose parameters are optimized;
               named_parameters() splits into weight matrices
               (ndim>=2, decayed) and 1D params (biases, Affine
               gain/bias, activation gamma/beta) not decayed.
    opt_opts = optimizer spec dict. "cls" is the optimizer
               class (e.g. optim.AdamW), stored as a value;
               "weight_decay" (optional, default 0.0) decays
               the weight-matrix group only; other keys forward
               to the constructor (betas, eps, ...).
    lr       = learning rate; injected (the sqrt-batch-scaled
               value), not in the dict.
    device   = device the model lives on; on CUDA the fused
               optimizer kernel is enabled.

  Returns:
    the optimizer, with two param groups: weight matrices decayed
    by opt_opts["weight_decay"], the rest at 0. fused on CUDA.
  """
  # decay weight matrices (ndim>=2); leave the 1D params undecayed.
  decay, no_decay = [], []
  for _, p in model.named_parameters():
    if p.ndim >= 2:
      decay.append(p)
    else:
      no_decay.append(p)
  wd    = opt_opts.get("weight_decay", 0.0)
  cls   = opt_opts["cls"]
  # forward every key except cls / weight_decay to the constructor.
  extra = {}
  for k, v in opt_opts.items():
    if k not in ("cls", "weight_decay"):
      extra[k] = v
  groups = [
    {"params": decay,    "weight_decay": wd},
    {"params": no_decay, "weight_decay": 0.0},
  ]
  # fused is a CUDA-only Adam/SGD-family speedup.
  if device.type == "cuda":
    extra["fused"] = True
  return cls(groups, lr=lr, **extra)


def make_scheduler(optimizer, sched_opts):
  """
  Build the LR scheduler from a spec dict.

  Mirrors make_model / make_optimizer: the scheduler class is a
  value in the dict, its settings the other keys, so swapping
  schedulers is a one-dict change.

  Arguments:
    optimizer  = the optimizer whose learning rate this
                 schedules; injected, not in the dict.
    sched_opts = scheduler spec dict. "cls" is the scheduler
                 class (e.g. lr_scheduler.ReduceLROnPlateau),
                 stored as a value; every other key forwards to
                 its constructor (mode, patience, factor, ...).

  Returns:
    the constructed scheduler.
  """
  cls   = sched_opts["cls"]
  # forward every key except cls to the constructor.
  extra = {}
  for k, v in sched_opts.items():
    if k != "cls":
      extra[k] = v
  return cls(optimizer, **extra)


def build_run_specs(train_args, model_cls, opt_cls, sched_cls):
  """
  Assemble the six run_emulator spec dicts from a config mapping.

  Each constructible component is a {"cls": <class>, **kwargs}
  spec -- the first-class-class trick make_model / make_optimizer
  / make_scheduler consume. The caller picks the class (a driver
  fixes ResMLP / AdamW / ReduceLROnPlateau, or swaps any), its
  settings come from the matching sub-block of train_args, spread
  in with **; the classless schedules (lr, trim, focus) are copied
  verbatim. Keyed by the exact run_emulator argument names, so a
  caller can splat it: run_emulator(..., **build_run_specs(...)).

  Spreading **train_args["model"] (etc.) means this never has to
  know a class's kwargs: whatever serializable settings the YAML
  lists flow through (int_dim_res / n_blocks for ResMLP,
  kernel_size / channels for ResCNN, ...). Each {...} / dict(...)
  builds a new dict, never mutating the input mapping.

  Arguments:
    train_args = mapping (e.g. a YAML "train_args" block) holding
                 the sub-mappings "model", "optimizer", "lr",
                 "scheduler", "trim", "focus". Each carries only
                 serializable settings; injected/runtime args
                 (device, in/out dims, lr value, a geometry) are
                 added later by make_X or the driver.
    model_cls  = model class for model_opts["cls"] (ResMLP, ...).
    opt_cls    = optimizer class for opt_opts["cls"] (AdamW, ...).
    sched_cls  = scheduler class for sched_opts["cls"]
                 (ReduceLROnPlateau, ...).

  Returns:
    dict with keys model_opts, opt_opts, lr_opts, sched_opts,
    trim_opts, focus_opts -- the six spec dicts run_emulator takes.
  """
  return {
    "model_opts": {"cls": model_cls, **train_args["model"]},
    "opt_opts":   {"cls": opt_cls,   **train_args["optimizer"]},
    "lr_opts":    dict(train_args["lr"]),
    "sched_opts": {"cls": sched_cls, **train_args["scheduler"]},
    "trim_opts":  dict(train_args["trim"]),
    "focus_opts": dict(train_args["focus"]),
  }


# --- hyperparameter search ranges inside train_args ---
# A train_args leaf is either a fixed scalar or a search range
# [default, min, max, kind] with kind one of "int" / "float" /
# "log" (a whitespace string "default min max kind" also works).
# The first value is the default: the plain driver uses it, a
# search warm-starts trial 0 from it. The three resolvers below
# share one walk over the (nested) mapping.
_SEARCH_KINDS = ("int", "float", "log")


def _as_search_range(value):
  """Return [default, min, max, kind] if value marks a search range.

  Accepts a YAML 4-list or a whitespace string "d min max kind";
  returns None for a fixed scalar (or any non-range value), leaving
  it untouched in the walk.
  """
  if isinstance(value, str):
    parts = value.split()
    value = parts if len(parts) == 4 else value
  if (isinstance(value, (list, tuple)) and len(value) == 4
      and str(value[3]) in _SEARCH_KINDS):
    return [value[0], value[1], value[2], str(value[3])]
  return None


def _range_default(rng):
  """A range's default (first) value, typed by its kind."""
  d, kind = rng[0], rng[3]
  return int(d) if kind == "int" else float(d)


def _suggest_range(trial, name, rng):
  """One Optuna suggestion for a search range.

  kind selects the suggestion: "int" -> suggest_int, "float" ->
  suggest_float (linear), "log" -> suggest_float(log=True). min/max
  are cast, so a YAML 1e-5 that parsed as a string still works.
  """
  _, lo, hi, kind = rng
  if kind == "int":
    return trial.suggest_int(name, int(lo), int(hi))
  if kind == "log":
    return trial.suggest_float(name, float(lo), float(hi), log=True)
  return trial.suggest_float(name, float(lo), float(hi))


def _walk_train_args(train_args, path, on_leaf):
  """Recurse train_args, applying on_leaf(path, value) to each leaf.

  Returns a new mapping with the same nesting (the comprehensions
  copy, so the input is never mutated); on_leaf decides each leaf.
  """
  if isinstance(train_args, dict):
    out = {}
    for k, v in train_args.items():
      child_path = f"{path}.{k}" if path else k
      out[k] = _walk_train_args(v, child_path, on_leaf)
    return out
  return on_leaf(path, train_args)


def default_train_args(train_args):
  """
  Resolve every search range to its default -- a fixed config.

  Walks train_args (any nesting) and replaces each
  [default, min, max, kind] range with its default value, leaving
  scalars untouched. This lets the plain training driver consume a
  YAML that also carries search ranges -- it uses each range's
  first value, so one YAML serves both the plain and search drivers.

  Arguments:
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    the same mapping with every range collapsed to its default.
  """
  def leaf(path, v):
    rng = _as_search_range(v)
    return _range_default(rng) if rng else v
  return _walk_train_args(train_args, "", leaf)


def suggest_train_args(trial, train_args):
  """
  Resolve train_args for one Optuna trial.

  Walks train_args; each [default, min, max, kind] range becomes an
  Optuna suggestion named by its dotted path (e.g. "lr.lr_base"),
  each scalar kept. Returns a fully-resolved train_args (same nested
  shape) ready for build_run_specs / run_emulator. Never imports
  optuna -- it only calls the passed trial's suggest_* .

  Arguments:
    trial      = an optuna Trial (the source of the suggestions).
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    train_args with every range replaced by this trial's sample.
  """
  def leaf(path, v):
    rng = _as_search_range(v)
    return _suggest_range(trial, path, rng) if rng else v
  return _walk_train_args(train_args, "", leaf)


def search_defaults(train_args):
  """
  The {dotted-path: default} of every search range in train_args.

  The warm-start point for a study (enqueue it as trial 0), keyed
  to match the names suggest_train_args registers. Empty when no
  range is present.

  Arguments:
    train_args = a YAML "train_args" mapping (may hold ranges).

  Returns:
    a dict {path: default} over the ranges, typed by their kind.
  """
  out = {}

  def leaf(path, v):
    rng = _as_search_range(v)
    if rng:
      out[path] = _range_default(rng)
    return v
  _walk_train_args(train_args, "", leaf)
  return out


def eval_val(model, lossfn, data, load, bs, thresholds):
  """
  Evaluate the model on the validation set.

  Streams the val rows in chunks of `load`, runs the model in
  fixed bs-sized batches, gathers the per-sample chi2, then
  summarizes. The chi2 is gathered first because the mean composes
  across chunks but the median and threshold fractions do not --
  they need the full distribution at once.

  The model runs in batches of exactly `bs` (the final partial
  batch padded up to bs, pad rows dropped after), so a
  torch.compile'd model sees one static input shape and never
  recompiles. Padding, not dropping, because evaluation must score
  every val point.

  Arguments:
    model      = the network, in eval mode.
    lossfn     = CosmolikeChi2; .chi2 gives the per-sample chi2 of
                 a prediction against its target.
    data       = dict with load_C, load_dv (the loaders) and vidx
                 (global validation row indices).
    load       = rows per streamed chunk.
    bs         = model batch size (same as training), so the
                 compiled model sees one fixed input shape.
    thresholds = 1D tensor of delta-chi2 cutoffs; the returned
                 fraction counts val points above each.

  Returns:
    median = median per-sample chi2 over the val set.
    mean   = mean per-sample chi2 over the val set.
    frac   = 1D tensor (len = #thresholds): fraction of val
             points with chi2 above each threshold.
  """
  load_C = data["load_C"]
  load_dv = data["load_dv"]
  vidx = data["idx"]
  chi2s = []

  with torch.no_grad():
    for cs in range(0, len(vidx), load):
      rows = np.sort(vidx[cs:cs+load])
      Cc  = load_C(rows)             # (m, Ncosmo)
      dvc = load_dv(rows)            # (m, out_dim)
      m   = Cc.shape[0]
      preds = []
      for s in range(0, m, bs):
        xb = Cc[s:s+bs]
        n  = xb.shape[0]             # real rows this batch
        if n < bs:
          # pad the final short batch up to bs, keeping the compiled
          # model on one fixed shape. Cc[:1] is the first row;
          # .expand(bs-n, -1) stretches its size-1 row axis to bs-n
          # copies (-1 = keep the column axis) as a stride-0 view,
          # no copy. Pad rows sliced off ([:n]) after the run.
          pad = Cc[:1].expand(bs - n, -1)
          xb  = torch.cat([xb, pad], dim=0)
        # clone: under reduce-overhead (CUDA graphs) the model
        # reuses a static output buffer per call, so the next
        # model(xb) overwrites this output -- and we stash several
        # in `preds` before cat, so copy each out now.
        preds.append(model(xb)[:n].clone())

      pred = torch.cat(preds, dim=0)     # (m, out_dim)
      if getattr(lossfn, "needs_params", False):
        chi2s.append(lossfn.chi2(pred=pred,
                                 target=dvc,
                                 params_whitened=Cc))
      else:
        chi2s.append(lossfn.chi2(pred=pred,
                                 target=dvc))

  c = torch.cat(chi2s).cpu() # per-sample chi2
  mean   = c.mean().item()
  median = c.median().item()

  # c[:, None] (Nval, 1) and thresholds[None, :] (1, T) broadcast
  # into a (Nval, T) boolean grid: entry [i, j] = "is point i's
  # chi2 above threshold j?". mean(0) over samples -> the fraction
  # past each threshold. ([:, None] is the numpy/torch unsqueeze.)
  frac = (c[:, None] > thresholds[None, :]).float().mean(0)
  return median, mean, frac


def eval_source_chi2(model,
                     param_geometry,
                     chi2fn,
                     source,
                     device,
                     bs):
  """
  Per-cosmology delta-chi2 of the emulator over one source.

  Scores every row of `source` in source["idx"]: encodes its
  parameters into model inputs, predicts the whitened data vector,
  and evaluates the full masked chi2 against the encoded truth.
  Returns plain numpy arrays aligned row-for-row, ready for a
  parameter-space plot.

  Works for plain CosmolikeChi2 and RescaledChi2: the rescaled
  geometry needs the params (to build R), which are the whitened
  inputs X, passed to encode and chi2 when chi2fn rescales.

  Arguments:
    model          = trained network; set to eval mode here.
    param_geometry = ParamGeometry; .encode whitens the raw
                     parameters into the model inputs.
    chi2fn         = CosmolikeChi2 or RescaledChi2.
    source         = source dict with "C", "dv", "idx".
    device         = device the model lives on.
    bs             = rows per forward batch (bounds memory).

  Returns:
    params = (N, n_param) float64 raw parameters of the rows.
    dchi2  = (N,) float64 per-row delta-chi2, same row order.
  """
  model.eval()
  rows   = np.sort(source["idx"])
  params = np.asarray(source["C"][rows], dtype="float64")

  with torch.no_grad():
    # whitened model inputs for these rows
    X = param_geometry.encode(
      torch.from_numpy(params).float().to(device))
    dv = torch.from_numpy(
      source["dv"][rows]).float().to(device)
    # rescaled geometry needs the params to build R; the plain
    # one does not.
    if getattr(chi2fn, "needs_params", False):
      T = chi2fn.encode(dv=dv, params_whitened=X)
    else:
      T = chi2fn.encode(dv)

    chunks = []
    for s in range(0, X.shape[0], bs):
      pred = model(X[s:s + bs])
      if getattr(chi2fn, "needs_params", False):
        c = chi2fn.chi2(pred=pred, target=T[s:s + bs], params_whitened=X[s:s + bs])
      else:
        c = chi2fn.chi2(pred=pred, target=T[s:s + bs])
      chunks.append(c.cpu())

  dchi2 = torch.cat(chunks).double().numpy()
  return params, dchi2


def training_loop_batched(nepochs,
                          optimizer,
                          scheduler,
                          model,
                          bs,
                          lossfn,
                          mode,
                          data,
                          trim_opts,
                          focus_opts,
                          thresholds,
                          warmup_epochs=0,
                          silent=False,
                          use_amp=False):
  """
  Train the emulator, with a validation pass per epoch.

  Each epoch reshuffles the training rows, streams them in
  chunks through the loaders, and steps the optimizer one
  minibatch at a time. After each epoch it evaluates on the val
  set and steps the scheduler on the val median (after an optional
  linear lr warmup over the first warmup_epochs epochs). Data
  placement (resident or streamed) is hidden behind the loaders,
  so this loop is identical in every regime.

  Arguments:
    nepochs    = number of passes over the training set.
    optimizer  = the optimizer (e.g. Adam).
    scheduler  = LR scheduler stepped on the val median each
                 epoch (e.g. ReduceLROnPlateau).
    model      = the network (possibly torch.compile'd).
    bs         = minibatch size.
    lossfn     = CosmolikeChi2; .loss(pred, target, mode) is
                 the training loss, .chi2 the eval metric.
    mode       = loss mode passed to .loss ("sqrt", "chi2", ...).
    data       = dict with load_C, load_dv (loaders), tidx
                 (training rows), and load (rows per chunk).
    trim_opts  = trim schedule (see anneal_value): "start"/"end"
                 trim fractions, "hold_epochs"/"anneal_epochs",
                 "shape". None -> hold 5% then cosine-anneal to 0.
    focus_opts = focal-weight schedule (see anneal_value): the
                 per-epoch focus exponent gamma (0 = uniform,
                 higher = harder points weighted more), via
                 "start"/"end", "hold_epochs"/"anneal_epochs",
                 "shape". None -> no focal weighting (gamma = 0).
    thresholds = delta-chi2 cutoffs for the val fractions.
    warmup_epochs = epochs of linear lr ramp before the plateau
                    scheduler takes over (0 = none).
    silent     = if True, suppress all per-epoch and summary
                 prints; metrics and returns are unchanged.
    use_amp    = if True, run the forward in bfloat16 autocast;
                 the loss stays in float32/64.

  Returns:
    train_losses, medians, means, fracs = per-epoch lists
      (fracs holds one fraction tensor per epoch).
  """
  # loaders: global row indices -> ready-to-train param inputs
  # / dv targets on the GPU (the regime hides where they live).
  load_C  = data["train"]["load_C"]
  load_dv = data["train"]["load_dv"]
  tidx    = data["train"]["idx"]   # global training rows (into C0/dv0)
  # device the model lives on; place new tensors here too.
  # model.parameters() is an iterator, so next(...), not [0].
  device  = next(model.parameters()).device
  ntrain  = len(tidx)              # training rows per epoch
  load    = data["train"]["load"]  # rows per streamed chunk
  # chunks per epoch = ceil(ntrain / load); + load - 1 rounds the
  # integer division up.
  nchunks = (ntrain + load - 1) // load

  if not silent:
    print(f"{load} rows/chunk, {nchunks} chunks/epoch, "
          f"amp={use_amp}, loss mode = {mode}")

  train_losses, medians, means, fracs = [], [], [], []

  # MPS (Apple Silicon) has no float64. Accumulate the loss in
  # float64 where supported (CUDA/CPU), float32 on MPS -- only
  # the epoch-mean train loss, so the fallback is harmless.
  acc_dtype = (torch.float32 if device.type == "mps"
                             else torch.float64)

  amp_dtype = (torch.float16 if device.type == "mps"
               else torch.bfloat16)

  # target lr per param group, captured before warmup ramps it;
  # warmup scales each group up to its own base.
  base_lrs = []
  for g in optimizer.param_groups:
    base_lrs.append(g["lr"])

  # track the best epoch by the inference metric -- the fraction
  # of val points with chi2 > the first threshold (0.2) -- to keep
  # the best model, not the last.
  best_frac  = float("inf")
  best_state = None
  best_epoch = 0
  best_median = float("inf")

  # kappa = chi2 scale where the focal weight turns on (fixed over
  # the run, unlike the annealed gamma); from focus_opts, default
  # 1.0. Feeds the loss as focus_scale.
  kappa = focus_opts.get("kappa", 1.0)

  for epoch in range(1, nepochs + 1):
    model.train()
    perm = tidx[torch.randperm(ntrain).numpy()]

    # this epoch's annealed trim fraction (large early, 0 late).
    # One value per epoch, shared by all its batches.
    rob = anneal_value(epoch=epoch, opts=trim_opts)

    # this epoch's focal weight exponent gamma (annealed): 0 early
    # -> uniform weighting (a plain mean, stable while the bulk is
    # still being learned), rising to focus_opts["end"] late ->
    # up-weight the hard points so the optimizer keeps chasing the
    # tail, not out-voted by the solved bulk. One per epoch.
    focus = anneal_value(epoch=epoch, opts=focus_opts)

    # epoch training loss, accumulated on-device
    run_sum = torch.zeros((),
                          device=device,
                          dtype=acc_dtype)

    run_n   = 0
    for cs in range(0, ntrain, load):
      rows = np.sort(perm[cs:cs+load])
      Cc  = load_C(rows)
      dvc = load_dv(rows)
      bp = torch.randperm(Cc.shape[0], device=device)
      # Drop the ragged last batch so every batch is one size.
      # This matters under torch.compile: it specializes per input
      # shape, and reduce-overhead (CUDA graphs) needs it fixed. bp
      # reshuffles each epoch, so dropped tail rows rotate -- no
      # data is permanently lost.
      n_full = (Cc.shape[0] // bs) * bs   # whole batches only
      for s in range(0, n_full, bs):
        b = bp[s:s+bs]
        with torch.autocast(device.type,
                            dtype=amp_dtype,
                            enabled=use_amp):
          pred = model(Cc[b])

        if getattr(lossfn, "needs_params", False):
          loss = lossfn.loss(pred=pred,
                             target=dvc[b],
                             params_whitened=Cc[b],
                             mode=mode,
                             trim=rob,
                             focus=focus,
                             focus_scale=kappa)
        else:
            loss = lossfn.loss(pred,
                               target=dvc[b],
                               mode=mode,
                               trim=rob,
                               focus=focus,
                               focus_scale=kappa)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        run_sum += loss.detach() * b.numel()
        run_n   += b.numel()
    train_loss = (run_sum / run_n).item()
    model.eval()
    median, mean, frac = eval_val(model=model,
                                  lossfn=lossfn,
                                  data=data["val"],
                                  load=load,
                                  bs=bs,
                                  thresholds=thresholds)
    train_losses.append(train_loss)
    medians.append(median)
    means.append(mean)
    fracs.append(frac)

    # f0 = this epoch's fraction of val points with chi2 >
    # thresholds[0] (0.2) -- the inference goal we minimize.
    # frac[0] is a 0-dim tensor; .item() pulls it to a Python
    # float (one host sync, once per epoch).
    f0 = frac[0].item()
    # Make this epoch the new best when it strictly lowers frac>0.2,
    # or ties the best fraction at a lower median. frac>0.2 is
    # coarse (k/Nval, a step function), so many epochs share a
    # value; the median tiebreaker then keeps the tightest bulk
    # among the equally-good fractions.
    if (f0 < best_frac or
        (f0 == best_frac and median < best_median)):
      best_frac   = f0
      best_median = median
      best_epoch  = epoch
      # snapshot the weights. state_dict() returns references to
      # the live parameters, which keep changing as training
      # continues, so clone to freeze them here. detach drops grad
      # tracking (a stored snapshot needs no autograd). One copy,
      # replaced whenever the best improves.
      best_state = {}
      for k, v in model.state_dict().items():
        best_state[k] = v.detach().clone()

    if epoch <= warmup_epochs:
      # linear warmup: ramp each group's lr from base/W up to base
      # over the first W = warmup_epochs epochs, then hand off to
      # the plateau scheduler -- not stepped during warmup, since
      # its no-improvement counter must not run while lr rises.
      scale = epoch / warmup_epochs
      for grp, base in zip(optimizer.param_groups, base_lrs):
        grp["lr"] = base * scale
    else:
      # steps once per epoch -- right for ReduceLROnPlateau and
      # epoch schedulers (StepLR, CosineAnnealingLR). A per-batch
      # scheduler (OneCycleLR) would step inside the batch loop
      # instead, not here.
      if isinstance(scheduler,
                    lr_scheduler.ReduceLROnPlateau):
        scheduler.step(median)
      else:
        scheduler.step()

    if not silent:
      lr_now = optimizer.param_groups[0]["lr"]
      pairs = []
      for t, f in zip(thresholds.tolist(), frac.tolist()):
        pairs.append(f"{t:g}:{f:.3f}")
      fr = ", ".join(pairs)
      print(f"epoch {epoch:3d}  lr {lr_now:.2e}"
            f"  train {train_loss:.4f}"
            f"  val {mean:.4f}  med {median:.4f}"
            f"  frac>[{fr}]")

  if best_state is not None:
    model.load_state_dict(best_state)
    if not silent:
      print(f"best epoch {best_epoch}: "
            f"frac>0.2 {best_frac:.4f}")
  return train_losses, medians, means, fracs


def run_emulator(train_set, val_set, chi2fn, param_geometry,
                 bs=128, nepochs=300, loss_mode="sqrt",
                 model_opts=None, opt_opts=None, lr_opts=None,
                 sched_opts=None, trim_opts=None, focus_opts=None,
                 thresholds=None, gpu_mem_gb=16, use_amp=False,
                 silent=False, device='gpu', seed=0):
  """
  One training run; model, optimizer, schedule auto-built.

  Builds the model, optimizer, scheduler, and the regime
  loaders, then trains. Three spec dicts (model_opts, opt_opts,
  lr_opts) group the related knobs, as block_opts groups a
  ResBlock's.

  Arguments:
    train_set    = training source dict: "C" full param dump,
                   "dv" full dv dump, "idx" rows to train on.
    val_set      = validation source dict, same three keys.
    chi2fn         = CosmolikeChi2 (output geometry + loss).
    param_geometry = ParamGeometry (input whitening).
    bs           = minibatch size.
    nepochs      = number of passes over the training set.
    loss_mode    = loss transform ("sqrt", "chi2", "sqrt_dchi2").
    model_opts   = model spec dict (see make_model): "cls" the
                   model class + constructor settings. None ->
                   ResMLP, int_dim_res 128, n_blocks 4, block_opts
                   {}.
    opt_opts     = optimizer spec dict (see make_optimizer):
                   "cls" + "weight_decay" + extra kwargs.
                   None -> AdamW, weight_decay 1e-4.
    lr_opts      = learning-rate dict:
                     "lr_base"/"bs_base" -> sqrt-batch rule
                       (lr = lr_base * sqrt(bs / bs_base))
                     "warmup_epochs"     -> linear lr warmup
                   None -> a sensible default.
    sched_opts   = scheduler spec dict (see make_scheduler):
                   "cls" + its kwargs (mode, patience, factor,
                   ...). None -> ReduceLROnPlateau, mode "min",
                   patience 15, factor 0.75.
    trim_opts    = trim schedule (see anneal_value): "start"/"end"
                   trim fractions, "hold_epochs"/"anneal_epochs",
                   "shape". None -> hold 5% then cosine-anneal to 0.
    focus_opts   = focal-weight schedule (see anneal_value): the
                   per-epoch focus exponent gamma (0 = uniform,
                   higher = harder points weighted more), via
                   "start"/"end", "hold_epochs"/"anneal_epochs",
                   "shape". None -> no focal weighting (gamma = 0).
    thresholds   = delta-chi2 cutoffs for the val fractions
                   (None -> [0.2, 1, 10, 100]).
    gpu_mem_gb   = emulated budget in GB (non-CUDA only; on CUDA
                   the real free VRAM is used).
    use_amp      = run the forward in low-precision autocast.
    silent       = suppress all printing if True.
    seed         = manual seed for init + per-epoch shuffles.

  Returns:
    model        = trained network, restored to the best frac>0.2
                   epoch.
    train_losses = per-epoch training loss (list).
    medians      = per-epoch val median chi2 (list).
    means        = per-epoch val mean chi2 (list).
    fracs        = per-epoch list of frac-over-threshold tensors.
  """
  if model_opts is None:
    model_opts = {"cls": ResMLP,
                  "int_dim_res": 128,
                  "n_blocks": 4,
                  "block_opts": {}}
  if opt_opts is None:
    opt_opts = {"cls": optim.AdamW,
                "weight_decay": 1e-4}
  if lr_opts is None:
    lr_opts = {"lr_base": 5e-3,
               "bs_base": 64.0,
               "warmup_epochs": 5}
  if sched_opts is None:
    sched_opts = {"cls": lr_scheduler.ReduceLROnPlateau,
                  "mode": "min",
                  "patience": 15,
                  "factor": 0.75}
  if thresholds is None:
    # delta-chi2 cutoffs for the reported val fractions (fraction
    # of val points with chi2 above each). The first, 0.2, is the
    # emulator goal and the best-model selection metric (frac >
    # thresholds[0]); the rest are diagnostic bands up the cascade.
    thresholds = torch.tensor([0.2, 1.0, 10.0, 100.0])
  if trim_opts is None:
    # hold a 5% trim, then cosine-anneal it to 0 over the run:
    # drop the worst points while they are junk, re-admit once the
    # model can fit them.
    trim_opts = {"start": 0.05,
                 "end": 0.0,
                 "hold_epochs": 50,
                 "anneal_epochs": max(1, nepochs - 100),
                 "shape": "cosine"}
  if focus_opts is None:
    # default: no focal weighting (the opt-in baseline). shape
    # "const" holds start every epoch, and a gamma (start) of
    # -1 is <= 0, so loss() takes the plain-mean path -- runs
    # match no-focus unless a real focus_opts is passed.
    focus_opts = {"shape": "const",
                  "start": -1.0}

  out_dim = chi2fn.dest_idx.numel()

  # sqrt-batch-size rule: lr ~ sqrt(bs).
  learning_rate = (lr_opts["lr_base"]
                   * (bs / lr_opts["bs_base"]) ** 0.5)

  torch.manual_seed(seed)

  model = make_model(model_opts=model_opts,
                     input_dim=train_set["C"].shape[1],
                     output_dim=out_dim,
                     device=device)

  opt = make_optimizer(model=model,
                       opt_opts=opt_opts,
                       lr=learning_rate,
                       device=device)

  sched = make_scheduler(optimizer=opt, sched_opts=sched_opts)

  if device.type == "cuda":
    budget = torch.cuda.mem_get_info()[0]   # NVIDIA only
  else:
    budget = gpu_mem_gb * 1024**3           # GB -> bytes

  data  = build_loaders(device=device,
                        train_set=train_set,
                        val_set=val_set,
                        param_geometry=param_geometry,
                        chi2fn=chi2fn,
                        model=model,
                        bs=bs,
                        budget=budget)

  wmupe = lr_opts["warmup_epochs"]

  (train_losses, medians, means,
   fracs) = training_loop_batched(nepochs=nepochs,
                                  optimizer=opt,
                                  scheduler=sched,
                                  model=model,
                                  bs=bs,
                                  lossfn=chi2fn,
                                  mode=loss_mode,
                                  data=data,
                                  thresholds=thresholds,
                                  warmup_epochs=wmupe,
                                  trim_opts=trim_opts,
                                  focus_opts=focus_opts,
                                  use_amp=use_amp,
                                  silent=silent)

  return model, train_losses, medians, means, fracs
