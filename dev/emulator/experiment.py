"""One configured single-network cosmic-shear emulator run.

This class factors the driver setup boilerplate (parse the config, pick the
device, stage the sources, build the parameter + data-vector geometries and
the chi2, assemble the run_emulator spec dicts, train) into one reusable
object a driver or sweep script (over N_train, or one hyperparameter) need
not copy. What exp.run() orchestrates:

    YAML config
       │  from_yaml / from_config   validate blocks, collapse search
       │                            ranges, (name, ia) -> model class
       ▼
    stage_train / stage_val         load the dumps, apply the physics
       │                            cuts, split the rows
       ▼
    build_geometry                  ParamGeometry (whiten in) +
       │                            DataVectorGeometry (whiten out)
       │                            + the chi2 (one cosmolike read);
       │                            needs_bins models also get the
       │                            shear angle map (bin_sizes)
       ▼
    train                           build_specs -> run_emulator
       │                            (model / optimizer / scheduler /
       │                            loaders; one- or two-phase)
       ▼
    model + histories on the instance, ready for diagnostics
       (frac_above, eval_source_chi2, plotting)

Build it from a YAML file (from_yaml) or a parsed config mapping (from_config,
e.g. load the YAML once and rebuild from a tweaked copy per sweep point); both
resolve the model class from train_args.model.name through MODELS. The
expensive pieces are built by explicit methods and cached:

  - a single run (the driver): exp = from_yaml(...); exp.run().
  - an N_train sweep (geometry per training subset):
      for N in sizes:
        exp.stage_train(n_train=N); exp.stage_val()
        exp.build_geometry(); exp.train()
        f = exp.frac_above(0.2)
  - a hyperparameter sweep (data + geometry fixed, only the spec varies):
      exp.stage_train(); exp.stage_val(); exp.build_geometry()
      for v in values:
        exp.train(train_args=tweaked_copy_with(v))
"""

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

from .data_staging import read_param_names, load_source, phys_cut_idx
from .geometries_parameter import ParamGeometry, AmplitudeFactorGeometry
from .loss_functions import make_chi2
from .emulator_designs import ResMLP, ResCNN, ResTRF
from .IA.emulator_designs import (TemplateMLP, TemplateResCNN,
                                  TemplateResTRF)
from .IA.loss_functions import (TemplateFactoredChi2, nla_coeffs,
                                tatt_coeffs)
from .activations import make_activation
from .training import (
  run_emulator, build_run_specs, pick_device, make_logger,
  default_train_args, eval_source_chi2)


# (architecture, ia) -> model class. Two orthogonal YAML choices:
# train_args.model.name picks the ARCHITECTURE (resmlp = residual MLP;
# rescnn = + a theta-order 1D-CNN correction head; restrf = + a
# bin-token transformer correction head, attention across the
# tomographic bins with per-bin unique MLPs), and the separate
# train_args.model.ia key layers a factored intrinsic-alignment design
# on it (absent/None = the plain emulator; "nla" = the model emits
# three templates from the non-amplitude inputs and the loss combines
# them in closed form as K0 + A1*K1 + A1^2*K2, so the amplitude never
# enters the network -- exact generalization in A1; "tatt" = the
# same factoring with 3 amplitudes (a1, a2, b_TA) and 10 templates,
# exact in all three -- see tatt_coeffs). The classes carry
# capability flags build_geometry / build_specs read: factored
# (AmplitudeFactorGeometry input + the template-combining loss),
# needs_geom (geom injected for the fixed full<->theta basis buffers;
# compile_mode defaulted to "default"), and needs_bins
# (build_shear_angle_map run on the data geometry, attaching the
# per-bin split the bin-token head needs).
MODELS = {("resmlp", None):   ResMLP,
          ("rescnn", None):   ResCNN,
          ("restrf", None):   ResTRF,
          ("resmlp", "nla"):  TemplateMLP,
          ("rescnn", "nla"):  TemplateResCNN,
          ("restrf", "nla"):  TemplateResTRF,
          # tatt reuses the SAME factored classes: only the
          # IA_DESIGNS entry (amplitude columns, polynomial,
          # template count) differs -- the classes are generic in
          # n_amps / n_templates.
          ("resmlp", "tatt"): TemplateMLP,
          ("rescnn", "tatt"): TemplateResCNN,
          ("restrf", "tatt"): TemplateResTRF}

# the amplitude column the NLA design factors out of the network input.
# LSST_A1_1 is the NLA amplitude (enters xi as a linear field
# coefficient, so it factors exactly); LSST_A1_2 is the redshift-
# evolution power eta, which sits inside the projection integral and
# stays an emulated input.
NLA_AMP_NAMES = ["LSST_A1_1"]

# the three amplitude columns the TATT design factors out, in
# tatt_coeffs order [a1, a2, b_TA]: LSST_A1_1 (the linear/TA
# amplitude), LSST_A2_1 (the quadratic/TT amplitude), LSST_BTA_1
# (the density-weighting bias). The redshift-evolution powers
# (LSST_A1_2, LSST_A2_2) sit inside the projection integrals and
# stay emulated inputs, exactly as NLA's eta does.
TATT_AMP_NAMES = ["LSST_A1_1", "LSST_A2_1", "LSST_BTA_1"]

# one entry per factored IA design (the model.ia key): the amplitude
# columns the input geometry appends (in coeff_fn order), the amplitude
# polynomial, and the template count the model emits. Everything
# downstream (AmplitudeFactorGeometry, TemplateFactoredChi2, the
# models' n_amps / n_templates, the conv head's groups values) reads
# this entry -- a new design is a new entry, never new code paths.
# NOTE (tatt): the entry is live, but the template training dumps do
# not exist yet -- a tatt run needs dv dumps holding the 10 templates.
IA_DESIGNS = {"nla":  {"amp_names":   NLA_AMP_NAMES,
                       "coeff_fn":    nla_coeffs,
                       "n_templates": 3},
              "tatt": {"amp_names":   TATT_AMP_NAMES,
                       "coeff_fn":    tatt_coeffs,
                       "n_templates": 10}}

# The nested model-block schema. The YAML groups each component's
# knobs in its own sub-block (mlp / activation / cnn / trf), so the
# nesting carries the context and the keys stay short -- no
# n_blocks-vs-n_blocks_cnn suffixes. The tables below map each
# sub-block's YAML keys onto the model constructors' flat argument
# names (the constructors are internal API; the YAML is the
# interface). An unknown key raises, listing what is allowed.
MODEL_BLOCK_KEYS = {
  "mlp": {"width":        "int_dim_res",
          "n_blocks":     "n_blocks"},
  "cnn": {"kernel_size":    "kernel_size",
          "rescale_kernel": "rescale_kernel",
          "groups":         "groups",
          "separable":      "separable",
          "film":           "film",
          "n_blocks":       "n_blocks_cnn",
          "gate_init":      "gate_init"},
  "trf": {"n_heads":      "n_heads",
          "n_blocks":     "n_blocks_trf",
          "n_mlp_blocks": "n_mlp_blocks",
          "shared_mlp":   "shared_mlp",
          "film":         "film",
          "gate_init":    "gate_init"},
}

# which head sub-block each architecture accepts (None = trunk only);
# a cnn: block under name: restrf (etc.) is a config error, caught in
# build_specs.
ARCH_HEAD = {"resmlp": None, "rescnn": "cnn", "restrf": "trf"}

# default reported delta-chi2 cutoffs; the first (0.2) is the emulator goal
# and the best-model-selection metric.
DEFAULT_THRESHOLDS = torch.tensor([0.2, 0.5, 1.0, 10.0, 100.0])


class EmulatorExperiment:
  """
  Configuration + environment for one single-network cosmic-shear (xi)
  emulator, reusable across a single run and across sweeps.

  The constructor builds only the cheap, config-derived state (device,
  parameter names, the quiet-gated logger); the staged data and geometry
  come from explicit cached methods, so a sweep rebuilds only what varies.
  The fixed single-emulator choices (probe = xi, AdamW, ReduceLROnPlateau,
  use_amp = False, the report thresholds) are constructor defaults, so a
  driver passes only what it varies (the YAML, rescale, activation, quiet);
  the model is the config's choice (train_args.model.name).
  """

  def __init__(self,
               data,
               train_args,
               model_cls,
               opt_cls=optim.AdamW,
               sched_cls=lr_scheduler.ReduceLROnPlateau,
               probe="xi",
               thresholds=None,
               use_amp=False,
               rescale="none",
               activation="H",
               device=None,
               quiet=False,
               raw_train_args=None):
    """
    Store the config + fixed choices; build the cheap derived state.

    Arguments:
      data       = the "data" block: input paths plus cut / split /
                   cosmolike-dataset settings. Keys:
                     train_dv = training data-vector .npy (memmapped);
                     train_params = training parameter .txt (columns:
                       weight, lnp, modeled params, chi2);
                     train_covmat = parameter covmat (header line =
                       parameter names);
                     val_dv = validation data-vector .npy;
                     val_params = validation parameter .txt;
                     cosmolike_data_dir = folder under
                       external_modules/data;
                     cosmolike_dataset = .dataset ini naming the cov /
                       mask / data-vector files;
                     omegabh2_cut = drop rows with omega_b h^2 >= this;
                     omegabh2_lo = optional lower bound on omega_b h^2
                       (rows at or below it dropped; omit for no cut);
                     omegam2h2_lo / omegam2h2_hi = optional window on
                       omegam^2 h^2 = (Omega_m H0/100)^2; rows outside
                       are dropped (omit a key for no cut on that side);
                     train_divisor / val_divisor = keep N // divisor of
                       train / val rows;
                     split_seed = seed for the cut+shuffle picking train /
                       val rows;
                     ram_frac = optional (default 0.7): RAM fraction the
                       staged subset may fill.
      train_args = resolved "train_args" block, range-free (search ranges
                   collapsed to scalars, e.g. by default_train_args).
                   Top keys:
                     nepochs = passes over the training set;
                     bs = minibatch size;
                     loss_mode = optional (default "sqrt"): per-sample
                       transform "sqrt" / "chi2" / "sqrt_dchi2";
                     silent = optional (default False): silence the run;
                     trunk_epochs = optional (default 0): two-phase
                       schedule -- see run_emulator;
                     trunk / head = optional symmetric mappings of
                       per-phase overrides (lr_base / loss_mode /
                       trim / focus / clip / rewind) over the shared
                       top-level defaults; need trunk_epochs > 0 --
                       see run_emulator;
                     clip = optional (default 0.0 = off): per-step
                       gradient-norm ceiling -- see run_emulator;
                     rewind = optional (default False): reload the
                       best weights + optimizer snapshot at every
                       plateau lr cut -- see run_emulator.
                   Plus six constructible sub-blocks (each a mapping):
                     model = the NESTED model block: "name" (the
                       architecture: resmlp | rescnn | restrf) and
                       "ia" (the factored IA design layered on it:
                       omit for plain, "nla" or "tatt"; the pair
                       picks the class), then one sub-block per
                       component --
                       "mlp" (width, n_blocks; the trunk, required),
                       "activation" ({type, n_gates} or a bare type
                       string; see the `activation` argument below),
                       "cnn" (kernel_size, rescale_kernel, groups,
                       separable, film, n_blocks, gate_init; name
                       rescnn only), "trf" (n_heads, n_blocks,
                       n_mlp_blocks, shared_mlp, film, gate_init;
                       name restrf only --
                       the tokens live at the natural bin width, so
                       there is no width knob) --
                       plus an optional flat "compile_mode".
                       build_specs translates the nesting onto the
                       constructors' flat kwargs (MODEL_BLOCK_KEYS)
                       and rejects unknown or misplaced keys;
                     optimizer = weight_decay (+ any extra AdamW kwargs);
                     lr = lr_base, bs_base, warmup_epochs (run sets
                       lr = lr_base * sqrt(bs / bs_base));
                     scheduler = mode, patience, factor (ReduceLROnPlateau
                       kwargs);
                     trim = trim schedule -- start, end, hold_epochs,
                       anneal_epochs, shape (see anneal_value);
                     focus = focal-weight schedule -- start, end,
                       hold_epochs, anneal_epochs, shape, kappa.
      model_cls  = the model class (ResMLP / ResCNN); from_config
                   resolves it from train_args.model.name.
      opt_cls    = optimizer class (default AdamW).
      sched_cls  = scheduler class (default ReduceLROnPlateau).
      probe      = cosmolike probe (default "xi").
      thresholds = reported delta-chi2 cutoffs (default
                   DEFAULT_THRESHOLDS); thresholds[0] selects the best model.
      use_amp    = run the forward in low-precision autocast (default False).
      rescale    = analytic-R mode forwarded to make_chi2 ("none" /
                   "rescaled" / "residual").
      activation = ResBlock activation name (make_activation): "H" /
                   "power" / "multigate" / "gated_power". from_config
                   resolves the precedence: an explicit value here (the
                   drivers' --activation) wins over the YAML's
                   train_args.model.activation, which wins over "H".
      device     = compute device (default: pick_device()).
      quiet      = if True, silence the instance logger and the
                   per-source / per-epoch prints.
      raw_train_args = un-collapsed train_args (search ranges intact), for
                   a search driver that resolves them per trial; defaults
                   to train_args (from_config supplies the raw block).
    """
    self.data       = data
    self.train_args = train_args
    self.model_cls  = model_cls
    # display name + IA design + architecture, overwritten by
    # from_config with the YAML's composed name/ia. The
    # direct-construction fallbacks: a factored class defaults to
    # "nla" (a direct-construction tatt run must set exp.ia = "tatt"
    # itself -- from_config does this from the YAML); arch stays
    # None, which skips build_specs' head-block-vs-architecture check.
    self.model_name = model_cls.__name__.lower()
    self.ia = ("nla" if getattr(model_cls, "factored", False)
               else None)
    self.arch = None
    self.opt_cls    = opt_cls
    self.sched_cls  = sched_cls
    self.probe      = probe
    self.thresholds = (DEFAULT_THRESHOLDS if thresholds is None
                       else thresholds)
    self.use_amp    = use_amp
    self.rescale    = rescale
    self.activation = activation
    self.quiet      = quiet

    # make_logger / pick_device (training.py): a print(*a) gated on quiet,
    # and the compute device (cuda > mps > cpu).
    self.log        = make_logger(quiet=quiet)
    self.device     = pick_device() if device is None else device

    # un-collapsed train_args (search ranges intact) for a per-trial search
    # driver; defaults to the resolved train_args when no raw block given.
    self.raw_train_args = (train_args if raw_train_args is None
                           else raw_train_args)

    # TF32 tensor-core float32 matmuls (Ampere+); no-op on CPU / MPS. A
    # one-time global switch.
    torch.set_float32_matmul_precision("high")

    # read_param_names (data_staging.py): parameter column names from the
    # covmat's "#"-prefixed header line. Reused for the val cut (same
    # columns).
    self.names = read_param_names(data["train_covmat"])

    # artifacts the methods below build; cached across a sweep (None until
    # built).
    self.train_set = None
    self.val_set   = None
    self.pgeom     = None
    self.geom      = None
    self.chi2fn    = None
    self.model     = None

  # --- alternative constructors ---
  @classmethod
  def from_config(cls, cfg, models=None, **kwargs):
    """
    Build from an already-parsed config mapping.

    Validates the required blocks, collapses train_args search ranges to
    their defaults, and resolves the (train_args.model.name,
    train_args.model.ia) pair -> a model class through `models`. Use it
    to rebuild from a tweaked copy of a config dict (one sweep point).

    Arguments:
      cfg    = mapping with a "data" block and a "train_args" block (the
               YAML schema; see __init__ for each block's keys).
      models = (name, ia) -> class registry (default MODELS: name is
               the architecture, resmlp | rescnn | restrf; ia the
               factored IA design layered on it, None | "nla" |
               "tatt").
      **kwargs = forwarded to __init__ (opt_cls, sched_cls, probe,
               thresholds, use_amp, rescale, activation, device, quiet).

    Returns:
      an EmulatorExperiment with the resolved data / train_args / model.
    """
    models = MODELS if models is None else models
    for block in ("data", "train_args"):
      if block not in cfg:
        raise KeyError(
          f"config is missing the required block: {block!r}")
    # default_train_args (training.py): walk train_args, collapsing every
    # [default, min, max, kind] search range to its default (first) value,
    # so a tuning YAML builds a concrete run.
    ta = default_train_args(cfg["train_args"])
    # read (not pop) name / ia -- build_specs strips both from the
    # spread, so they never reach the model constructor. A YAML `ia:
    # none` parses as the STRING "none" (YAML's nulls are null/~), so
    # accept it as None too.
    name = str(ta["model"].get("name", "resmlp")).lower()
    ia   = ta["model"].get("ia")
    ia   = None if ia in (None, "none") else str(ia).lower()
    if (name, ia) not in models:
      archs = []
      ias   = []
      for n, i in models:
        if n not in archs:
          archs.append(n)
        if i is not None and i not in ias:
          ias.append(i)
      raise ValueError(
        f"unknown model: name={name!r}, ia={ia!r}. name picks the "
        f"architecture ({' | '.join(sorted(archs))}); the separate ia "
        f"key layers a factored intrinsic-alignment design on it "
        f"({' | '.join(sorted(ias))}; omit it for the plain emulator)")
    # activation precedence, resolved once here: an explicit caller choice
    # (the drivers' --activation flag; they pass None when the flag is
    # absent) wins over the YAML's model.activation block, which wins
    # over the "H" default. The block nests {type, n_gates}; a bare
    # string (activation: H) is accepted as shorthand for the type.
    # build_specs consumes n_gates and drops the block (it is not a
    # model-constructor kwarg).
    if kwargs.get("activation") is None:
      act_blk = ta["model"].get("activation")
      if isinstance(act_blk, dict):
        kwargs["activation"] = str(act_blk.get("type", "H"))
      elif act_blk is not None:
        kwargs["activation"] = str(act_blk)
      else:
        kwargs["activation"] = "H"
    exp = cls(data=cfg["data"], train_args=ta,
              model_cls=models[(name, ia)],
              raw_train_args=cfg["train_args"], **kwargs)
    # the composed display name (run_tag / the banner / file names):
    # the architecture, suffixed by the IA design when one is layered
    # (resmlp_nla, rescnn_nla). exp.ia drives the factored-design
    # lookups (IA_DESIGNS) and exp.arch the head-block validation
    # (ARCH_HEAD) in build_geometry / build_specs.
    exp.model_name = name if ia is None else f"{name}_{ia}"
    exp.ia   = ia
    exp.arch = name
    return exp

  @classmethod
  def from_yaml(cls, path, models=None, **kwargs):
    """
    Build from a YAML config file.

    Thin wrapper: read the file, then from_config (see it for the
    resolution and **kwargs).

    Arguments:
      path   = path to the YAML config (data + train_args blocks).
      models = name -> class registry (default MODELS).
      **kwargs = forwarded to from_config -> __init__.

    Returns:
      an EmulatorExperiment.
    """
    with open(path) as f:
      cfg = yaml.safe_load(f)
    return cls.from_config(cfg, models=models, **kwargs)

  # --- the startup banner ---
  def print_design(self):
    """
    Print the resolved run design to stdout (the startup banner).

    Announces the full design before anything trains, so a stale YAML
    is caught at launch and not one 17-minute training (or one whole
    sweep / study) later. Shared by every driver; quiet-gated through
    self.log, so --quiet silences it with the rest.

    Lines printed, in order:

        device / model class / activation / rescale
           |  the environment line: what runs where
           v
        model spec                the resolved model block (name, ia,
           |                      mlp / cnn / trf sub-blocks) after
           |                      default_train_args collapsed ranges
           v
        run: nepochs bs loss_mode (+ the two-phase split when
           |                       trunk_epochs > 0)
           v
        guards: clip / rewind     only when either is set
           |
           v
        one line per sub-block    optimizer / lr / scheduler / trim /
           |                      focus / trunk / head, each printed
           |                      only when present in train_args
           v
        cuts                      the physical omegabh2 / omegam2h2
                                  windows from the data block

    A sweep or a study varies pieces per point / per trial; this
    banner shows the resolved defaults those variations start from.
    """
    ta = self.train_args
    d  = self.data
    self.log(f"device: {self.device}  |  "
             f"model: {self.model_cls.__name__}  |  "
             f"activation: {self.activation}  |  "
             f"rescale: {self.rescale}")
    self.log(f"model spec: {ta['model']}")
    # trunk_epochs > 0 = the two-phase schedule (trunk then frozen-trunk
    # head); print it only when active, so ordinary runs stay unchanged.
    tk = ta.get("trunk_epochs", 0)
    ph = (f"  (two-phase: {tk} trunk + {ta['nepochs'] - tk} head)"
          if tk else "")
    self.log(f"run: nepochs {ta['nepochs']}  bs {ta['bs']}  "
             f"loss_mode {ta.get('loss_mode', 'sqrt')}{ph}")
    # the stability guards (training.py run_emulator: clip = per-step
    # gradient-norm ceiling, rewind = reload the best snapshot at every
    # plateau lr cut); printed only when set, like the two-phase line.
    clip   = ta.get("clip", 0.0)
    rewind = ta.get("rewind", False)
    if clip or rewind:
      self.log(f"guards: clip {clip}  rewind {rewind}")
    # the remaining train_args sub-blocks, one dict per line (including
    # the per-phase trunk / head override blocks), so the whole resolved
    # config is on the terminal.
    for block in ("optimizer", "lr", "scheduler", "trim", "focus",
                  "trunk", "head"):
      if block in ta:
        self.log(f"{block}: {ta[block]}")
    self.log(f"cuts: omegabh2 in "
             f"({d.get('omegabh2_lo')}, {d.get('omegabh2_cut')})  "
             f"omegam2h2 in "
             f"({d.get('omegam2h2_lo')}, {d.get('omegam2h2_hi')})")

  # --- staging + geometry (the expensive, cached pieces) ---
  def stage_train(self, n_train=None):
    """
    Stage the training source (cached as self.train_set).

    A generator freshly seeded from data["split_seed"] fixes the
    cut+shuffle pool, so slicing it to different sizes gives nested subsets
    -- the right thing for a learning-curve sweep.

    Arguments:
      n_train = absolute number of training rows to keep; None (default)
                uses the YAML data["train_divisor"] (N // divisor).

    Returns:
      the training source dict.
    """
    d   = self.data
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): memmap the dv .npy, apply the physical
    # cuts (omega_b h^2 < cut; optional omegam^2 h^2 window), keep n_keep
    # (or N // divisor) rows of the seeded shuffle, stage in RAM if they
    # fit (else the memmap), return {C, dv, idx} (+ C_mean / dv_mean with
    # with_means).
    self.train_set = load_source(
      dv_path=d["train_dv"],
      params_path=d["train_params"],
      names=self.names,
      cut=d["omegabh2_cut"],
      omegabh2_lo=d.get("omegabh2_lo"),
      omegam2h2_lo=d.get("omegam2h2_lo"),
      omegam2h2_hi=d.get("omegam2h2_hi"),
      divisor=(None if n_train is not None else d["train_divisor"]),
      n_keep=n_train,
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=True,
      verbose=not self.quiet)
    return self.train_set

  def stage_val(self, n_val=None):
    """
    Stage the validation source (cached as self.val_set).

    Seeded from data["split_seed"] like the train source (the val file
    differs, so the same seed gives an independent selection). Carries no
    means -- geometry centers come from the training source only.

    Arguments:
      n_val = absolute number of validation rows to keep; None (default)
              uses the YAML data["val_divisor"].

    Returns:
      the validation source dict.
    """
    d   = self.data
    gen = torch.Generator().manual_seed(int(d["split_seed"]))
    # load_source (data_staging.py): same staging as stage_train, on the
    # val files; with_means=False (val borrows the training centers).
    self.val_set = load_source(
      dv_path=d["val_dv"],
      params_path=d["val_params"],
      names=self.names,
      cut=d["omegabh2_cut"],
      omegabh2_lo=d.get("omegabh2_lo"),
      omegam2h2_lo=d.get("omegam2h2_lo"),
      omegam2h2_hi=d.get("omegam2h2_hi"),
      divisor=(None if n_val is not None else d["val_divisor"]),
      n_keep=n_val,
      gen=gen,
      ram_frac=d.get("ram_frac", 0.7),
      with_means=False,
      verbose=not self.quiet)
    return self.val_set

  def pool_size(self):
    """
    Number of physically-cut training rows available -- the natural top
    of an N_train sweep.

    Loads the training parameter file, keeps the modeled columns, applies
    the physical cuts (the omega_b h^2 bound and the optional
    omegam^2 h^2 window, same cuts as stage_train), counts the survivors.
    Order-independent, so no shuffle or staging.

    Returns:
      the number of training rows passing the physical cuts (an int).
    """
    d = self.data
    # modeled parameter columns (drop leading weight / lnp and trailing
    # chi2), as load_source does by default.
    C   = np.loadtxt(d["train_params"], dtype="float32")[:, slice(2, -1)]
    idx = np.arange(C.shape[0])
    # phys_cut_idx (data_staging.py): keep rows with omega_b h^2 < cut
    # and omegam^2 h^2 inside the optional window (both rarefied,
    # catastrophically-failing corners).
    phys = phys_cut_idx(C=C, idx=idx, names=self.names,
                        cut=d["omegabh2_cut"],
                        omegabh2_lo=d.get("omegabh2_lo"),
                        omegam2h2_lo=d.get("omegam2h2_lo"),
                        omegam2h2_hi=d.get("omegam2h2_hi"))
    return int(len(phys))

  def build_geometry(self, train_set=None):
    """
    Build the input + output geometries and the chi2 (cached as
    self.pgeom / self.geom / self.chi2fn).

    Whitening centers come from the training means, so this depends on the
    training subset: rebuild per subset in an N_train sweep, build once for
    a hyperparameter sweep (independent of model / train_args).

    Arguments:
      train_set = training source dict with "C_mean" / "dv_mean" / "C" /
                  "idx" (default: self.train_set, from stage_train).

    Returns:
      (pgeom, geom, chi2fn).
    """
    train_set = self.train_set if train_set is None else train_set
    d = self.data
    # config validation first, before the cosmolike import below: a bad
    # combination should fail fast, and stays testable off-workstation.
    if getattr(self.model_cls, "factored", False) and self.rescale != "none":
      raise ValueError(
        f"model {self.model_name!r} does not compose with --rescale "
        "(the factored loss owns the target construction)")
    # lazy import: DataVectorGeometry.from_cosmolike pulls in cosmolike,
    # which lives only on the workstation -- importing here keeps the module
    # importable for the config logic without cosmolike.
    from .geometries_output import DataVectorGeometry

    # The input whitening. The plain designs whiten every parameter
    # (ParamGeometry); the factored designs (the models' factored flag,
    # picked by model.ia) instead whiten only the non-amplitude
    # columns and append the raw amplitudes last
    # (AmplitudeFactorGeometry), so the model can drop them and the loss
    # can read them -- the amplitudes never enter the network. Which
    # columns / polynomial / template count is the IA_DESIGNS[self.ia]
    # entry.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      self.pgeom = AmplitudeFactorGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"],
        amp_names=des["amp_names"])
    else:
      # ParamGeometry.from_covmat (geometries_parameter.py): eigendecompose
      # the parameter covmat so encode() centers, rotates, unit-scales the
      # params the model sees.
      self.pgeom = ParamGeometry.from_covmat(
        device=self.device,
        center=train_set["C_mean"],
        covmat_path=d["train_covmat"])

    # DataVectorGeometry.from_cosmolike (geometries_output.py): the output
    # geometry -- read cosmolike's cov / mask / inverse-cov, eigendecompose
    # the kept (unmasked) block, so encode()/chi2 whiten + score the dv.
    self.geom = DataVectorGeometry.from_cosmolike(
      device=self.device,
      dv_center=train_set["dv_mean"],
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"],
      probe=self.probe)

    # bin-token heads (restrf; the needs_bins flag) split the dv per
    # tomographic bin: build_shear_angle_map (geometries_output.py)
    # attaches bin_sizes to the geometry, reading only the dataset ini
    # and the n(z) file -- no cosmolike.
    if getattr(self.model_cls, "needs_bins", False):
      from .geometries_output import build_shear_angle_map
      build_shear_angle_map(geom=self.geom,
                            data_dir=d["cosmolike_data_dir"],
                            dataset=d["cosmolike_dataset"])

    # The loss. The factored designs combine the model's templates in
    # closed form (nla: xi = K0 + A1*K1 + A1^2*K2 via nla_coeffs),
    # reading each sample's own amplitudes off the encoded input's last
    # columns, then score the plain chi2 on the combined xi.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      self.chi2fn = TemplateFactoredChi2(
        geom=self.geom,
        coeff_fn=des["coeff_fn"],
        n_amps=len(des["amp_names"]))
      return self.pgeom, self.geom, self.chi2fn

    # make_chi2 (loss_functions.py): wrap geom in the loss -- plain
    # CosmolikeChi2, or the analytic-R RescaledChi2 / ResidualBaseChi2 when
    # rescale != "none". cosmo_mid = training-cloud mean (R = 1 there for a
    # rescaled chi2; the plain chi2 ignores it).
    self.chi2fn = make_chi2(
      geom=self.geom,
      rescale=self.rescale,
      param_geometry=self.pgeom,
      cosmo_mid=train_set["C"][train_set["idx"]].mean(0),
      data_dir=d["cosmolike_data_dir"],
      dataset=d["cosmolike_dataset"])

    return self.pgeom, self.geom, self.chi2fn

  # --- per-run pieces ---
  def build_specs(self, train_args=None):
    """
    Assemble the six run_emulator spec dicts for one run.

    build_run_specs from train_args, then inject the named activation and
    -- for ResCNN only -- the data geometry (see body comments). A
    hyperparameter sweep passes a varied train_args.

    Arguments:
      train_args = resolved train_args mapping (default:
                   self.train_args). Range-free (from default_train_args
                   / suggest_train_args). A leftover model.name is
                   stripped here, so a suggest_train_args result works too.

    Returns:
      the keyed spec dict run_emulator consumes as **specs.
    """
    train_args = self.train_args if train_args is None else train_args
    # Translate the nested YAML model block into the constructors'
    # flat kwargs (MODEL_BLOCK_KEYS). name / ia picked the class
    # (from_config resolved them); the activation type was resolved by
    # from_config too (re-reading it here would let the YAML overrule
    # an explicit --activation), so only its n_gates is consumed here;
    # compile_mode passes through (make_model strips it). Everything
    # else must be one of the component sub-blocks. A head block for a
    # head this architecture does not have is IGNORED, not an error:
    # keeping cnn: and trf: both configured lets a run switch
    # architectures by changing name: alone. Unknown keys -- top-level
    # or inside the ACTIVE blocks -- still raise, so a misspelled knob
    # that would affect the run fails loudly.
    ta = dict(train_args)
    model_opts = {}
    n_gates = 3
    head = ARCH_HEAD.get(self.arch) if self.arch is not None else None
    for key, sub in ta["model"].items():
      if key in ("name", "ia"):
        continue
      if key == "activation":
        if isinstance(sub, dict) and "n_gates" in sub:
          n_gates = int(sub["n_gates"])
        continue
      if key == "compile_mode":
        model_opts["compile_mode"] = sub
        continue
      if key in MODEL_BLOCK_KEYS:
        # skip the inactive head's block entirely (its contents are
        # not even validated -- a stale key in a block that cannot
        # affect this run should not stop it). With arch unknown
        # (direct construction) every present block is translated.
        if (self.arch is not None and key != "mlp"
            and key != head):
          continue
        table = MODEL_BLOCK_KEYS[key]
        for k2, v2 in sub.items():
          if k2 not in table:
            raise ValueError(
              f"unknown key model.{key}.{k2}; allowed: "
              f"{' / '.join(sorted(table))}")
          model_opts[table[k2]] = v2
        continue
      raise ValueError(
        f"unknown model key {key!r}; the model block nests its "
        "knobs: name / ia / mlp / activation / cnn / trf / "
        "compile_mode")
    if "int_dim_res" not in model_opts:
      raise ValueError(
        "the model.mlp block (width, n_blocks) is required -- every "
        "architecture is built on the ResMLP trunk")
    ta["model"] = model_opts

    # build_run_specs (training.py): turn the train_args sub-blocks into the
    # six {cls, **kwargs} spec dicts run_emulator consumes (model_opts /
    # opt_opts / lr_opts / sched_opts / trim_opts / focus_opts), with this
    # experiment's fixed classes.
    specs = build_run_specs(
      train_args=ta,
      model_cls=self.model_cls,
      opt_cls=self.opt_cls,
      sched_cls=self.sched_cls)

    # make_activation (activations.py): map the activation name to a
    # factory act(dim) -> nn.Module (the paper's H, or a Power / Gated /
    # GatedPower variant). A callable, so it cannot live in the YAML; inject
    # it into the ResBlock options (setdefault keeps config-set block_opts).
    # n_gates (YAML model.activation.n_gates, default 3) sizes the
    # multi-gate families.
    specs["model_opts"].setdefault(
      "block_opts", {})["act"] = make_activation(self.activation,
                                                 n_gates=n_gates)

    # Geometry-consuming heads (the needs_geom flag: the conv and TRF
    # models) get geom injected for their fixed full<->theta
    # basis-change buffers (+ bin_sizes for restrf); ResMLP /
    # TemplateMLP take none. compile_mode falls back to "default" for
    # them (reduce-overhead's CUDA-graph capture trips on the gated
    # skip-add); setdefault keeps a YAML-set choice.
    if getattr(self.model_cls, "needs_geom", False):
      specs["model_opts"]["geom"] = self.geom
      specs["model_opts"].setdefault("compile_mode", "default")

    # The factored models (IA/emulator_designs.py) need the
    # factored-design shape from IA_DESIGNS[self.ia]: how many
    # amplitude columns AmplitudeFactorGeometry appended (dropped from
    # the trunk input) and how many templates to emit (3 for nla:
    # GG, GI, II). setdefault keeps YAML-set overrides.
    if getattr(self.model_cls, "factored", False):
      des = IA_DESIGNS[self.ia]
      specs["model_opts"].setdefault("n_amps", len(des["amp_names"]))
      specs["model_opts"].setdefault("n_templates",
                                     des["n_templates"])

    return specs

  def train(self, train_args=None, silent=None):
    """
    Train one model on the staged sources; return its histories.

    Uses the cached sources / geometry / chi2 (build them first via
    stage_train / stage_val / build_geometry, or call run). train_args
    overrides the resolved config for this run (a hyperparameter sweep
    passes a varied copy); the model and histories stay on the instance.

    Arguments:
      train_args = resolved train_args for this run (default:
                   self.train_args).
      silent     = override run_emulator's per-epoch printing; None
                   (default) -> train_args["silent"] or self.quiet. A
                   search driver passes silent=True so trials train quietly
                   regardless of self.quiet.

    Returns:
      (model, train_losses, medians, means, fracs) -- run_emulator's
      return, the model at its best frac>0.2 epoch.
    """
    train_args = self.train_args if train_args is None else train_args
    specs = self.build_specs(train_args=train_args)
    # None -> the config/quiet default; a search driver forces silent.
    silent_run = (train_args.get("silent", False) or self.quiet
                  if silent is None else silent)

    # run_emulator (training.py): build the model / optimizer / scheduler
    # from the specs and the regime-aware loaders, train nepochs with a
    # per-epoch val pass, return the model (restored to its best frac>0.2
    # epoch) plus the histories.
    out = run_emulator(
      train_set=self.train_set,
      val_set=self.val_set,
      chi2fn=self.chi2fn,
      param_geometry=self.pgeom,
      nepochs=train_args["nepochs"],
      bs=train_args["bs"],
      loss_mode=train_args.get("loss_mode", "sqrt"),
      # two-phase schedule (trunk-then-head, factored conv/TRF heads):
      # epochs of pure-trunk training before the trunk freezes and the
      # head learns the residual; 0 / absent = ordinary joint training.
      # The symmetric trunk: / head: blocks override each pass's
      # objective (lr_base / loss_mode / trim / focus) over the shared
      # top-level defaults -- by the handoff the trunk has absorbed
      # most outliers, so the head may want e.g. loss_mode chi2 with
      # no trim.
      trunk_epochs=train_args.get("trunk_epochs", 0),
      trunk_opts=train_args.get("trunk"),
      head_opts=train_args.get("head"),
      # stability guards (both default off; the trunk: / head:
      # blocks can override either per phase): clip = per-step
      # gradient-norm ceiling (kills single-batch kicks from
      # monster outliers under a quadratic loss); rewind = on every
      # plateau lr cut reload the best weights + optimizer
      # snapshot, so an excursion into a bad basin costs at most
      # `patience` epochs instead of freezing the run there.
      clip=train_args.get("clip", 0.0),
      rewind=train_args.get("rewind", False),
      thresholds=self.thresholds,
      use_amp=self.use_amp,
      silent=silent_run,
      device=self.device,
      **specs)

    (self.model, self.train_losses, self.medians,
     self.means, self.fracs) = out
    return out

  def run(self, n_train=None, train_args=None):
    """
    The full pipeline (the driver's body) in one call.

    Stage the train + val sources, build the geometry + chi2 from the
    training subset, train once. The artifacts (train_set, val_set, pgeom,
    geom, chi2fn, model) stay on the instance for diagnostics.

    Arguments:
      n_train    = absolute training-row count (default: the YAML
                   divisor) -- the N_train sweep knob.
      train_args = resolved train_args for this run (default:
                   self.train_args).

    Returns:
      (model, train_losses, medians, means, fracs).
    """
    self.stage_train(n_train=n_train)
    self.stage_val()
    self.build_geometry(train_set=self.train_set)
    return self.train(train_args=train_args)

  # --- a sweep metric ---
  def frac_above(self, threshold=0.2, source=None, bs=256):
    """
    Fraction of a source's points with delta-chi2 > threshold.

    Scores the trained model on a source (default the val set) with
    eval_source_chi2 -- the learning-curve / sweep metric (the number
    frac>thresholds[0] tracks per epoch, recomputed here).

    Arguments:
      threshold = the delta-chi2 cutoff (default 0.2, the goal).
      source    = source dict to score (default self.val_set).
      bs        = forward batch size for the scoring.

    Returns:
      the fraction over `threshold`, a float.
    """
    source = self.val_set if source is None else source
    # eval_source_chi2 (training.py): score every row of `source` -- encode
    # params -> model -> per-row delta-chi2 against the encoded truth
    # (returns numpy params + dchi2, aligned row-for-row).
    _, dchi2 = eval_source_chi2(
      model=self.model,
      param_geometry=self.pgeom,
      chi2fn=self.chi2fn,
      source=source,
      device=self.device,
      bs=bs)
    return float((dchi2 > threshold).mean())
