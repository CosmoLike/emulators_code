---
name: notebook-to-python-translation
description: "TODO (next, after the high-T TATT stress test): translate the pytorch1.ipynb emulator notebook into standalone Python files with command-line options. STYLE/STRUCTURE based on dataset_generator_lensing.py (in the roman_real cocoa project, external_modules/code/emulators/emultrf/emultraining/): argparse with one --flag per option (dest=, help=, type=, required=/default=, choices= where bounded), a YAML training_args block driving config, a single orchestrating class run from __main__, ROOTDIR-relative paths, MPI + checkpointing patterns where relevant. Many files will be CLI tools. During translation: double-check/upgrade documentation; and where a piece of the notebook code is not yet understood, leave a VISIBLE WARNING marker -- the user will explicitly request those be explained/figured out later (do NOT silently guess)."
metadata:
  node_type: memory
  type: project
---

**STATUS 2026-06-29: STRUCTURE DONE.** The notebook's "Data Vector emulator
exercise 1" section is now the `emulator/` package + `driver/` CLI scripts (full
layout, helpers, drivers, search-range convention in [[emulator-python-package]];
the .py code style in [[py-module-style-conventions]]). The two rules below STILL
apply to the remaining cells and any new modules. What's left: wider CLI coverage
of the rest of the notebook, more drivers/variants (ResCNN, IA/TATT), and the
per-module documentation double-check.

TODO queued 2026-06-26 (do AFTER the high-T TATT stress test,
[[npce-and-ia-template-factoring]]): translate pytorch1.ipynb (the cosmic-shear
emulator) from a Jupyter notebook into standalone **Python files with
command-line options**. Many of the resulting files will be CLI tools.

**Coding style / structure -- based on `dataset_generator_lensing.py`** (the
user's data-generation script in the cocoa roman_real project, at
external_modules/code/emulators/emultrf/emultraining/; the user dropped it as the
template). Salient patterns to mirror:
- `argparse.ArgumentParser(prog=...)`, one `parser.add_argument("--flag", dest=...,
  help=..., type=..., required=True / default=..., choices=[...] where bounded)`
  per option; `args, unknown = parser.parse_known_args()`.
- A **YAML** config drives the run -- a `train_args` block (the script validates
  required keys and blocks: params / likelihood / train_args, probe / ord /
  fiducial / params_covmat_file).
- A single orchestrating **class** (`class dataset:`) whose `__init__` does
  setup-then-run; `__name__ == "__main__"` builds it and `MPI.Abort` on exception.
- **ROOTDIR-relative paths** (`os.environ["ROOTDIR"]` + project subfolders);
  outputs to `chains/` with name-encoded suffixes (probe, temperature).
- Heavy infra it already solves and we may reuse: MPI master/worker dispatch with
  per-task timeouts, RAM-vs-memmap staging gated on psutil, checkpoint
  save/load/append (--freqchk/--loadchk/--append), capture_native_output() to trap
  Fortran/C (CAMB/cosmolike) stdout/stderr at the fd level.

**Two standing instructions for the translation itself:**
- **Double-check documentation** as we go (formal Arguments/Returns docstrings,
  [[docstrings-formal-arguments-block]]; the house teaching style).
- **Where a piece of the notebook code is NOT yet understood, leave a VISIBLE
  WARNING marker in place** (the user could not fully parse some pieces and will
  EXPLICITLY request those be explained/resolved later). Do NOT silently guess or
  paper over them -- flag, and wait for the user to ask.

**Why:** records the next major work item and pins the exact coding-style template
(dataset_generator_lensing.py) + the two translation rules, so the translation
starts in the right structure without re-litigating style.
