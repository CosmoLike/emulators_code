"""Cocoa-framework path resolution shared by the CLI drivers.

The drivers run inside the cocoa framework, launched from $ROOTDIR rather than
from the data folder, so every path is resolved against the project layout
instead of the current directory. A --root names the project folder under
$ROOTDIR (e.g. projects/lsst_y1), a --fileroot a subfolder of it holding this
emulator's YAML and outputs (e.g. emulators/training_scripts), and the training
/ validation data files sit under the project's chains/ folder (where
dataset_generator_lensing.py writes them). add_cocoa_path_args registers
the three shared flags (--root, --fileroot, --yaml); resolve_cocoa_config reads
$ROOTDIR, builds the two roots, ensures the project chains/ folder exists, loads
the YAML from the fileroot, and rewrites the config's data paths to absolute so
the experiment reads them regardless of the launch directory. It returns the
config plus the fileroot (configs) and chains (data + run products) folders;
cocoa_output places a run output under one of them.

PS: ROOTDIR is the cocoa install root, an environment variable cocoa exports;
every project path is taken relative to it.
"""

import os
from pathlib import Path

import yaml


# data-block keys naming input files on disk; each is resolved against the
# project root. The cosmolike_* keys are NOT here: they resolve against
# $ROOTDIR/external_modules/data inside the output geometry, not the project.
_DATA_PATH_KEYS = (
  "train_dv",
  "train_params",
  "train_covmat",
  "val_dv",
  "val_params",
)


def add_cocoa_path_args(parser):
  """
  Register the shared cocoa path flags on an argument parser.

  Adds the three flags every driver shares: --root and --fileroot (the
  ROOTDIR-relative project layout) and --yaml (the config file under
  fileroot). Call once per driver before parse_known_args.

  Arguments:
    parser = the argparse.ArgumentParser to add the flags to.
  """
  parser.add_argument("--root",
                      dest="root",
                      help="project folder under $ROOTDIR holding the "
                           "data and this emulator (e.g. "
                           "projects/lsst_y1)",
                      type=str,
                      required=True)
  parser.add_argument("--fileroot",
                      dest="fileroot",
                      help="subfolder of --root holding this emulator's "
                           "YAML and outputs (e.g. "
                           "emulators/training_scripts)",
                      type=str,
                      required=True)
  parser.add_argument("--yaml",
                      dest="yaml",
                      help="config YAML under --fileroot (data + "
                           "train_args blocks); default test.yaml",
                      type=str,
                      default=None)


def resolve_cocoa_config(args):
  """
  Resolve the cocoa project layout and load the path-resolved config.

  Reads $ROOTDIR, joins --root and --fileroot under it, ensures the
  project chains/ folder exists, loads the YAML from the fileroot
  (test.yaml when --yaml is unset), and rewrites every data-block file
  path (train / val dv, params, covmat -- bare filenames in the YAML) to
  an absolute path under the project's chains/ folder. Resolving here,
  not in the YAML, lets the driver run from $ROOTDIR (the cocoa launch
  directory) without a cwd-relative path breaking.

  Arguments:
    args = the parsed CLI namespace; reads args.root, args.fileroot,
           and args.yaml (the flags add_cocoa_path_args registered).

  Returns:
    cfg      = the parsed config mapping, its data paths made absolute.
    fileroot = absolute emulator folder (<root>/<fileroot>), holding the
               YAML configs.
    chains   = absolute project chains/ folder (<root>/chains), holding
               the data files and this driver's run products.
  """
  # $ROOTDIR/<root>/<fileroot>, mirroring dataset_generator_lensing.py:
  # root holds the data and a chains/ folder; fileroot holds this
  # emulator's YAML configs.
  root_env = os.environ.get("ROOTDIR")
  if not root_env:
    raise RuntimeError("ROOTDIR environment variable is not set")
  root = root_env.rstrip("/")
  root = f"{root}/{args.root.rstrip('/')}"
  fileroot = f"{root}/{args.fileroot.rstrip('/')}"
  chains = f"{root}/chains"
  Path(chains).mkdir(parents=True, exist_ok=True)

  # the YAML lives under the emulator's fileroot; default test.yaml.
  yaml_path = (f"{fileroot}/test.yaml" if args.yaml is None
               else f"{fileroot}/{args.yaml}")
  if not os.path.isfile(yaml_path):
    raise FileNotFoundError(f"YAML file not found: {yaml_path}")
  with open(yaml_path) as f:
    cfg = yaml.safe_load(f)
  if not isinstance(cfg, dict):
    raise ValueError(f"config did not parse to a mapping: {yaml_path}")

  # rewrite each input data path to absolute, under the project chains/
  # folder -- where dataset_generator_lensing.py writes the dvs / params /
  # covmat. The YAML lists bare filenames; os.path.join puts each under
  # root/chains (and passes an absolute path through unchanged). The
  # block-presence check is left to EmulatorExperiment.from_config, which
  # gives a clearer error.
  data = cfg.get("data")
  if isinstance(data, dict):
    for key in _DATA_PATH_KEYS:
      if key in data:
        data[key] = os.path.join(root, "chains", data[key])

  return cfg, fileroot, chains


def cocoa_output(base, path):
  """
  Place a run-output path under a base folder.

  Joins a relative output path under `base` (the chains folder for run
  products like the diagnostics PDF, or the fileroot for configs); an
  absolute path passes through unchanged (os.path.join drops the
  earlier parts on an absolute tail).

  Arguments:
    base = absolute folder to place the output in (fileroot or chains,
           both from resolve_cocoa_config).
    path = the output path, relative to base or absolute.

  Returns:
    the resolved output path.
  """
  return os.path.join(base, path)
