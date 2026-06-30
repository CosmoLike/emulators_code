---
name: cocoa-rootdir-env
description: "Where Cocoa's ROOTDIR env var comes from (the anchor our driver/ scripts use as os.environ['ROOTDIR']). It is NOT in the README: it is exported in Cocoa/set_installation_options.sh as ROOTDIR=$(pwd -P) (Darwin branch pipes through sed 's:^//:/:' to fix the macOS // prefix), i.e. the absolute physical path of the Cocoa/ directory at the moment the script is sourced. set_installation_options.sh is sourced by both setup_cocoa.sh and start_cocoa.sh; you re-source start_cocoa.sh every session (from inside cocoa/Cocoa/) to activate the .local/ venv + conda env and re-establish ROOTDIR. Because the scripts are SOURCED not executed, the export lands in the interactive shell, so a Python driver later reads it via os.environ['ROOTDIR']. Everything hangs off it: ${ROOTDIR:?}/.local, /projects, /external_modules/code (CAMB, cosmolike, emulators/emultrf), /external_modules/data, the clik/Planck paths, cobaya symlinks. The :? form aborts loudly if unset. Workflow: cd cocoa/Cocoa; source setup_cocoa.sh; source compile_cocoa.sh; source start_cocoa.sh."
metadata:
  node_type: memory
  type: reference
---

Where Cocoa's **ROOTDIR** comes from -- the absolute path anchor our `driver/`
scripts use as `os.environ["ROOTDIR"]` to build project/data paths
([[notebook-to-python-translation]], [[emulator-python-package]]).

**It is NOT in the README.** ROOTDIR is exported in
**`Cocoa/set_installation_options.sh`**:

    # DEFINING ROOTDIR (DO NOT CHANGE)
    case "$(uname -s)" in
      Linux)  export ROOTDIR=$(pwd -P) ;;
      Darwin) export ROOTDIR=$(pwd -P | sed 's:^//:/:') ;;   # fix macOS // prefix
    esac

So ROOTDIR = the **absolute physical path of the `Cocoa/` directory** at the
moment the script is sourced (`pwd -P` resolves symlinks; the Darwin branch
strips a leading `//`).

**How it reaches Python.** `set_installation_options.sh` is sourced by both
`setup_cocoa.sh` and `start_cocoa.sh`. The three scripts are run from inside
`cocoa/Cocoa/` and are SOURCED, not executed, so the `export` lands in the
interactive shell; a driver launched in that shell then reads
`os.environ["ROOTDIR"]`. A driver that reads it with no Cocoa env sourced fails
with KeyError -- the intended guard.

    cd cocoa/Cocoa
    source setup_cocoa.sh     # clone/build external_modules; sources set_installation_options.sh
    source compile_cocoa.sh   # compile CAMB, cosmolike, ...
    source start_cocoa.sh     # re-source every session: activate .local/ venv + conda, re-export ROOTDIR

**What it anchors** (all referenced as `${ROOTDIR:?}/...`; the `:?` aborts loudly
if unset): `.local/` (venv + locally-installed libs/bins), `projects/` (analysis
projects), `external_modules/code/` (CAMB, cosmolike, emulators/emultrf),
`external_modules/data/`, the clik/Planck `CLIK_*` paths, and the cobaya
likelihood/theory symlinks.

**Why:** the driver style ([[notebook-to-python-translation]]) builds every path
ROOTDIR-relative; this records the external fact (outside the repo I can see) that
ROOTDIR = `$(pwd -P)` of `Cocoa/`, set by sourcing `start_cocoa.sh`, so I do not
re-hunt for it. Pairs with [[emulator-python-package]] (the drivers).
