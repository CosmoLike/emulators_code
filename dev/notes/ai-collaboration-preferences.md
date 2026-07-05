---
name: ai-collaboration-preferences
description: "Portable in-repo mirror of the durable AI-collaboration preferences for this project (written 2026-07-05 ahead of moving dev/ into its own repo, because the ~/.claude auto-memory is keyed to the OLD absolute path and does NOT travel with the files). Two roles: an INDEX pointing at the notes that already hold each preference (code style -> py-module-style-conventions.md; dual-agent workflow + audit-is-Fable -> dual-fable-opus-workflow.md; shape-flow diagrams -> py-module-style-conventions.md; plots -> plots-no-red-green.md; machines -> dev-machine-mac-m2-32gb.md + test-workstation-gpus.md), PLUS the FULL CAPTURE of the two preferences that lived only in auto-memory and were nowhere else in notes/: (1) YAML in BLOCK style (nested keys one per line, never inline {...} flow); (2) the REPO-NOTES RITUAL (update notes/ design-spec blocks + resume state + MEMORY.md index at every milestone, unprompted; Vivian checks). After the repo move, this file is the read-me-first for how Claude collaborates with Vivian here."
metadata:
  node_type: memory
  type: feedback
---

Durable working preferences for how Claude collaborates with Vivian on this
project — a **portable, in-repo** copy. Written 2026-07-05 ahead of moving
`dev/` into its own git repo.

**Why this file exists.** The `~/.claude` auto-memory is keyed to the project's
absolute path (`.../emulators_code`); when `dev/` becomes its own repo at a new
path, that memory does NOT follow. The repo's `notes/` travels with the files,
so the durable preferences are mirrored here. Most already live in other notes
(indexed below); the two that lived ONLY in auto-memory are captured in full at
the bottom.

## Index — preferences already recorded elsewhere in notes/

- **Python code style** (`emulator/` package + drivers) →
  [[py-module-style-conventions]]. Named parameters everywhere the callee
  allows; irreducibly-positional args stay positional + a naming comment;
  paren-alignment one item per line at 90 cols; didactic comments on
  shape ops; prose module docstrings with a `PS:` jargon glossary; formal
  `Arguments:` docstring blocks; no comprehensions in cold code (explicit
  C-style loops); README in cocoa style (nested TOC + per-file appendices +
  ASCII flow diagrams).
- **Shape-flow diagrams in docstrings** →
  [[py-module-style-conventions]] (the 2026-07-04 block). Vertical `│`/`▼`
  tensor-pipeline graphs; EVERY symbol defined in a trailing `(legend: ...)`;
  no magic numbers without a named-symbol derivation. ("I love this SO MUCH.")
- **Dual-agent workflow + AUDIT is Fable's domain** →
  [[dual-fable-opus-workflow]]. Fable 5 = Architect/Auditor, Opus 4.8 =
  Implementer; handoff blocks persist to `notes/` first; the audit never moves
  off Fable.
- **Plots: no red+green** → [[plots-no-red-green]]. Colorblind-safe palette,
  explicit per-line colors, never the default cycle.
- **Machines** → [[dev-machine-mac-m2-32gb]] (Mac M2/MPS, dev) +
  [[test-workstation-gpus]] (NVIDIA test rig). Code branches on `device.type`.

## Captured in full (lived only in auto-memory)

### 1. YAML in block style, never inline flow

Write YAML config in **block style** — nested keys indented one per line — and
never inline `{...}` flow style ("noticed how I prefer to write the keys, using
tabs and not {}", 2026-07-04). Configs are read and hand-edited constantly; one
key per line diffs cleanly and scans vertically. Align values in a column when
the surrounding file does.

```yaml
trim:
  start:         0.03
  end:           0.01
  hold_epochs:   15
```

Never `trim: {start: 0.03, end: 0.01, hold_epochs: 15}`. Applies to
`example_yamls/` and any YAML in chat or docs; keep new additions in block
style.

### 2. Repo-notes update ritual (every milestone, unprompted)

Update the project knowledge base — this `notes/` folder (dated design-spec
blocks, the resume-state addenda, the `notes/MEMORY.md` index lines) — as part
of finishing any milestone, NOT as a separate step Vivian has to request. Vivian
verifies ("update memories then we compact"; "did you update memory files?").

The notes are the cross-session continuity mechanism for a months-long project;
a milestone that isn't recorded there is, to Vivian, unfinished work. When
closing a feature/build/doc task: append the dated design-spec block, refresh
the resume state, and update the touched `notes/MEMORY.md` index line in the
**same turn** as the code — then say so in the summary. Format rules those notes
follow: [[py-module-style-conventions]] (shape-flow diagrams) and the YAML block
style above.
