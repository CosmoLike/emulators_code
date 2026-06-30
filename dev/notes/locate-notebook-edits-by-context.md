---
name: locate-notebook-edits-by-context
description: "When flagging notebook issues/edits, locate them by context (function/class name, a quoted nearby line, a marker comment) — never by cell number or my internal line numbers."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

The user works in Jupyter and does **not** see cell numbers (and never sees my
internal extraction line numbers). When pointing to an issue, a suggested edit,
or a place to paste, identify the spot by **context the user can actually see**:
the enclosing function/class name, a quoted nearby line, or a marker comment to
search for (e.g. "the `device = next(model.parameters()).device` line right
above the `# MPS ... has no float64` comment").

**Why:** "cell 258" / "line 1070" reference my own dump of the notebook, which
the user cannot map back to their view, so the guidance is unactionable.

**How to apply:** in reviews and fixes, anchor every finding to a searchable
landmark in the code, not a position. Pairs with how I extract notebook cells
to /tmp for review — that numbering is mine alone. Related: [[hanging-indent-not-paren-alignment]].
