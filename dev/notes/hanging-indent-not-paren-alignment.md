---
name: hanging-indent-not-paren-alignment
description: "For multiline calls/signatures, use a 2-space hanging indent; never align continuation lines under the opening paren."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

When a call or function signature wraps across lines in the teaching notebook,
break after the opening `(` and indent the continuation lines by **2 spaces**
(a hanging indent). Do **not** align the arguments under the opening
parenthesis.

**SCOPE (2026-06-29): this is the slide-NOTEBOOK rule.** The ported .py package
(`emulator/`) and the CLI drivers (`driver/`) use **paren-alignment** instead
(one argument per line aligned under the opening paren) at a 90-col budget -- the
user's choice for real code. See [[py-module-style-conventions]].

```python
# preferred
load_C, load_dv, load = build_loaders(
  device=device,
  C0=C0,
  ...)

# avoid (aligned under the paren)
load_C, load_dv, load = build_loaders(device,
                                      C0,
                                      ...)
```

**Why:** aligning under the `(` pushes the arguments far to the right, eating
into the ~60-char slide width budget and overflowing the slide; a 2-space
hanging indent keeps everything near the left margin.

**How to apply:** applies to multiline function calls and `def` signatures
alike. Same 2-space rule the [[weight-decay-only-on-weight-matrices]] examples
already use. Captured in the [[pytorch-teaching-style]] skill (Code width).
