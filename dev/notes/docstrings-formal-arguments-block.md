---
name: docstrings-formal-arguments-block
description: "Every function/method docstring must use a formal Arguments: block documenting ALL parameters (name = description) — never prose that covers only some."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

When writing a docstring for any function or method in the teaching notebook,
always include a formal **`Arguments:`** block that lists **every** parameter as
`name = description`, aligned, one entry per parameter — the same style used for
`run_emulator`, `plot_history`, `stream_stats`, `DataVectorGeometry.__init__`,
etc. A short prose overview can precede it, but prose is **not** a substitute:
do not describe only the "interesting" arguments (e.g. a spec dict) and leave
the injected ones (input_dim, device, ...) mentioned only in passing. Add a
`Returns:` block too.

**Why:** the user has asked for this repeatedly ("formal explanation of all
input", "list all arguments"); a partial/prose docstring reads as incomplete and
they have to ask again. Helper functions (make_model, make_optimizer) are NOT
exempt.

**How to apply:** for every `def`, enumerate all parameters in an `Arguments:`
block; note which are injected vs. user-supplied in their descriptions rather
than omitting them. Pairs with [[locate-notebook-edits-by-context]] and the
[[pytorch-teaching-style]] house style.
