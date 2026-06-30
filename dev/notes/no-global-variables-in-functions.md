---
name: no-global-variables-in-functions
description: "Functions must take everything as parameters and read no module-level data global (device, normalization stats, a fitted geometry, config, dataset arrays); imports and helper defs are fine. A global read is an extreme exception that must be flagged in-code with an emoji + WARNING comment, never silent. Audit free names with symtable."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

The user requires that functions take what they need as parameters and read no
module-level data global. Imports (`np`, `torch`) and module-level helper
functions/classes are not "globals" here -- the rule is about data and state:
`device`, normalization stats, a fitted geometry (`param_geom` / `pgeom`), a
config object, dataset arrays. A global read is an extreme exception: when a
value genuinely cannot be threaded through, the read must be flagged in-code on
that line with an emoji and the word WARNING naming the global, e.g.
`# ⚠️ WARNING: reads module global device (not a param here)`. A silent global
read is unacceptable -- a latent bug and a review red flag. WARNING is the one
sanctioned all-caps token (a flag label, not emphasis).

**Why:** a silent global read hides the function's real inputs and breaks when
the function is reused, moved, or called before the global exists. The rename
hazard is the worst case: rename a definition (`param_geom` -> `pgeom`) and a
caller still passing `param_geometry=param_geom` either raises NameError or
silently binds a stale value left in the kernel from an earlier run. Flagging
the rare exception and otherwise passing parameters removes both failure modes.

**How to apply:** audit each function's free names with a `symtable` pass over
the cells -- they should be only parameters, locals, imports, and helper defs;
filter those out and what remains is the data-global leak set. Flag any
unavoidable global with the emoji + WARNING marker. In the emulator notebook
this caught `run_emulator` still reading the global `device` (the only
loader-chain function not taking it as a param -- `_build_loaders_one` and
`build_loaders` both do) and a stale `param_geom` in the batch-sweep call after
the def was renamed `pgeom`.

Pairs with [[locate-notebook-edits-by-context]], [[probe-generalization-bugs]],
[[shared-budget-across-sequential-calls]]; captured in the
[[pytorch-teaching-style]] skill (No global variables in functions + checklist).
