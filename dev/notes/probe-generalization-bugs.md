---
name: probe-generalization-bugs
description: "When reviewing the emulator, always hunt for cosmic-shear/xi-only assumptions that silently break for ggl/wtheta/3x2pt -- block-local vs global indices, hardcoded full-dv length, probe-name reuse, cosmolike full-3x2pt array assumptions."
metadata:
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

The user wants the cosmic-shear emulator ([[emulator-pipeline-and-goal]]) to
extend to ggl and w(theta), not just xi. On every notebook review, actively
look for bugs that work for cosmic shear but break for the other probes.

**The canonical trap:** xi is block 0 of the 3x2pt vector, starting at global
position 0, so block-local indices equal global indices and many things "work
by coincidence." Anything correct only because `block_start == 0` is a latent
ggl/wtheta bug (ggl/wtheta start partway into the full vector).

**Checklist of where these hide (found in this codebase):**
- Block-local vs global indices. Index the full data vector / full-vector mean
  by the GLOBAL `dest_idx`, never the block-local `kept_cols`
  (`squeeze`, `center`). For xi they're equal; for ggl/wtheta they differ.
- Dataset must be full 3x2pt. `dv0` and its column-mean must span the whole
  1560-long 3x2pt vector (it does here: dv0 is (100000, 1560)), so global
  `dest_idx` lands on real data. A cosmic-shear-only file would mis-index ggl.
  Add `assert dv0.shape[1] == chi2fn.total_size` to fail loudly.
- cosmolike boundary. `get_mask` / `get_cov_masked` / `get_inv_cov_masked` /
  `compute_data_vector_3x2pt_real_sizes` are assumed full-3x2pt-length even
  under a single-probe `init_probes`; xi survives a block-only return, ggl/
  wtheta do not. Verify on the workstation.
- Probe-name reuse. The same string is both a `PROBE_BLOCKS` key and the
  `ci.init_probes(possible_probes=probe)` argument; "xi" matches but cosmolike
  may call galaxy-galaxy lensing "gammat", etc. Decouple key from cosmolike name.
- Hardcoded full-dv length. `dv_len=2000` sizes the resident full `Cinv`
  (dv_len^2) and chi2 buffers; derive it from `total_size`, not a literal
  (conservative for Y1's 1560, under-budgets a larger 3x2pt).

**Why:** the user explicitly asked to always check for this class of bug while
the emulator is xi-only but destined for ggl/w(theta).

**How to apply:** during any emulator review, trace each probe-dependent
quantity for "is this only right because xi starts at 0 / because the file is
cosmic-shear?" Flag block-local indexing, dataset width assumptions, cosmolike
array-length assumptions, and probe-name reuse. Pairs with
[[locate-notebook-edits-by-context]].

Captured in the [[pytorch-teaching-style]] skill too: the DataVectorGeometry case
study now indexes by the global dest_idx (not block-local keep_local), with a
"probe generalization is a review axis" decision bullet and a clause in "Review
before a long run".
