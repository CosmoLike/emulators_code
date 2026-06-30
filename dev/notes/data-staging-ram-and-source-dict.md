---
name: data-staging-ram-and-source-dict
description: "RAM-efficient data loading for the emulator: memmap the full dv dump (never load it whole), then stage_source materializes the used subset into RAM if it fits 70% of psutil.available, reindexing locally (idx->arange) so the rest of the pipeline is unchanged. The source dict carries C/dv/idx plus the training centers C_mean/dv_mean, so the geometry reads them and you carry only train_set. References, not deep copies."
metadata: 
  node_type: memory
  type: project
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

How the emulator ([[emulator-pipeline-and-goal]]) loads data RAM-efficiently,
adopted this session for scaling toward millions of dvs.

**Memmap the full dump.** `dv0 = np.load(path, mmap_mode="r", allow_pickle=False)`
keeps the big dv array on disk -- it never enters RAM whole. The params table is
tiny (~12 cols vs ~3000), so `np.loadtxt(path, dtype="float32")[:, 2:-1]` straight
to RAM is fine. The physical cut ([[emulator-floor-is-data-coverage]],
phys_cut_idx on omega_b h^2) is params-only, so it never touches the dv.

**Stage the used subset to RAM if it fits (host-side regime gate).** A helper
`stage_source(C, dv, idx, ram_frac=0.7)`: `rows = sort(unique(idx))`;
`nbytes = rows.size * dv.shape[1] * dv.dtype.itemsize`; if
`nbytes < ram_frac * psutil.virtual_memory().available`, materialize the COMPACT
subset (`np.asarray(C[rows])`, `np.asarray(dv[rows])`) and reindex locally
(`idx_src = np.arange(rows.size)`); else return `(C, dv, idx)` unchanged (dv
still the memmap, idx still global). The training cut is ~1/10 and val ~1/50, so
both usually fit even when the full dump does not. The full memmap is dropped
after staging, so RAM holds only the subset (one copy).

**Local reindex is the trick that keeps the pipeline unchanged.** The loaders,
geometry, eval, and analysis only ever touch C/dv THROUGH idx
(`dv[used_rows]`, `C[rows]`, `used_rows = unique(idx)`). So whether idx is global
(into the full memmap) or local arange (into the compact subset), every consumer
works untouched -- no loader change needed for staging. The invariant: idx
matches its own C/dv.

**Source dict carries the centers.** `train_set = {"C", "dv", "idx", "C_mean",
"dv_mean"}` -- the training-mean centers (dvt_off via stream_stats over the
memmap/subset in chunks; c_off via param_stats) computed once and stored, so the
geometry build reads `train_set["dv_mean"]` / `train_set["C_mean"]` instead of
loose globals (`chi2fn = CosmolikeChi2.from_cosmolike(device,
train_set["dv_mean"], probe="xi")`; `pgeom =
ParamGeometry.from_covmat(device, train_set["C_mean"], cov_path)`). val_set has
no means (geometry is training-only). Bug fixed in passing: the draft computed
the means from `train_set[...]` before train_set was defined -- compute from the
local arrays.

**Why deep copies are wrong here.** The user asked if carrying only train_set
needs deep copies. No: a dict entry is a REFERENCE, not a copy; the array stays
alive while train_set references it, even after `del C0`. Deep-copying would just
double RAM for no benefit (arrays are never mutated in place). So reference + an
optional `del` of the loose names is the efficient, self-contained answer; the
memmap makes dv's reference disk-backed.

**Why:** records the RAM design (memmap + stage_source + local reindex + centers
in the source dict) so the next session does not reinvent it or reintroduce the
full-dump-in-RAM waste; pairs with the [[pytorch-teaching-style]] performance
"regime ladder" (this is its host-side RAM gate).
