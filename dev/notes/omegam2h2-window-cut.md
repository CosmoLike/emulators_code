---
name: omegam2h2-window-cut
description: "2026-07-03, first T=256 run (resmlp, N_train 250k, frac>0.2 = 0.59, median 0.30): a SECOND physical cut was added and implemented -- a two-sided window 0.015 < omegam^2 h^2 < 0.08 (= Gamma^2, Gamma = Omega_m h the transfer shape parameter; Planck Gamma^2 = 0.045). EVIDENCE (forensic extraction of the diagnostic PDF's vector scatter, ~2000 val points with positions + viridis-inverted chi2): (1) hardness direction fit log10 dchi2 ~ 0.66 ln(Om) + 1.47 ln(h), b/a = 2.2 -> iso-hardness tracks Om h^2, NOT the product; BUT (2) the user's coverage argument wins the framing: the divisor draws N_train from the CUT pool, so cutting rarefied volume densifies rather than costs data, and BOTH local sparsity and failure rate are U-SHAPED along omegam^2 h^2 (core 0.02-0.08: frac>10 ~ 0.00-0.02; tails <0.01 / >0.2: sparsity 2.5x and frac>10 up to 0.85) -- the low-omh2 and high-omh2 failure clusters are the two tips of ONE needle coordinate. Window (0.015, 0.08): keeps ~61% of T=128 val, removes 96% of dchi2>100, frac>10 among kept 0.012. IMPLEMENTED: phys_cut_idx(omegam2h2_lo/hi) in data_staging.py, threaded through load_source + EmulatorExperiment (stage_train/stage_val/pool_size), YAML keys omegam2h2_lo/hi in the data block (absent = no cut, backward compatible). WARNING: keep = n_fullfile // divisor is computed PRE-cut, so the window can trip 'physical pool too small' -- cs_128 val with val_divisor 2 (12.5k) is right at the edge; use val_divisor 3+ or n_keep."
metadata:
  node_type: memory
  type: project
---

First T=256 production run (2026-07-03, resmlp width 128, N_train 250k of the
cs_256 dump, val cs_128/2): best frac>0.2 = 0.59, median 0.30, catastrophic
tail to dchi2 ~ 1e8. Diagnostics: coverage-limited (sparse-region failures),
hardness R^2 low, omega_b h^2 NOT the driver (matches the T=16 story).

**THE LIVE CUTS (user-confirmed 2026-07-03, their production YAML):**

    omegabh2_cut: 0.035    # omega_b h^2 upper bound
    omegabh2_lo:  0.005    # omega_b h^2 lower bound (user chose 0.005,
                           # looser than the suggested 0.014)
    omegam2h2_lo: 0.015    # Gamma^2 window
    omegam2h2_hi: 0.08

**The new cut (user's call, implemented): 0.015 < omegam^2 h^2 < 0.08.**
omegam^2 h^2 = (Omega_m H0/100)^2 = Gamma^2 with Gamma = Omega_m h (the
transfer shape parameter); Planck sits at 0.045, dead center.

**How the decision went (both halves matter):**
- I fit the hardness direction on ~2000 val points extracted from the
  diagnostic PDF's vector graphics (positions + viridis-inverted colors;
  the wedge identity Om h^2 = Om (H0/100)^2 validated the extraction at
  99.9%): log10 dchi2 ~ 0.66 ln Om + 1.47 ln h, exponent ratio 2.2 ->
  iso-hardness is essentially Om h^2 (near-horizontal in the panel), NOT
  the -45-degree product. A failure-targeted cut would be Om h^2 < 0.25.
- The user reframed it as a COVERAGE cut and won: the divisor draws
  N_train from the cut pool, so cutting rarefied volume DENSIFIES the
  kept region instead of costing data. Local kNN sparsity tracks
  |ln omegam^2h^2 - median| at Spearman +0.58 (beats om h^2's +0.49),
  and BOTH sparsity and failure rate are U-shaped along the product:
  core (0.02-0.08) frac>10 ~ 0.000-0.023; tails (<0.01, >0.2) sparsity
  ~2.5x and frac>10 up to 0.85. The low-omh2 and high-omh2 failure
  clusters are the two tips of one needle coordinate (the sampling
  cloud's long axis in (ln Om, ln h)) -- one variable, one story.
- Window (0.015, 0.08) on T=128 val: keeps 61%, removes 96% of
  dchi2>100, frac>10 among kept 0.012 (vs 0.113 uncut).

**Implementation (2026-07-03):** `phys_cut_idx(C, idx, names, cut,
omegam2h2_lo=None, omegam2h2_hi=None)` in data_staging.py (None = that
side uncut -> old configs unchanged); threaded through `load_source` and
EmulatorExperiment's stage_train / stage_val / pool_size via
`data.get("omegam2h2_lo"/"omegam2h2_hi")`; YAML keys documented in the
example yamls, driver header, experiment docstring, README. Unit-tested
(verbatim extraction; backcompat + joint mask + order + one-sided).

**WARNING (the pool guard):** load_source computes keep = n_fullfile //
divisor BEFORE the cut and raises "physical pool too small" if the cut
pool cannot supply it. cs_128 val with val_divisor 2 (keep 12.5k) is
right at the edge once the window is on (obh2 ~0.81 x window ~0.61 of
25k ~ 12.3k) -- use val_divisor 3+ (or n_keep) with the window active.

**OUTCOME (same day, 1500 epochs, bs 768, 17 min total at 0.68 s/epoch):
frac>0.2 = 0.156, median = 0.053** (vs 0.59 / 0.30 uncut, 50 min) -- best
epoch 1497, ladder 0.5: 0.058, 1: 0.028, 10: 0.005, 100: 0.002; val mean
plateaus at ~22 (owned by the residual 0.2% tail). Post-cut diagnostics:
(1) the residual failures HUG THE CUT BOUNDARIES in the triangle (one-sided
training support at the window edge), plus a diffuse scatter; (2) the
hard-direction joint R^2 fell 0.35 -> 0.18 (hardness now mostly diffuse;
leading combo again As * omegam * H0 = structure amplitude, omegab
irrelevant); (3) LSST_A1_1 (IA amplitude) now 5th in the univariate ranking
-- first hint the wide-prior IA direction starts costing at T=256 (the
[[npce-and-ia-template-factoring]] lever's regime). Only 250k of the ~1M
post-cut pool was used -> the N_train sweep is the obvious next lever;
after that, a train-wider-grade-narrower buffer strip would address the
boundary-hugging (needs separate train/val window keys, not yet built).

**Why:** records the cut's evidence and the framing lesson (failure-cut
vs coverage-cut: with divisor-drawn N_train, cutting no-man's-land is
densification, not data loss), so future probes (ggl/wtheta) and dataset
generations reuse the window rather than re-deriving it. Pairs with
[[emulator-floor-is-data-coverage]] (the omegabh2 precedent) and
[[emulator-python-package]].
