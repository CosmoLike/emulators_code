"""Training-history, learning-curve, coverage, and xi plots.

The matplotlib figures (a colorblind-safe palette, no red/green).
plot_history draws the training history, plot_diagnostics the multipage
diagnostics PDF (history, coverage, the local-linear floor, the
hard-direction regression, a chi2-colored LCDM triangle, and the
ln-parameter PCA plane colored by chi2 and by training sparsity), and
plot_learning_curves overlays
f(delta-chi2 > thr) vs N_train curves (the sweep / bake-off output).
source_param_samples, dv_to_xi, and plot_xi handle the parameter-coverage
triangle and the xi correlation-function curves. The "_"-prefixed helpers
draw the individual panels the public functions share.
"""

import itertools
import warnings
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from getdist import MCSamples
from getdist import plots as gdplots

# colorblind-safe palette, no red/green (Wong 2011 minus its green
# and vermillion): blue, orange, reddish-purple, black, sky-blue.
_CB = ["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"]


def _finish(fig, savepath):
  """Save the figure and close, or show it.

  If savepath is given, write the figure there (format from the
  extension, e.g. .pdf) and close it -- a batch script has no
  display; if None, show it interactively.
  """
  if savepath is not None:
    fig.savefig(savepath, bbox_inches="tight")
    plt.close(fig)
  else:
    plt.show()


def _history_panels(ax_loss, ax_frac, train_losses, medians,
                    means, fracs, thresholds):
  """
  Draw the two training-history panels.

  ax_loss: train loss / val median / val mean vs epoch (log y, as
  the mean is heavy-tailed far above the median). ax_frac: fraction
  of val points over each delta-chi2 threshold vs epoch. Shared by
  plot_history (1x2) and plot_diagnostics (2x2) so the two never
  drift apart.

  Arguments:
    ax_loss, ax_frac = the two axes to draw on.
    train_losses     = per-epoch training loss (list).
    medians / means  = per-epoch val median / mean chi2 (lists).
    fracs            = per-epoch list of 1D tensors; fracs[i] is
                       the fraction over each threshold at epoch i+1.
    thresholds       = 1D tensor of delta-chi2 cutoffs (labels).
  """
  epochs = range(1, len(medians) + 1)
  # x = epoch, y = each per-epoch curve: train loss, val median,
  # val mean.
  ax_loss.semilogy(epochs,
                   train_losses,
                   color=_CB[0],
                   label="train")
  ax_loss.semilogy(epochs,
                   medians,
                   color=_CB[1],
                   label="val median")
  ax_loss.semilogy(epochs,
                   means,
                   color=_CB[2],
                   label="val mean")
  ax_loss.set_xlabel("epoch")
  ax_loss.set_ylabel("loss")
  ax_loss.legend(frameon=False)

  fr = torch.stack(fracs).cpu()      # (nepochs, n_thr)
  for j, t in enumerate(thresholds.tolist()):
    # x = epochs, y = fraction of val points over threshold j.
    ax_frac.plot(epochs,
                 fr[:, j],
                 color=_CB[j % len(_CB)],
                 label=f"> {t:g}")
  ax_frac.set_xlabel("epoch")
  ax_frac.set_ylabel("fraction of val points")
  ax_frac.legend(frameon=False, title="delta chi2")


# the chi2 color band in log10: floor and saturation shared by every
# chi2-colored scatter, and the pinned colorbar limits (pinning keeps
# the colorbars identical across runs and pages, so shades compare).
_CHI2_CBAND = (-2.0, 1.5)


def _log_dchi2_color(dchi2, lo=_CHI2_CBAND[0], hi=_CHI2_CBAND[1]):
  """
  log10 delta-chi2 clipped to a fixed color band.

  The raw chi2 spans many decades and a handful of catastrophic
  outliers would set the color scale, washing every ordinary point
  into one dark shade. Clipping to [lo, hi] (default delta-chi2 in
  [0.01, ~30]) spends the whole colormap on the decision-relevant
  band around the 0.2 goal; anything beyond saturates at the end
  color. The floor also guards log10 against a numerically zero
  chi2.

  Arguments:
    dchi2 = per-point delta-chi2 values.
    lo    = lower color bound in log10 (default -2, chi2 = 0.01).
    hi    = upper color bound in log10 (default +1.5, chi2 ~ 30).

  Returns:
    (N,) clipped log10 values, ready to be a scatter color.
  """
  c = np.asarray(dchi2, dtype="float64")
  return np.clip(np.log10(np.maximum(c, 1e-12)), lo, hi)


def _coverage_panels(ax_scatter, ax_hist, knn_dist, dchi2, k_nn):
  """
  Draw the two coverage-diagnostic panels.

  ax_scatter: per-val hardness log10(dchi2) vs local sparsity (mean
  distance to the k nearest training points), with the 0.2 goal
  line. ax_hist: sparsity distributions of the good (dchi2<=0.2) and
  bad (dchi2>0.2) populations -- a right-shifted "bad" histogram
  means failures live where training is scarce.

  Arguments:
    ax_scatter, ax_hist = the two axes to draw on.
    knn_dist = (Nval,) mean distance to the k nearest train points.
    dchi2    = (Nval,) per-val delta-chi2 (same row order).
    k_nn     = the k used (for the axis labels).
  """
  y   = np.log10(np.maximum(dchi2, 1e-4))
  bad = dchi2 > 0.2

  # (a) hardness vs local sparsity. x = knn_dist, y = log10 dchi2
  # (full range, so outliers stay visible as points); color = the
  # clipped log10 (saturated ends, so the bulk keeps color
  # resolution). Points draw dark-to-bright so the rare hard ones
  # sit on top of the overplotted bulk instead of being buried.
  # The dashed line is the 0.2 goal.
  c = _log_dchi2_color(dchi2)
  order = np.argsort(c)
  sc = ax_scatter.scatter(knn_dist[order], y[order], s=5,
                          c=c[order],
                          cmap="viridis",
                          vmin=_CHI2_CBAND[0],
                          vmax=_CHI2_CBAND[1])
  ax_scatter.axhline(np.log10(0.2), color="0.4", lw=1, ls="--")
  ax_scatter.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_scatter.set_ylabel(r"$\log_{10}\,\Delta\chi^2$")
  # ax.figure is the parent figure; add the colorbar to it. The
  # extend arrows mark that values continue past the clipped ends.
  ax_scatter.figure.colorbar(sc,
                             ax=ax_scatter,
                             extend="both",
                             label=r"$\log_{10}\,\Delta\chi^2$")

  # (b) good vs bad sparsity. x = knn_dist, y = density; shared
  # bins so the two histograms are comparable. The range clips the
  # far knn tails (a lone outlier stretches equal-width bins until
  # the narrow population falls into a single bar), and the bin
  # width follows the NARROWER population (Freedman-Diaconis per
  # population, take the smaller), so a tight "good" peak still
  # shows its shape next to a broad "bad" one.
  lo = np.percentile(knn_dist, 0.5)
  hi = np.percentile(knn_dist, 99.5)
  widths = []
  for m in (~bad, bad):
    if m.sum() >= 2:
      e = np.histogram_bin_edges(knn_dist[m], bins="fd")
      if len(e) > 1 and e[1] > e[0]:
        widths.append(e[1] - e[0])
  width = min(widths) if widths else (hi - lo) / 40.0
  nbins = int(np.ceil((hi - lo) / max(width, 1e-12)))
  nbins = min(max(nbins, 20), 150)   # readable floor / ceiling
  bins = np.linspace(lo, hi, nbins + 1)
  ax_hist.hist(knn_dist[~bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[0],
               label="good (dchi2<0.2)")
  ax_hist.hist(knn_dist[bad],
               bins=bins,
               density=True,
               alpha=0.6,
               color=_CB[1],
               label="bad (dchi2>0.2)")
  ax_hist.set_xlabel(f"mean dist to {k_nn} nearest train pts")
  ax_hist.set_ylabel("density")
  ax_hist.legend(frameon=False)


def plot_history(train_losses,
                 medians,
                 means,
                 fracs,
                 thresholds,
                 savepath=None):
  """
  Plot a run_emulator training history (the two history panels).

  Left: train loss, val median, val mean vs epoch (log y). Right:
  fraction of val points over each delta-chi2 threshold vs epoch.

  Arguments (the four run_emulator histories, plus the thresholds):
    train_losses = per-epoch training loss (list of floats); the
                   sqrt-trimmed objective, on a different scale than
                   the raw-chi2 val metrics.
    medians      = per-epoch val median chi2 (list).
    means        = per-epoch val mean chi2 (list).
    fracs        = per-epoch list of 1D tensors; fracs[i] holds the
                   fraction of val points over each threshold at
                   epoch i+1.
    thresholds   = 1D tensor of delta-chi2 cutoffs used in
                   training; labels the right panel.
    savepath     = if given, write the figure there and close; if
                   None (default), show it interactively.
  """
  fig, ax = plt.subplots(1, 2, figsize=(11, 4))
  _history_panels(ax[0], ax[1], train_losses, medians, means,
                  fracs, thresholds)
  fig.tight_layout()
  _finish(fig, savepath)


def plot_learning_curves(curves,
                         threshold=0.2,
                         target=0.10,
                         savepath=None):
  """
  Overlay one or more learning curves: f(delta-chi2 > threshold) vs
  N_train.

  One descending curve per entry, on log-log axes (spreading out the
  small-N regime where methods separate). A curve still falling at the
  largest N is data-limited (more data helps); a flat tail is
  capacity / architecture-limited. Lines use a colorblind palette + a
  marker cycle (no red/green); a single-config sweep passes a
  one-entry dict, a bake-off one entry per variant.

  Arguments:
    curves    = mapping label -> the curve, where the curve is either a
                {N_train: frac} dict or an (sizes, fracs) pair (both
                sorted by N here). label is the legend text.
    threshold = the delta-chi2 cutoff the fraction counts (default 0.2,
                the emulator goal); labels the y axis.
    target    = a horizontal guide at the target fraction (default 0.10);
                None to omit it.
    savepath  = if given, write the figure there and close; else show.
  """
  # marker cycle so overlaid curves stay distinguishable in print.
  markers = ["o", "D", "^", "s", "v", "P"]
  fig, ax = plt.subplots(figsize=(6.8, 5.6))

  for k, (label, curve) in enumerate(curves.items()):
    # accept {N: frac} or a (sizes, fracs) pair.
    if isinstance(curve, dict):
      keys  = sorted(curve)
      sizes = np.array(keys, dtype="float64")
      fvals = []
      for n in keys:
        fvals.append(curve[n])
      fracs = np.array(fvals, dtype="float64")
    else:
      sizes = np.asarray(curve[0], dtype="float64")
      fracs = np.asarray(curve[1], dtype="float64")
      order = np.argsort(sizes)            # plot left-to-right in N
      sizes, fracs = sizes[order], fracs[order]
    # x = N_train, y = fraction over the threshold.
    ax.plot(sizes,
            fracs,
            "-" + markers[k % len(markers)],
            color=_CB[k % len(_CB)],
            lw=2.5,
            ms=8,
            label=label)

  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel(r"$N_{\rm train}$")
  ax.set_ylabel(rf"$f(\Delta\chi^2 > {threshold:g})$")
  if target is not None:
    ax.axhline(target, color="0.6", ls="--", lw=1,
               label=f"target {target:g}")
  ax.legend(frameon=False)
  fig.tight_layout()
  _finish(fig, savepath)


def plot_sweep_curve(param,
                     values,
                     fracs,
                     threshold=0.2,
                     target=0.10,
                     savepath=None):
  """
  One-hyperparameter sweep figure: f(delta-chi2 > threshold) vs the
  swept values.

  The generic twin of plot_learning_curves for an arbitrary knob.
  Numeric values draw a connected curve (log x when the values are
  all positive and span more than a factor 20, so a log-spaced lr
  grid reads evenly); categorical values (activation names, a
  film True/False pair) draw one marker per value with the labels
  as x ticks. y is logarithmic when every fraction is positive,
  linear otherwise (a perfect 0.0 point would break a log axis).

  Arguments:
    param     = the swept hyperparameter's dotted YAML path (the x
                label).
    values    = the swept values, in sweep order (numbers, strings,
                or booleans).
    fracs     = per-value fractions aligned with `values`.
    threshold = the delta-chi2 cutoff the fraction counts (labels
                the y axis).
    target    = a horizontal guide at the target fraction (default
                0.10); None to omit it.
    savepath  = if given, write the figure there and close; else
                show.
  """
  # booleans are ints in Python; label them instead of plotting 0/1.
  numeric = True
  for v in values:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
      numeric = False
  fr = np.asarray(fracs, dtype="float64")

  fig, ax = plt.subplots(figsize=(6.8, 5.6))
  if numeric:
    xs = np.asarray(values, dtype="float64")
    order = np.argsort(xs)                # draw left-to-right
    # x = the swept values, y = fraction over the threshold.
    ax.plot(xs[order],
            fr[order],
            "-o",
            color=_CB[0],
            lw=2.5,
            ms=8)
    if xs.min() > 0 and xs.max() / xs.min() > 20:
      ax.set_xscale("log")
  else:
    xs = np.arange(len(values))
    # x = value index (ticks carry the labels), y = the fraction.
    ax.plot(xs,
            fr,
            "o",
            color=_CB[0],
            ms=10)
    labels = []
    for v in values:
      labels.append(str(v))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
  if np.all(fr[np.isfinite(fr)] > 0):
    ax.set_yscale("log")
  ax.set_xlabel(param)
  ax.set_ylabel(rf"$f(\Delta\chi^2 > {threshold:g})$")
  if target is not None:
    ax.axhline(target, color="0.6", ls="--", lw=1,
               label=f"target {target:g}")
    ax.legend(frameon=False)
  fig.tight_layout()
  _finish(fig, savepath)


def _floor_panel(ax, floor):
  """
  Draw the local-linear data-floor panel.

  Per val point: the model's delta-chi2 vs the data-only floor (the
  local-linear prediction's delta-chi2), log-log. Points on the
  diagonal mean the net is at what a smooth local method extracts
  from the data (data-limited); points well above mean it has
  headroom. Dotted lines mark the 0.2 goal on each axis.

  Arguments:
    ax    = the axis to draw on.
    floor = the dict local_linear_floor returned (dchi2_floor,
            dchi2_model, f_floor, f_model, f_hard).
  """
  lo = 1e-3
  dchi2_floor = floor["dchi2_floor"]
  dchi2_model = floor["dchi2_model"]
  # x = data-only floor dchi2, y = model dchi2; clip both at lo so
  # a near-zero point stays on the log axes.
  ax.scatter(np.maximum(dchi2_floor, lo),
             np.maximum(dchi2_model, lo),
             s=5, alpha=0.4, color=_CB[0])
  mx = max(dchi2_floor.max(), dchi2_model.max())
  ax.plot([lo, mx], [lo, mx], "k--", lw=1)
  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.axhline(0.2, color="0.6", lw=1, ls=":")
  ax.axvline(0.2, color="0.6", lw=1, ls=":")
  ax.set_xlabel(r"data-only $\Delta\chi^2$ (local-linear floor)")
  ax.set_ylabel(r"model $\Delta\chi^2$")
  ax.set_title(f"f_model {floor['f_model']:.3f}  vs  "
               f"f_floor {floor['f_floor']:.3f}  "
               f"(pure hardness {floor['f_hard']:.3f})")


def _hard_direction_panels(ax_uni, ax_joint, hd):
  """
  Draw the hard-direction regression as two bar charts.

  ax_uni: each feature's univariate |correlation| with log10 dchi2
  (a collinearity-robust ranking). ax_joint: the joint log-linear
  OLS coefficients (the alpha, beta, ... combination). The joint R^2
  and the ln(omega_b h^2)-alone R^2 are in the titles. Both panels
  share the feature order (descending univariate |corr|).

  Arguments:
    ax_uni, ax_joint = the two axes to draw on.
    hd = the dict hard_direction_regression returned (labels,
         univariate, joint_coef, r2, r2_omega).
  """
  labels = hd["labels"]
  uni    = hd["univariate"]
  coef   = hd["joint_coef"]
  # order features by descending univariate |corr|; both panels
  # share this order so the bars line up.
  order = np.argsort(np.abs(uni))[::-1]
  ypos  = np.arange(len(order))
  names = []
  for j in order:
    names.append(labels[j])

  # barh(y, width): y = bar slot, width = value.
  ax_uni.barh(ypos, np.abs(uni)[order], color=_CB[0])
  ax_uni.set_yticks(ypos)
  ax_uni.set_yticklabels(names)
  ax_uni.invert_yaxis()                 # strongest feature at top
  ax_uni.set_xlabel("univariate |corr| with log10 dchi2")
  ax_uni.set_title("univariate ranking")

  ax_joint.barh(ypos, coef[order], color=_CB[1])
  ax_joint.set_yticks(ypos)
  ax_joint.set_yticklabels(names)
  ax_joint.invert_yaxis()
  ax_joint.axvline(0.0, color="0.6", lw=1)
  ax_joint.set_xlabel("joint log-linear coefficient")
  ax_joint.set_title(f"joint R2 {hd['r2']:.3f}  |  "
                     f"ln(omega_b h2) alone {hd['r2_omega']:.3f}")


def _save_pages(figs, savepath):
  """
  Save figures as a multipage PDF, or show them.

  If savepath is given, write every figure as one page of a single
  PDF (matplotlib's PdfPages) and close them -- a batch script has
  no display; if None, show them interactively.

  Arguments:
    figs     = list of matplotlib Figures, one per page.
    savepath = the .pdf path, or None to show.
  """
  if savepath is None:
    plt.show()
    return
  from matplotlib.backends.backend_pdf import PdfPages
  with PdfPages(savepath) as pdf:
    for f in figs:
      pdf.savefig(f, bbox_inches="tight")
      plt.close(f)


# The basic LCDM subset for the chi2-colored triangle: lowercased dump
# names seen in covmat headers, each with its getdist LaTeX label (no
# surrounding $). tau is not sampled in these dumps; w0 / wa and the
# nuisances are deliberately left out (the triangle reads best small).
_LCDM_ALIASES = (
  ("as",      r"A_\mathrm{s}"),
  ("as_1e9",  r"10^9 A_\mathrm{s}"),
  ("logas",   r"\ln(10^{10} A_\mathrm{s})"),
  ("loga",    r"\ln(10^{10} A_\mathrm{s})"),
  ("ns",      r"n_\mathrm{s}"),
  ("n_s",     r"n_\mathrm{s}"),
  ("h0",      r"H_0"),
  ("omegab",  r"\Omega_\mathrm{b}"),
  ("omega_b", r"\Omega_\mathrm{b}"),
  ("omegam",  r"\Omega_\mathrm{m}"),
  ("omega_m", r"\Omega_\mathrm{m}"),
)


def _lcdm_columns(names):
  """
  The LCDM subset of a dump's parameter names, with LaTeX labels.

  Matches each name (case-insensitively) against _LCDM_ALIASES and
  returns the ones present, in the alias table's canonical order
  (A_s, n_s, H0, Omega_b, Omega_m). Names the table does not know
  (w0, wa, photo-z shifts, IA amplitudes) are skipped.

  Arguments:
    names = parameter column names, in the dump's column order.

  Returns:
    (kept, labels): the matched dump names and their labels.
  """
  kept   = []
  labels = []
  for alias, label in _LCDM_ALIASES:
    for n in names:
      if n.lower() == alias and n not in kept:
        kept.append(n)
        labels.append(label)
  return kept, labels


# which lowercased dump names play each role in the physical cuts
# (plus our own derived omegamh2 column); used to find the triangle
# panels whose two axes determine a cut variable.
_CUT_ROLES = (
  ("h0",   ("h0",)),
  ("ob",   ("omegab", "omega_b")),
  ("om",   ("omegam", "omega_m")),
  ("omh2", ("omegamh2",)),
)


def _cut_role(name):
  """The cut role ("h0" / "ob" / "om" / "omh2") of a column name, or
  None when the column plays no part in the physical cuts."""
  low = name.lower()
  for role, aliases in _CUT_ROLES:
    if low in aliases:
      return role
  return None


def _cut_exclusion(rx, ry, xx, yy, cuts):
  """
  Boolean mask of the physically-cut region on one triangle panel.

  A cut on a derived product is a sharp 2D region only on a panel
  whose two axes determine that product:

    omega_b h^2  = Omega_b (H0/100)^2      on the (H0, Omega_b) panel
    omegam^2 h^2 = (Omega_m H0/100)^2      on (H0, Omega_m),
                 = Omega_m * (Omega_m h^2) on (Omega_m, omegam h^2),
                 = (Omega_m h^2)^2/(H0/100)^2 on (H0, omegam h^2).

  Every other panel only thins marginally, so it gets no shading.

  Arguments:
    rx, ry = cut roles of the panel's x and y axes (see _cut_role).
    xx, yy = meshgrid of axis values covering the panel.
    cuts   = mapping with "omegabh2_cut" / "omegabh2_lo" /
             "omegam2h2_lo" / "omegam2h2_hi" (any may be None =
             that cut is off).

  Returns:
    a boolean grid, True where the cuts exclude, or None when this
    panel determines no cut variable (or the cut is off).
  """
  obcut = cuts.get("omegabh2_cut")
  oblo  = cuts.get("omegabh2_lo")
  lo    = cuts.get("omegam2h2_lo")
  hi    = cuts.get("omegam2h2_hi")
  pair  = {rx, ry}

  if pair == {"h0", "ob"} and (obcut is not None or oblo is not None):
    h0 = xx if rx == "h0" else yy
    ob = yy if rx == "h0" else xx
    obh2 = ob * (h0 / 100.0) ** 2
    bad = np.zeros(obh2.shape, dtype=bool)
    if obcut is not None:
      bad |= obh2 >= obcut
    if oblo is not None:
      bad |= obh2 <= oblo
    return bad

  g2 = None
  if pair == {"h0", "om"}:
    h0 = xx if rx == "h0" else yy
    om = yy if rx == "h0" else xx
    g2 = (om * h0 / 100.0) ** 2
  elif pair == {"om", "omh2"}:
    om   = xx if rx == "om" else yy
    omh2 = yy if rx == "om" else xx
    g2 = om * omh2
  elif pair == {"h0", "omh2"}:
    h0   = xx if rx == "h0" else yy
    omh2 = yy if rx == "h0" else xx
    g2 = omh2 ** 2 / (h0 / 100.0) ** 2
  if g2 is None or (lo is None and hi is None):
    return None
  bad = np.zeros(g2.shape, dtype=bool)
  if lo is not None:
    bad |= g2 <= lo
  if hi is not None:
    bad |= g2 >= hi
  return bad


def _shade_cuts(g, plot_names, cuts):
  """
  Gray out the physically-cut regions on a getdist triangle.

  Walks the lower-triangle panels; where a panel's two axes
  determine a cut variable (see _cut_exclusion), fills the excluded
  region light gray under the points, so an empty corner reads as
  "removed by the cut", not as an emulator failure. Adds one caption
  line naming the convention.

  Arguments:
    g          = the getdist subplot plotter (after triangle_plot).
    plot_names = the plotted column names, in triangle order
                 (panel [i][j] has x = plot_names[j],
                 y = plot_names[i]).
    cuts       = the cut values (see _cut_exclusion).
  """
  n = len(plot_names)
  for i in range(1, n):
    for j in range(i):
      ax = g.subplots[i][j]
      if ax is None:
        continue
      rx = _cut_role(plot_names[j])
      ry = _cut_role(plot_names[i])
      if rx is None or ry is None:
        continue
      xs = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 200)
      ys = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 200)
      xx, yy = np.meshgrid(xs, ys)
      bad = _cut_exclusion(rx=rx, ry=ry, xx=xx, yy=yy, cuts=cuts)
      if bad is None or not bad.any():
        continue
      # zorder 0 renders the fill under the already-drawn points;
      # the [0.5, 1.5] level band selects exactly the True cells.
      ax.contourf(xx, yy, bad.astype(float),
                  levels=[0.5, 1.5], colors=["0.88"], zorder=0)
      # contourf can widen the limits; pin them back to the data's.
      ax.set_xlim(xs[0], xs[-1])
      ax.set_ylim(ys[0], ys[-1])
  g.fig.text(0.99, 0.99,
             "gray: region removed by the physical cuts",
             ha="right", va="top", fontsize=9, color="0.35")


def _lcdm_triangle_fig(source, names, dchi2, cuts=None):
  """
  getdist triangle of a source's LCDM parameters, colored by chi2.

  Each off-diagonal panel is a scatter of the source's cosmologies
  (one point per used row, in the same sorted-idx order
  eval_source_chi2 scores), colored by log10 delta-chi2; the
  diagonal shows the 1D densities. It answers where in LCDM space
  the emulator fails, not just how often. When both Omega_m and H0
  are present, the derived omega_m h^2 = Omega_m (H0/100)^2 (the
  structure-amplitude direction) is appended as an extra triangle
  axis. Returns None when fewer than two LCDM columns are
  recognized in `names`.

  Arguments:
    source = source dict with "C" (param dump) and "idx" (used rows).
    names  = parameter column names, in the dump's column order.
    dchi2  = (N,) per-row delta-chi2, sorted-idx order (as returned
             by coverage_diagnostic / eval_source_chi2).
    cuts   = optional mapping with the physical-cut values
             ("omegabh2_cut" / "omegam2h2_lo" / "omegam2h2_hi");
             when given, the excluded regions are shaded gray on the
             panels that determine them (see _shade_cuts), so an
             empty corner is not mistaken for an emulator failure.

  Returns:
    the matplotlib Figure of the triangle, or None.
  """
  lcdm, labels = _lcdm_columns(names)
  if len(lcdm) < 2:
    return None

  rows = np.sort(source["idx"])
  # raw physical parameters of the used rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  cols = []
  for n in lcdm:
    cols.append(names.index(n))

  # color = clipped log10 delta-chi2 (saturated ends so outliers do
  # not wash out the bulk; see _log_dchi2_color for the band).
  logc = _log_dchi2_color(dchi2)

  # derived omega_m h^2 = Omega_m * (H0 / 100)^2, added as its own
  # triangle axis when both parents are among the matched columns
  # (a derived column appended before MCSamples is built).
  vals = P[:, cols]
  i_h0 = None
  i_om = None
  for i, n in enumerate(lcdm):
    if n.lower() == "h0":
      i_h0 = i
    if n.lower() in ("omegam", "omega_m"):
      i_om = i
  plot_names = list(lcdm)
  all_labels = list(labels)
  columns    = [vals]
  if i_h0 is not None and i_om is not None:
    omh2 = vals[:, i_om] * (vals[:, i_h0] / 100.0) ** 2
    columns.append(omh2[:, None])
    plot_names.append("omegamh2")
    all_labels.append(r"\Omega_\mathrm{m} h^2")

  # one extra MCSamples column holds the color, so getdist's
  # plot_3d_with_param can read it by name.
  columns.append(logc[:, None])
  data = np.concatenate(columns, axis=1)
  samples = MCSamples(samples=data,
                      names=plot_names + ["logdchi2"],
                      labels=all_labels + [r"\log_{10}\Delta\chi^2"],
                      settings={"smooth_scale_1D": 0.3,
                                "smooth_scale_2D": 0.3,
                                "fine_bins_2D": 512})

  g = gdplots.get_subplot_plotter(width_inch=9)
  # viridis: sequential and colorblind-safe (the palette rule).
  g.settings.colormap_scatter = "viridis"
  # plot_3d_with_param turns every off-diagonal panel into a scatter
  # colored by that column, with one shared colorbar.
  g.triangle_plot(samples,
                  plot_names,
                  plot_3d_with_param="logdchi2")
  # shade the physically-cut regions so their emptiness reads as
  # "removed by the cut", not as a failure of the emulator.
  if cuts:
    _shade_cuts(g=g, plot_names=plot_names, cuts=cuts)
  return g.fig


def _monomial(vec, labels):
  """
  The product-of-powers string for one ln-space direction.

  A direction with exponent vector v in ln-parameter space is the
  monomial exp(direction) = prod_i p_i^(v_i); the string spells it
  out with the exponents sorted by size and rescaled so the largest
  is 1 (a direction's overall normalization is arbitrary).

  Arguments:
    vec    = (n_params,) monomial exponents of the direction.
    labels = LaTeX labels of the parameters (no surrounding $).

  Returns:
    the mathtext string, e.g. "$\\propto \\ln[H_0^{+1.00}\\,...]$".
  """
  w = vec / np.abs(vec).max()
  order = np.argsort(-np.abs(w))
  terms = []
  for i in order:
    terms.append(f"{labels[i]}^{{{w[i]:+.2f}}}")
  return r"$\propto \ln[" + r"\,".join(terms) + r"]$"


def _pc_label(k, vec, frac, labels):
  """
  Axis label for one ln-parameter principal component.

  The "PCk (share of ln-var)" prefix plus the component's monomial
  (see _monomial).

  Arguments:
    k      = component number (1-based, the "PC1" prefix).
    vec    = (n_params,) monomial exponents of the component (for a
             standardized PCA, eigenvector over the per-parameter
             sigma).
    frac   = this component's share of the total (standardized)
             ln-variance.
    labels = LaTeX labels of the parameters (no surrounding $).

  Returns:
    the axis-label string (matplotlib mathtext).
  """
  return (f"PC{k} ({100.0 * frac:.0f}% of ln-var)  "
          + _monomial(vec, labels))


def _lnparam_pca_fig(source, names, color, clabel, title,
                     fit_target=None, vmin=None, vmax=None):
  """
  First two ln-parameter principal components, colored per row.

  Each ln parameter is centered and scaled to unit variance, and the
  PCA eigendecomposes their sample correlation matrix. The
  standardization is essential: the raw ln variances differ wildly
  (the As prior spans a factor of ten, ns a few percent), so a
  covariance PCA reads back the per-parameter prior widths (PC1 =
  pure As) instead of any correlated combination. On standardized
  variables a component PC = sum_i w_i (ln p_i - mu_i) / sigma_i is
  still a product of parameter powers,
    exp(PC) proportional to  As^(w_As/sigma_As) * ns^(w_ns/sigma_ns) * ...,
  so the axis labels report the effective exponents w_i / sigma_i
  (omega_m h^2 is one such monomial, so the base columns already
  span it; the derived column stays out, keeping the matrix
  non-singular).

  The scatter colors each row by `color` (clipped log10 delta-chi2
  for the hardness page; local training sparsity for the coverage
  page): a color gradient along a direction names the power-law
  combination the colored quantity grows along. With `fit_target`,
  that direction is also measured: a least-squares fit of
  ln(fit_target) on the standardized ln parameters, annotated under
  the title as a monomial with its R^2 (low R^2 = the quantity is
  diffuse, not directional). Returns None when fewer than two LCDM
  columns are recognized, or when a parameter is not strictly
  positive (ln undefined).

  Arguments:
    source     = source dict with "C" (param dump) and "idx" (used
                 rows).
    names      = parameter column names, in the dump's column order.
    color      = (N,) per-row scatter colors, sorted-idx order,
                 already transformed / clipped for display.
    clabel     = colorbar label (mathtext allowed).
    title      = axes title.
    fit_target = optional (N,) positive per-row values; fits
                 ln(fit_target) on the standardized ln parameters
                 and annotates the fitted monomial + R^2.
    vmin, vmax = optional pinned colorbar limits; None lets
                 matplotlib scale to the data (pin the chi2 page to
                 _CHI2_CBAND so colorbars compare across runs).

  Returns:
    the matplotlib Figure, or None.
  """
  lcdm, labels = _lcdm_columns(names)
  if len(lcdm) < 2:
    return None

  rows = np.sort(source["idx"])
  # raw physical parameters of the used rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  cols = []
  for n in lcdm:
    cols.append(names.index(n))
  V = P[:, cols]
  if np.any(V <= 0.0):
    return None

  # center each ln parameter and scale it to unit variance (the
  # per-parameter normalization: subtracting the mean is the
  # ln-space form of dividing by a central value; dividing by sigma
  # puts wide and narrow priors on the same footing).
  L   = np.log(V)
  X   = L - L.mean(axis=0)
  sig = X.std(axis=0)
  Z   = X / sig
  # sample correlation matrix of the ln parameters (the covariance
  # of the standardized variables).
  S = np.cov(Z, rowvar=False)

  # eigh returns ascending eigenvalues; flip to descending so
  # column j of evecs is the j-th principal direction.
  evals, evecs = np.linalg.eigh(S)
  order = np.argsort(evals)[::-1]
  evals = evals[order]
  evecs = evecs[:, order]

  # effective monomial exponents: PC = sum_i w_i (ln p_i - mu)/sig_i
  # means exp(PC) carries p_i to the power w_i / sig_i. One column
  # per plotted component. The [:, None] adds a size-1 axis so sig
  # ((k,)) divides each column of evecs ((k, 2)) elementwise.
  expo = evecs[:, :2] / sig[:, None]

  # deterministic sign: make the largest-|exponent| positive (an
  # eigenvector's sign is arbitrary; this fixes the label). Flip the
  # eigenvector and its exponents together so points and label match.
  for j in range(2):
    imax = np.argmax(np.abs(expo[:, j]))
    if expo[imax, j] < 0.0:
      evecs[:, j] = -evecs[:, j]
      expo[:, j]  = -expo[:, j]

  # project the standardized ln parameters on the first two
  # directions.
  pcs  = Z @ evecs[:, :2]
  frac = evals / evals.sum()

  # optional direction fit: ln(fit_target) = a + Z @ b. The fitted b
  # over sigma gives the monomial the quantity grows along; R^2 says
  # how directional (vs diffuse) it is.
  fitline = None
  if fit_target is not None:
    t = np.log(np.maximum(np.asarray(fit_target, dtype="float64"),
                          1e-300))
    A = np.column_stack([np.ones(Z.shape[0]), Z])
    coefs, *_ = np.linalg.lstsq(A, t, rcond=None)
    pred = A @ coefs
    ss_res = ((t - pred) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-300)
    fitline = ("grows along " + _monomial(coefs[1:] / sig, labels)
               + f"  ($R^2$ = {r2:.2f})")

  fig, ax = plt.subplots(figsize=(8, 6))
  # draw dark-to-bright so the rare high-color points sit on top of
  # the overplotted bulk instead of being buried under it.
  cvals = np.asarray(color)
  order = np.argsort(cvals)
  sc = ax.scatter(pcs[order, 0],    # x = PC1 projection
                  pcs[order, 1],    # y = PC2 projection
                  c=cvals[order],
                  cmap="viridis",   # sequential, colorblind-safe
                  s=4,
                  vmin=vmin,
                  vmax=vmax)
  # extend arrows: values continue past the clipped color ends.
  fig.colorbar(sc, ax=ax, extend="both", label=clabel)
  ax.set_xlabel(_pc_label(k=1,
                          vec=expo[:, 0],
                          frac=frac[0],
                          labels=labels))
  ax.set_ylabel(_pc_label(k=2,
                          vec=expo[:, 1],
                          frac=frac[1],
                          labels=labels))
  if fitline is None:
    ax.set_title(title)
  else:
    # two-line title: the page name, then the fitted direction.
    ax.set_title(title + "\n" + fitline, fontsize=11)
  fig.tight_layout()
  return fig


def plot_diagnostics(train_losses,
                     medians,
                     means,
                     fracs,
                     thresholds,
                     coverage,
                     floor=None,
                     hard_dir=None,
                     val_set=None,
                     names=None,
                     cuts=None,
                     savepath=None):
  """
  All available diagnostics as a single multipage figure / PDF.

  Page 1 (2x2): the training history (loss curves; fraction over
    each delta-chi2 threshold vs epoch) and the coverage diagnostic
    (hardness vs local sparsity; good/bad sparsity histograms).
  Page 2: the local-linear data-only floor (model vs floor
    delta-chi2), if `floor` is given.
  Page 3: the hard-direction regression (univariate ranking and
    joint log-linear coefficients), if `hard_dir` is given.
  Page 4: the getdist LCDM triangle of the val cosmologies, colored
    by log10 delta-chi2, if `val_set` and `names` are given.
  Page 5: the first two ln-parameter PCA directions of the val
    cosmologies, colored the same way (a principal direction in ln
    space is a product of parameter powers), same condition.
  Page 6: the same PCA plane colored by local training sparsity
    (coverage's knn_dist), with the fitted sparsity direction
    annotated -- names the combinations where training is thin,
    independent of the chi2; same condition.

  floor / hard_dir / val_set are optional so a run can drop a page
  it cannot produce (e.g. the local-linear floor is defined only for
  a plain chi2fn, so a --rescale run omits it).

  Arguments:
    train_losses, medians, means, fracs, thresholds = the
      run_emulator histories (see plot_history).
    coverage = the dict coverage_diagnostic returned.
    floor    = the dict local_linear_floor returned, or None.
    hard_dir = the dict hard_direction_regression returned, or None.
    val_set  = the validation source dict ("C" / "idx"), or None;
               its rows must be the ones coverage's dchi2 scored.
    names    = parameter column names in the dump's order, or None.
    cuts     = optional physical-cut values ("omegabh2_cut" /
               "omegam2h2_lo" / "omegam2h2_hi") to shade gray on the
               triangle page (empty cut regions then read as removed,
               not as failures).
    savepath = if given, write a (multipage) PDF there and close;
               if None, show each page interactively.
  """
  figs = []
  # page 1: history (top row) + coverage (bottom row).
  f1, ax = plt.subplots(2, 2, figsize=(12, 9))
  _history_panels(ax[0, 0], ax[0, 1], train_losses, medians,
                  means, fracs, thresholds)
  _coverage_panels(ax[1, 0], ax[1, 1], coverage["knn_dist"],
                   coverage["dchi2"], coverage["k_nn"])
  f1.tight_layout()
  figs.append(f1)

  # page 2: the local-linear data floor (plain chi2fn only).
  if floor is not None:
    f2, a2 = plt.subplots(figsize=(6, 6))
    _floor_panel(a2, floor)
    f2.tight_layout()
    figs.append(f2)

  # page 3: the hard-direction regression.
  if hard_dir is not None:
    f3, a3 = plt.subplots(1, 2, figsize=(13, 6))
    _hard_direction_panels(a3[0], a3[1], hard_dir)
    f3.tight_layout()
    figs.append(f3)

  # pages 4-6: where in parameter space the failures live. Page 4 is
  # the LCDM triangle (getdist lays it out itself, so no
  # tight_layout); page 5 the ln-parameter PCA plane colored by chi2
  # (hardness); page 6 the same plane colored by local training
  # sparsity (coverage), with the fitted sparsity direction --
  # aligned gradients on 5 and 6 say the failures are coverage,
  # diverging ones say the hardness is intrinsic.
  if val_set is not None and names is not None:
    f4 = _lcdm_triangle_fig(source=val_set,
                            names=names,
                            dchi2=coverage["dchi2"],
                            cuts=cuts)
    if f4 is not None:
      figs.append(f4)
    f5 = _lnparam_pca_fig(source=val_set,
                          names=names,
                          color=_log_dchi2_color(coverage["dchi2"]),
                          clabel=r"$\log_{10}\Delta\chi^2$",
                          title="ln-parameter PCA of the val "
                                "cosmologies, colored by hardness",
                          fit_target=np.maximum(
                            np.asarray(coverage["dchi2"],
                                       dtype="float64"), 1e-12),
                          vmin=_CHI2_CBAND[0],
                          vmax=_CHI2_CBAND[1])
    if f5 is not None:
      figs.append(f5)
    # sparsity color: percentile-clipped so one far outlier does not
    # own the colorbar; the direction fit uses the unclipped values.
    knn = np.asarray(coverage["knn_dist"], dtype="float64")
    lo, hi = np.percentile(knn, [1.0, 99.0])
    f6 = _lnparam_pca_fig(source=val_set,
                          names=names,
                          color=np.clip(knn, lo, hi),
                          clabel=(f"mean dist to {coverage['k_nn']} "
                                  "nearest train pts (whitened)"),
                          title="ln-parameter PCA, colored by "
                                "training sparsity",
                          fit_target=knn,
                          vmin=lo,
                          vmax=hi)
    if f6 is not None:
      figs.append(f6)

  _save_pages(figs, savepath)


def source_param_samples(source, names, labels, label):
  """
  getdist MCSamples of one source's cosmological parameters.

  Pulls the rows the source actually uses (source["idx"]) from its
  parameter dump and wraps them as equally-weighted samples for a
  coverage triangle (no likelihood, no chi2). Reads no module
  globals -- source, names, labels, and the legend label all arrive
  as arguments.

  Arguments:
    source = source dict with "C" (full param dump) and "idx"
             (global rows actually in use).
    names  = parameter column names, in the dump's column
             order (pgeom.names).
    labels = LaTeX labels for those columns (no surrounding $).
    label  = legend label for this set (e.g. "train").

  Returns:
    an MCSamples over the source's used parameter rows.
  """
  # the rows this source uses -- coverage is about what was
  # trained / validated on, not the whole file.
  rows = np.sort(source["idx"])
  # raw physical parameters of those rows (never whitened).
  P = np.asarray(source["C"][rows], dtype="float64")
  return MCSamples(samples=P, 
                   names=names, 
                   labels=labels,
                   label=label,
                   settings={"smooth_scale_1D": 0.3,
                             "smooth_scale_2D": 0.3,
                             "fine_bins_2D": 512})


def dv_to_xi(dv_row, geom):
  """
  Reshape one full data-vector row into the (theta, xip, xim)
  matrix layout of plot_xi, using its cosmic-shear block.

  Takes the leading xi_size entries (xi_plus then xi_minus, pairs
  (i<=j) outer / theta inner) and scatters each pair's ntheta values
  into the (i, j) slot of an (ntheta, ntomo, ntomo) array (upper
  triangle filled; the rest stay 0 and plot_xi never reads them).

  Arguments:
    dv_row = (total_size,) full data vector; only the leading
             geom.xi_size cosmic-shear entries are used.
    geom   = geometry carrying ntheta / source_ntomo /
             theta_centers / xi_size.
  Returns:
    (theta, xip, xim): theta (ntheta,) [arcmin]; xip, xim
    (ntheta, ntomo, ntomo).
  """
  nt    = geom.source_ntomo
  ntha  = geom.ntheta
  block = np.asarray(dv_row[:geom.xi_size], dtype="float64")
  pairs = []
  for i in range(nt):
    for j in range(i, nt):
      pairs.append((i, j))
  half  = len(pairs) * ntha
  xip = np.zeros((ntha, nt, nt))
  xim = np.zeros((ntha, nt, nt))
  for p, (i, j) in enumerate(pairs):
    xip[:, i, j] = block[p * ntha:(p + 1) * ntha]
    xim[:, i, j] = block[half + p * ntha:
                         half + (p + 1) * ntha]
  return (geom.theta_centers, xip, xim)


def plot_xi(pm, xi, xi_ref = None, param = None, colorbarlabel = None,
            marker = None, linestyle = None, linewidth = None,
            ylim = [0.88,1.12], cmap = 'gist_rainbow', legend = None,
            legendloc = (0.6,0.78), yaxislabelsize = 16, yaxisticklabelsize = 10,
            xaxisticklabelsize = 20, bintextpos = [[0.8, 0.875],[0.2,0.875]],
            bintextsize = 15, figsize = (12, 12), show = None, thetashow=[3,1000],
            colorbar=1):
    """
    Tomographic grid of xi+ / xi- curves (a visual check, not a
    diagnostic page). One panel per redshift-bin pair: xi+ on the
    lower triangle, xi- on the upper; absolute curves when xi_ref
    is None, ratios xi / xi_ref otherwise. Ported byte-faithfully
    from the notebook, so its body keeps the original style.

    Arguments:
      pm            = "p", "m", or "pm": which of xi+ / xi- to draw.
      xi            = list of (theta, xip, xim) triples (dv_to_xi
                      output), one curve set per line drawn.
      xi_ref        = reference triple; when given, panels show the
                      ratio xi / xi_ref and share both axes.
      param         = per-curve scalar (e.g. a cosmology parameter)
                      that colors the curves through cmap; length
                      must match xi.
      colorbarlabel = label for the param colorbar.
      marker        = marker cycle (list) or None.
      linestyle     = linestyle cycle (list); solid when None.
      linewidth     = linewidth cycle (list) or None.
      ylim          = y range of the ratio panels.
      cmap          = matplotlib colormap name for param coloring.
      legend        = per-curve legend labels or None.
      legendloc     = legend anchor (axes fraction).
      yaxislabelsize / yaxisticklabelsize / xaxisticklabelsize
                    = font sizes for the y label and tick labels.
      bintextpos    = [[x, y] lower, [x, y] upper] axes-fraction
                      anchors of the per-panel bin annotation.
      bintextsize   = font size of that annotation.
      figsize       = figure size in inches.
      show          = list of (i, j) bin pairs to draw; None = all.
      thetashow     = [min, max] theta range (arcmin) shown.
      colorbar      = 1 to draw the param colorbar, None to skip.

    Returns:
      the matplotlib figure (0 on malformed input, with a message).
    """

    (theta, xip, xim) = xi[0]
    (ntheta, ntomo, ntomo2) = xip.shape    

    if ntomo != ntomo2:
        print("Bad Input (ntomo)")
        return 0
            
    if ntheta != len(theta):
        print("Bad Input (theta)")
        return 0

    if xi_ref is None:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = False, 
            gridspec_kw = {'wspace': 0.25, 'hspace': 0.05})
    else:
        fig, axes = plt.subplots(
            nrows = ntomo, 
            ncols = ntomo, 
            figsize = figsize, 
            sharex = True, 
            sharey = True, 
            gridspec_kw = {'wspace': 0.0, 'hspace': 0.0})    

    cm = plt.get_cmap(cmap)

    if not (param is None or colorbar is None):
        norm = matplotlib.colors.Normalize(vmin=param[0],
                                           vmax=param[-1])
        cb = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax = axes.ravel().tolist(), 
            orientation = 'vertical', 
            aspect = 50, 
            pad = -0.16, 
            shrink = 0.5
        )
        if not (colorbarlabel is None):
            cb.set_label(label = colorbarlabel, 
                         size = 20, 
                         weight = 'bold', 
                         labelpad = 2)
        if len(param) != len(xi):
            print("Bad Input")
            return 0

    if not (marker is None):
        markercycler = itertools.cycle(marker)
    
    if not (linestyle is None):
        linestylecycler = itertools.cycle(linestyle)
    else:
        linestylecycler = itertools.cycle(['solid'])
    
    if not (linewidth is None):
        linewidthcycler = itertools.cycle(linewidth)
    else:
        linewidthcycler = itertools.cycle([1.0])
        
    for i in range(ntomo):
        for j in range(ntomo):
            if i>j:                
                axes[j,i].axis('off')
            else:
                ximin = []
                ximax = []
                for (theta, xip, xim) in xi:
                    if pm > 0:
                        ximin.append(np.min(theta*xip[:,i,j]*10**4))
                        ximax.append(np.max(theta*xip[:,i,j]*10**4))
                    else:
                        ximin.append(np.min(theta*xim[:,i,j]*10**4))
                        ximax.append(np.max(theta*xim[:,i,j]*10**4))
                        
                axes[j,i].set_xlim(thetashow)
                
                if xi_ref is None:
                    axes[j,i].set_ylim([np.min(ylim[0]*np.array(ximin)), 
                                        np.max(ylim[1]*np.array(ximax))])
                else:
                    tmp = np.array(ylim) - 1
                    axes[j,i].set_ylim(tmp.tolist())
                axes[j,i].set_xscale('log')
                axes[j,i].set_yscale('linear')
                
                if i == 0:
                    if xi_ref is None:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"$\theta \xi_{+} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"$\theta \xi_{-} \times 10^4$", 
                                                 fontsize=yaxislabelsize)
                    else:
                        if pm > 0:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{+})$", 
                                                 fontsize=yaxislabelsize)
                        else:
                            axes[j,i].set_ylabel(r"frac. diff. ($\xi_{-})$", 
                                                 fontsize=yaxislabelsize)

                if j == ntomo-1:
                    axes[j,i].set_xlabel(r"$\theta$ [arcmin]", fontsize=16)
                for item in (axes[j,i].get_yticklabels()):
                    item.set_fontsize(yaxisticklabelsize)
                for item in (axes[j,i].get_xticklabels()):
                    item.set_fontsize(xaxisticklabelsize)

                if pm > 0:
                    axes[j,i].text(bintextpos[0][0], 
                                   bintextpos[0][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=bintextsize,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)
                else:
                    axes[j,i].text(bintextpos[1][0], 
                                   bintextpos[1][1], 
                                   "$(" +  str(i) + "," +  str(j) + ")$", 
                                   horizontalalignment='center', 
                                   verticalalignment='center',
                                   fontsize=15,
                                   usetex=True,
                                   transform=axes[j,i].transAxes)

                if xi_ref is None:
                    # plot(x, y, ...): x = theta, y = theta *
                    # xi_+/- * 1e4 (the scaled correlation fn).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xip[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                axes[j,i].plot(theta, theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               theta*xim[:,i,j]*10**4, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                else:
                    (theta_ref, xip_ref, xim_ref) = xi_ref
                    # plot(x, y, ...): x = theta, y = xi_+/- /
                    # xi_ref - 1 (the fractional difference).
                    for x, (theta, xip, xim) in enumerate(xi):
                        if not np.array_equal(theta, theta_ref):
                            print("inconsistent theta bins")
                            return 0
                        if pm > 0:
                            if marker is None:
                                axes[j,i].plot(theta, xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               linewidth=next(linewidthcycler), 
                                               linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xip[:,i,j]/xip_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None',
                                               marker=next(markercycler),  
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', 
                                               markersize=3)
                        else:
                            if marker is None:   
                                lines = axes[j,i].plot(theta, 
                                                       xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                                       color=cm(x/len(xi)), 
                                                       linewidth=next(linewidthcycler), 
                                                       linestyle=next(linestylecycler))
                            else:
                                axes[j,i].plot(theta, 
                                               xim[:,i,j]/xim_ref[:,i,j]-1.0, 
                                               color=cm(x/len(xi)), 
                                               markerfacecolor='None', 
                                               marker=next(markercycler), 
                                               markeredgecolor=cm(x/len(xi)), 
                                               linestyle='None', markersize=3)    
    if not (legend is None):
        if len(legend) != len(xi):
            print("Bad Input")
            return 0
        fig.legend(legend, 
                   loc=legendloc,
                   borderpad=0.1,
                   handletextpad=0.4,
                   handlelength=1.5,
                   columnspacing=0.35,
                   scatteryoffsets=[0],
                   frameon=False)  
    if not (show is None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.show()
    else:
        return (fig, axes)
