---
name: plots-no-red-green
description: "In matplotlib plots never use red+green together (colorblind accessibility); set every line's color explicitly from a colorblind-safe palette."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: a703cd31-5515-4fe4-8d50-bdf7c9f08651
---

Never rely on matplotlib's default color cycle for multi-line plots — its 3rd
and 4th colors are green then red, the worst pair for red-green color blindness
(the most common form). Set every line's `color=` explicitly from a
colorblind-safe palette.

Default palette to reach for: Bang Wong (2011) minus its green and vermillion —
blue / orange / reddish-purple / black / sky-blue:
`["#0072B2", "#E69F00", "#CC79A7", "#000000", "#56B4E9"]`. Or set it globally
once with `plt.style.use("tableau-colorblind10")`. For grayscale robustness,
also vary linestyle, not color alone.

**Why:** the user flagged green+red as a hard "no no" on plots.

**How to apply:** any new plot — explicit colorblind-safe colors per line, no
default-cycle reliance. Pairs with [[pytorch-teaching-style]].
