"""ランナ（製品の下端より下）の各行で、中心が固まっていない幅 [mm] を測り、図を描く。"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path(sys.argv[1])
tags = sys.argv[2:]
fig, axes = plt.subplots(1, len(tags), figsize=(5 * len(tags), 5), squeeze=False)
for ax, tag in zip(axes[0], tags):
    d = np.load(out / f"{tag}.npz")
    info = json.loads((out / f"{tag}.json").read_text())
    cell = info["cell"]
    mask = d["mask"]
    th = d["thermal_theta_end"][:, :, 0]
    ny, nx = mask.shape
    y = (np.arange(ny) + 0.5) * cell - float(d["y_edge"])
    x = (np.arange(nx) + 0.5) * cell - float(d["axis_x"])
    live = mask & (th > 0.3)
    print(tag)
    for yy in (-2, -5, -10, -15, -20, -25):
        i = int(np.argmin(np.abs(y - yy)))
        row = mask[i]
        print(
            f"  Y={y[i]:6.1f}  runner {row.sum() * cell:6.1f} mm  melt {live[i].sum() * cell:6.1f} mm  "
            f"thk {np.unique(np.round(d['thk'][i][row], 2))}"
        )
    sel = (y > -32) & (y < 25)
    xs = np.abs(x) < 160
    img = np.where(mask, th, np.nan)[np.ix_(sel, xs)]
    ax.imshow(
        img,
        origin="lower",
        cmap="viridis",
        vmin=0,
        vmax=1,
        extent=[x[xs][0], x[xs][-1], y[sel][0], y[sel][-1]],
        aspect="auto",
    )
    ax.contour(x[xs], y[sel], np.nan_to_num(img, nan=-1), levels=[0.3], colors="r", linewidths=0.8)
    ax.axhline(0, color="w", lw=0.5)
    ax.set_title(f"{tag}: centre θ at end of fill (red = 0.3)")
fig.tight_layout()
fig.savefig(out / ("channel_" + "_".join(tags) + ".png"), dpi=110)
