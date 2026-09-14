"""Thickness map of the drawing's defaults (and the balancer variant), for
eyeballing the builder against the drawing. Regenerates
``docs/draft/geometry_draft.png``::

    .venv/bin/python docs/draft/geometry_draft.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.fan_runner import FanRunnerPlateConfig, build_fan_runner_plate_geometry  # noqa: E402

OUT = Path(__file__).with_suffix(".png")


def _panel(ax, cfg: FanRunnerPlateConfig, title: str) -> None:
    g = build_fan_runner_plate_geometry(cfg)
    x0, y0 = g.display_origin_mm()
    thk = np.where(g.mask, g.thickness_mm, np.nan)
    ny, nx = g.shape
    extent = (-x0, nx * g.cell_size_mm - x0, -y0, ny * g.cell_size_mm - y0)
    im = ax.imshow(thk, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=4)
    ax.contour(
        g.compression_mask.astype(float),
        levels=[0.5],
        colors="red",
        linewidths=0.8,
        extent=extent,
        origin="lower",
    )
    vx, vy, vr = g.valve_marker_mm
    ax.add_patch(plt.Circle((vx - x0, vy - y0), vr, fill=False, color="white", lw=0.8))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm] (product edge = 0)")
    return im


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    im = _panel(
        axes[0], FanRunnerPlateConfig(), "defaults (red = compression zone, white = gate φ3)"
    )
    ax = axes[1]
    g = build_fan_runner_plate_geometry(FanRunnerPlateConfig(cell_size_mm=0.5))
    x0, y0 = g.display_origin_mm()
    thk = np.where(g.mask, g.thickness_mm, np.nan)
    ny, nx = g.shape
    extent = (-x0, nx * g.cell_size_mm - x0, -y0, ny * g.cell_size_mm - y0)
    ax.imshow(thk, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=4)
    ax.set_xlim(-160, 160)
    ax.set_ylim(-45, 25)
    ax.axhline(-20, color="white", lw=0.5, ls="--")
    ax.axhline(-30, color="white", lw=0.5, ls=":")
    ax.set_title("runner close-up, 0.5 mm cells (-- ramp end d=20, .. axis d=30)", fontsize=9)
    ax.set_xlabel("x [mm]")
    _panel(axes[2], FanRunnerPlateConfig(balancer_on=True), "balancer on (w100 / h15 / t1.0)")
    axes[2].set_xlim(-160, 160)
    axes[2].set_ylim(-45, 25)
    fig.colorbar(im, ax=axes, shrink=0.8, label="thickness [mm]")
    fig.savefig(OUT, dpi=130)
    print(OUT)


if __name__ == "__main__":
    main()
