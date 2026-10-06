"""Thickness maps of Gate 1 (the drawing, and the balancer variant) and
Gate 2 (the pentagon), for eyeballing the builder against the drawings. Regenerates
``docs/draft/geometry_draft.png``::

    .venv/bin/python docs/draft/geometry_draft.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from core.fan_runner import (  # noqa: E402
    GATE2_DEFAULTS,
    FanRunnerPlateConfig,
    build_fan_runner_plate_geometry,
)

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


def _closeup(ax, cfg: FanRunnerPlateConfig, title: str) -> None:
    g = build_fan_runner_plate_geometry(cfg)
    x0, y0 = g.display_origin_mm()
    thk = np.where(g.mask, g.thickness_mm, np.nan)
    ny, nx = g.shape
    extent = (-x0, nx * g.cell_size_mm - x0, -y0, ny * g.cell_size_mm - y0)
    ax.imshow(thk, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=4)
    ax.set_xlim(-160, 160)
    ax.set_ylim(-45, 25)
    ax.axhline(-cfg.ramp_end_depth_mm, color="white", lw=0.5, ls="--")
    ax.axhline(-cfg.runner_len_mm, color="white", lw=0.5, ls=":")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x [mm]")


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), constrained_layout=True)
    _gate1(axes[0])
    g2 = FanRunnerPlateConfig(**GATE2_DEFAULTS)
    _panel(axes[1][0], g2, "Gate 2 defaults (side 28 below the rim, slants tangent to R12)")
    _closeup(
        axes[1][1],
        FanRunnerPlateConfig(**GATE2_DEFAULTS, cell_size_mm=0.5),
        "Gate 2 close-up, 0.5 mm (-- ramp start d_s = R12 top 18, .. axis d=30)",
    )
    _closeup(
        axes[1][2],
        FanRunnerPlateConfig(**GATE2_DEFAULTS, runner_ramp_on=False, cell_size_mm=0.5),
        "Gate 2, ramp off (band t1.0 x 1, then t_o 2.5; disc 2.5)",
    )
    fig.savefig(OUT, dpi=130)
    print(OUT)


def _gate1(axes) -> None:
    im = _panel(
        axes[0], FanRunnerPlateConfig(), "Gate 1 defaults (red = compression zone, white = gate φ3)"
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
    ax.set_title("Gate 1 close-up, 0.5 mm cells (-- ramp end d=20, .. axis d=30)", fontsize=9)
    ax.set_xlabel("x [mm]")
    _panel(axes[2], FanRunnerPlateConfig(balancer_on=True), "balancer on (w100 / h15 / t1.0)")
    axes[2].set_xlim(-160, 160)
    axes[2].set_ylim(-45, 25)
    axes[0].figure.colorbar(im, ax=axes, shrink=0.8, label="thickness [mm]")


if __name__ == "__main__":
    main()
