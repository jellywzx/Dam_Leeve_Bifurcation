#!/usr/bin/env python3
"""Shared plotting for all production basin-mask products."""

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    plt = None
    _PLOT_IMPORT_ERROR = exc
else:
    _PLOT_IMPORT_ERROR = None


def plot_basin_mask(
    fraction: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    output_file: Path,
    title: str,
    outlet_points=(),
) -> bool:
    """Write one consistently styled basin preview; return False if unavailable."""
    if plt is None:
        print(
            "  INFO: matplotlib is unavailable; preview PNG skipped: "
            f"{_PLOT_IMPORT_ERROR}"
        )
        return False

    inside = np.isfinite(fraction) & (fraction > 0)
    rows, cols = np.where(inside)
    if rows.size == 0:
        raise ValueError(f"Cannot plot empty basin: {title}")

    row_margin = max(2, int(round(3.0 / abs(float(lat[1] - lat[0])))))
    col_margin = max(2, int(round(3.0 / abs(float(lon[1] - lon[0])))))
    r0 = max(0, int(rows.min()) - row_margin)
    r1 = min(len(lat), int(rows.max()) + row_margin + 1)
    c0 = max(0, int(cols.min()) - col_margin)
    c1 = min(len(lon), int(cols.max()) + col_margin + 1)

    view = np.where(inside[r0:r1, c0:c1], fraction[r0:r1, c0:c1], np.nan)
    dx = abs(float(lon[1] - lon[0]))
    dy = abs(float(lat[1] - lat[0]))
    extent = [
        float(lon[c0] - dx / 2),
        float(lon[c1 - 1] + dx / 2),
        float(lat[r1 - 1] - dy / 2),
        float(lat[r0] + dy / 2),
    ]

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    image = ax.imshow(
        view,
        origin="upper",
        extent=extent,
        interpolation="nearest",
        aspect="equal",
        vmin=0,
        vmax=1,
        cmap="Blues",
    )
    points = list(outlet_points)
    if points:
        ax.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="*",
            s=90,
            color="red",
            edgecolor="black",
            linewidth=0.6,
            label="Outlet/boundary",
            zorder=3,
        )
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.grid(color="0.75", linewidth=0.4, linestyle=":")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    colorbar.set_label("CoLM mapping-weight share")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)
    print(f"  Preview -> {output_file}")
    return True
