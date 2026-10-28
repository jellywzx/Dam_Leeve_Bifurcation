"""Shared configuration and helpers for OpenBench dam/river plotting scripts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OPENBENCH_OUTPUT_ROOT = Path("/share/home/dq134/wzx/openbench/openbench_1015/output")
OUTPUT_ROOT = Path("/share/home/dq134/wzx/CoLM/dam_analysis")
DEFAULT_INPUT_SUFFIX = Path("comparisons/Diff_Plot")
DEFAULT_DAMS = OUTPUT_ROOT / "dams_global.csv"
DEFAULT_DAM_BUILT_BEFORE = 2005
DEFAULT_ROUTING_FILE = Path(
    "/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment/"
    "grid_routing_data_06min.nc"
)
EXTENT_PADDING_FRACTION = 0.02
MIN_EXTENT_PADDING_DEGREES = 0.25
RIVER_DETAIL_CONFIG = {
    "all": {"minimum_uparea_km2": None, "linewidth": 0.35},
    "fine": {"minimum_uparea_km2": 500.0, "linewidth": 0.45},
    "medium": {"minimum_uparea_km2": 5_000.0, "linewidth": 0.65},
    "coarse": {"minimum_uparea_km2": 50_000.0, "linewidth": 0.85},
    "none": {"minimum_uparea_km2": None, "linewidth": 0.0},
}
DEFAULT_RIVER_DETAIL = "fine"


def region_name_from_evaluation(evaluation_name: str) -> str:
    """Return the display-region name encoded in an OpenBench evaluation name."""
    normalized = evaluation_name.strip().lower()
    if normalized.startswith("lowermississippi_"):
        return "Lower Mississippi"
    if normalized.startswith("missouri_"):
        return "Missouri"
    if normalized.startswith("amazon_"):
        return "Amazon"
    if normalized.startswith("mekong_"):
        return "Mekong"
    if normalized.startswith("sa_") or normalized.startswith("streamflow_sa_"):
        return "SA"

    raw_region = re.split(r"_(?:inundation|streamflow)_", evaluation_name, maxsplit=1)[0]
    raw_region = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_region)
    return raw_region.replace("_", " ").strip() or evaluation_name


def openbench_input_dir(evaluation_name: str, input_suffix: Path) -> Path:
    """Join an evaluation and configurable input suffix under the OpenBench root."""
    evaluation_name = evaluation_name.strip()
    if not evaluation_name:
        raise ValueError("The evaluation name must not be empty.")
    input_suffix = Path(input_suffix)
    if input_suffix.is_absolute() or ".." in input_suffix.parts:
        raise ValueError("The input suffix must be a relative path below the evaluation directory.")
    return OPENBENCH_OUTPUT_ROOT / evaluation_name / input_suffix


def infer_region_name(path: Path, configured_region: str | None) -> str | None:
    """Return an explicit region or infer it from an OpenBench output path."""
    if configured_region is not None and configured_region.strip():
        return configured_region.strip()
    for parent in path.parents:
        if parent.parent.name == "output":
            return region_name_from_evaluation(parent.name)
    return None


def validate_extent(
    extent: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    west, east, south, north = extent
    if not np.isfinite(extent).all():
        raise ValueError("--extent values must all be finite.")
    if west >= east or south >= north:
        raise ValueError("--extent requires WEST < EAST and SOUTH < NORTH.")
    if west < -180 or east > 180 or south < -90 or north > 90:
        raise ValueError("--extent must stay within longitude [-180, 180] and latitude [-90, 90].")
    return extent


def filter_dams_by_year(
    dams: pd.DataFrame,
    built_before: int | None,
    built_by: int | None,
) -> tuple[pd.DataFrame, str | None]:
    if built_before is None and built_by is None:
        return dams, None

    valid_year = dams["year"].between(1800, 2030)
    if built_before is not None:
        filtered = dams.loc[valid_year & (dams["year"] < built_before)].copy()
        label = f"dam construction year < {built_before}"
    else:
        filtered = dams.loc[valid_year & (dams["year"] <= built_by)].copy()
        label = f"dam construction year ≤ {built_by}"
    if filtered.empty:
        raise ValueError(f"No dams remain after applying: {label}.")
    return filtered, label


def filter_dams_to_extent(
    dams: pd.DataFrame,
    extent: tuple[float, float, float, float],
) -> pd.DataFrame:
    west, east, south, north = extent
    visible = dams["lon"].between(west, east) & dams["lat"].between(south, north)
    return dams.loc[visible].copy()


def dam_marker_sizes(storage: pd.Series) -> np.ndarray:
    values = pd.to_numeric(storage, errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    scaled = np.log10(values + 1.0)
    upper = np.nanmax(scaled)
    if not np.isfinite(upper) or upper == 0:
        return np.full(values.shape, 35.0)
    return 28.0 + 115.0 * scaled / upper


def storage_marker_size(storage_mcm: float, maximum_storage_mcm: float) -> float:
    maximum = np.log10(maximum_storage_mcm + 1.0)
    if maximum <= 0:
        return 35.0
    return 28.0 + 115.0 * np.log10(storage_mcm + 1.0) / maximum
