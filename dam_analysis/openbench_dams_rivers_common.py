"""Shared configuration and helpers for OpenBench dam/river plotting scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

OPENBENCH_OUTPUT_ROOT = Path("/share/home/dq134/wzx/openbench/openbench_1015/output")
OUTPUT_ROOT = Path("/share/home/dq134/wzx/openbench/postprocess/dam_analysis/dam_data")
DEFAULT_INPUT_SUFFIX = Path("comparisons/Diff_Plot")
DEFAULT_DAMS = OUTPUT_ROOT / "dams_global.csv"
DEFAULT_DAM_BUILT_BEFORE = None
FALLBACK_DAM_BUILT_BEFORE = 2005
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


def dam_built_before_from_manifest(input_path: Path) -> tuple[int | None, Path | None]:
    """Return the evaluation end year and manifest found above an input path."""
    input_path = Path(input_path)
    start = input_path if input_path.is_dir() else input_path.parent
    manifest_path = next(
        (
            directory / "run_manifest.json"
            for directory in (start, *start.parents)
            if (directory / "run_manifest.json").is_file()
        ),
        None,
    )
    if manifest_path is None:
        return None, None

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    years = (
        manifest.get("canonical_config", {})
        .get("project", {})
        .get("years")
    )
    if not isinstance(years, list) or not years:
        raise ValueError(
            f"run_manifest.json must contain a non-empty "
            f"canonical_config.project.years list: {manifest_path}"
        )
    try:
        end_year = int(years[-1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid evaluation end year {years[-1]!r} in {manifest_path}"
        ) from exc
    if not 1800 <= end_year <= 2030:
        raise ValueError(
            f"Evaluation end year must be between 1800 and 2030; "
            f"found {end_year} in {manifest_path}"
        )
    return end_year, manifest_path


def routing_numeric_1d(h5: h5py.File, name: str) -> np.ndarray:
    """Read and decode one numeric one-dimensional routing variable."""
    if name not in h5:
        raise ValueError(f"Routing file is missing required variable: {name}")
    dataset = h5[name]
    values = np.asarray(dataset[...]).reshape(-1)
    if values.dtype.kind not in "iuf":
        raise ValueError(f"Routing variable {name} must be numeric.")
    values = values.astype(float, copy=False)
    for attribute in ("_FillValue", "missing_value"):
        if attribute in dataset.attrs:
            missing = float(np.asarray(dataset.attrs[attribute]).reshape(-1)[0])
            values[values == missing] = np.nan
    if "scale_factor" in dataset.attrs:
        values *= float(np.asarray(dataset.attrs["scale_factor"]).reshape(-1)[0])
    if "add_offset" in dataset.attrs:
        values += float(np.asarray(dataset.attrs["add_offset"]).reshape(-1)[0])
    return values


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
