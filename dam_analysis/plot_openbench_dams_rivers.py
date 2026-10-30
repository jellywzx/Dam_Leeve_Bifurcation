#!/usr/bin/env python3
"""在 OpenBench 空间结果图上叠加河网、流域边界和水坝。

运行说明
--------
建议先进入脚本目录，以便正确导入同目录下的公共模块：

    cd /share/home/dq134/wzx/openbench/postprocess/dam_analysis

1. 使用脚本内的默认评估名称和路径，批量识别并绘制 CSV 和 NetCDF：

    python plot_openbench_dams_rivers.py

2. 指定 OpenBench 评估名称，使用由该名称推导出的默认输入、输出路径：

    python plot_openbench_dams_rivers.py --evaluation-name Amazon_inundation_D

3. 绘制单个站点 CSV 或网格 NetCDF，并指定输出文件：

    python plot_openbench_dams_rivers.py \
        --evaluation /path/to/metric__anomaly.csv \
        --output /path/to/metric__dams_rivers.png

4. 从指定的 metrics、scores 或 ``Diff_Plot`` 目录批量绘图：

    python plot_openbench_dams_rivers.py \
        --evaluation-dir /path/to/comparisons/Diff_Plot \
        --output-dir /path/to/output_directory

输入 CSV 需包含站点经纬度，或包含可从同一 case 原始 evaluation CSV 回查的站点 ID。
输入 NetCDF 需包含一维经纬度坐标和空间二维变量。
如需指定指标列、地图范围、河网精度、水坝年份或关闭流域边界，可使用相应参数。
查看全部参数及默认值：

    python plot_openbench_dams_rivers.py --help
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


def _use_openbench_source_from_cli() -> Path | None:
    """Prefer the OpenBench source tree that owns the requested result path."""
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--evaluation-dir", "--grid-dir", dest="evaluation_dir", type=Path)
    bootstrap.add_argument("--evaluation", "--grid-file", dest="evaluation", type=Path)
    known, _ = bootstrap.parse_known_args()
    result_path = known.evaluation_dir or known.evaluation
    if result_path is None:
        return None
    resolved = result_path.expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        source_root = parent / "src"
        if (source_root / "openbench" / "__init__.py").is_file():
            sys.path.insert(0, str(source_root))
            return source_root
    return None


OPENBENCH_SOURCE_ROOT = _use_openbench_source_from_cli()

from openbench.config.adapter import build_fig_nml
from openbench.core.registry import IMPLEMENTED_METRICS, IMPLEMENTED_SCORES
from openbench.data.unit import UnitProcessing
from openbench.visualization.Fig_Basic_Plot import colorbar_extend, rounded_limits
from openbench.visualization.Fig_toolbox import convert_unit, get_index, process_unit

from openbench_dams_rivers_common import (
    DEFAULT_DAMS,
    DEFAULT_DAM_BUILT_BEFORE,
    FALLBACK_DAM_BUILT_BEFORE,
    DEFAULT_INPUT_SUFFIX,
    DEFAULT_RIVER_DETAIL,
    DEFAULT_ROUTING_FILE,
    EXTENT_PADDING_FRACTION,
    MIN_EXTENT_PADDING_DEGREES,
    OUTPUT_ROOT,
    RIVER_DETAIL_CONFIG,
    dam_marker_sizes,
    dam_built_before_from_manifest,
    filter_dams_by_year,
    filter_dams_to_extent,
    infer_region_name,
    openbench_input_dir,
    region_name_from_evaluation,
    routing_numeric_1d,
    validate_extent,
)

EVALUATION_NAME = "LowerMississippi_inundation_D"
RIVER_FEATURE_SCALES = {
    "all": "10m",
    "fine": "10m",
    "medium": "50m",
    "coarse": "110m",
}
DAM_STORAGE_CLASSES = (
    (0.0, 1000.0, "o", "Dam storage < 1,000 million m³"),
    (1000.0, 10000.0, "s", "Dam storage 1,000–10,000 million m³"),
    (10000.0, np.inf, "^", "Dam storage ≥ 10,000 million m³"),
)
STATION_METADATA_COLUMNS = {
    "ID",
    "lon",
    "lat",
    "ref_lon",
    "ref_lat",
    "use_syear",
    "use_eyear",
    "ref_dir",
    "status",
    "reason",
    "n_models",
}

# Default paths and parameters
DEFAULT_BASIN_CACHE_DIR = Path(
    "/share/home/dq134/wzx/CoLM/dam_analysis/basin_cache"
)
OUTLET_CONFIGS = {
    "missouri": {
        "station": "Hermann",
        "lat": 38.709806,
        "lon": -91.438500,
        "grdc_id": "4122900",
        "other_id": "USGS 06934500",
    },
    "lower mississippi": {
        "station": "Tarbert Landing",
        "lat": 31.008512,
        "lon": -91.623729,
        "grdc_id": "4127930",
        "other_id": "USGS 07295100",
        "auxiliary": "Vicksburg (GRDC 4127800; USGS 07289000)",
    },
    "amazon": {
        "station": "Obidos",
        "lat": -1.950000,
        "lon": -55.510000,
        "grdc_id": "3629000",
        "other_id": "",
    },
    "mekong": {
        "station": "Stung Treng",
        "lat": 13.533000,
        "lon": 105.945000,
        "grdc_id": "2569005",
        "other_id": "MRC 014501",
        "auxiliary": "Pakse (GRDC 2469260; MRC 013901)",
    },
}


BASIN_MASK_ROOT = Path(
    "/share/home/dq134/wzx/openbench/postprocess/basin_mask/outputs/basin_masks_06min"
)
BASIN_MASKS_BY_REGION = {
    "Amazon": BASIN_MASK_ROOT / "amazon/amazon_basin.nc",
    "Lower Mississippi": BASIN_MASK_ROOT / "lower_mississippi/lower_mississippi_basin.nc",
    "Missouri": BASIN_MASK_ROOT / "missouri/missouri_basin.nc",
    "Mekong": BASIN_MASK_ROOT / "mekong/mekong_basin.nc",
}


@dataclass(frozen=True)
class GridData:
    longitude: np.ndarray
    latitude: np.ndarray
    values: np.ndarray
    variable: str
    attrs: dict[str, object]


@dataclass(frozen=True)
class BasinMask:
    data: xr.DataArray
    extent: tuple[float, float, float, float]
    path: Path


@dataclass(frozen=True)
class RiverNetwork:
    segments: np.ndarray
    detail: str
    minimum_uparea_km2: float | None
    linewidth: float
    source_path: Path
    candidate_count: int

    @property
    def legend_label(self) -> str:
        if self.minimum_uparea_km2 is None:
            return "CoLM river network (all)"
        threshold = f"{self.minimum_uparea_km2:,.0f}"
        return f"CoLM river network ({self.detail}, ≥{threshold} km²)"


@dataclass(frozen=True)
class ResultMetadata:
    item: str
    reference: str
    simulations: tuple[str, ...]
    measure: str
    measure_type: str
    product_type: str
    data_type: str
    unit: str
    section: str


@dataclass(frozen=True)
class OpenBenchPlotStyle:
    option: dict[str, object]
    cmap: object
    ticks: np.ndarray
    norm: object
    levels: np.ndarray
    extend: str
    title: str
    colorbar_label: str


def _case_root(path: Path) -> Path | None:
    for parent in (path.parent, *path.parents):
        if (parent / "run_manifest.json").is_file():
            return parent
    return None


def _manifest_document(path: Path) -> dict[str, object]:
    root = _case_root(path)
    if root is None:
        return {}
    try:
        with (root / "run_manifest.json").open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, ValueError):
        return {}
    return manifest if isinstance(manifest, dict) else {}


def _run_manifest(path: Path) -> dict[str, object]:
    config = _manifest_document(path).get("canonical_config", {})
    return config if isinstance(config, dict) else {}


def _manifest_display_unit(
    path: Path,
    item: str,
    reference: str,
    simulations: tuple[str, ...],
    product_type: str,
) -> str:
    runtime = _manifest_document(path).get("resolved_runtime", {})
    namelists = runtime.get("namelists", {}) if isinstance(runtime, dict) else {}
    raw_unit = ""
    if product_type in {"anomaly", "difference"} and simulations:
        simulation = namelists.get("simulation", {}) if isinstance(namelists, dict) else {}
        item_config = simulation.get(item, {}) if isinstance(simulation, dict) else {}
        if isinstance(item_config, dict):
            raw_unit = str(item_config.get(f"{simulations[0]}_varunit") or "")
    if not raw_unit:
        references = namelists.get("reference", {}) if isinstance(namelists, dict) else {}
        item_config = references.get(item, {}) if isinstance(references, dict) else {}
        if isinstance(item_config, dict):
            raw_unit = str(item_config.get(f"{reference}_varunit") or "")
    if not raw_unit:
        return ""
    if "evapotranspiration" in item.lower():
        return convert_unit("mm day-1")
    return convert_unit(UnitProcessing.display_unit(raw_unit, item))


def _manifest_names(manifest: dict[str, object], key: str) -> tuple[str, ...]:
    values = manifest.get(key, [])
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values)


def _result_section(path: Path) -> str:
    parts = {part.lower(): part for part in path.parts}
    if "metrics" in parts:
        return "metrics"
    if "scores" in parts:
        return "scores"
    if "diff_plot" in parts:
        return "Diff_Plot"
    return path.parent.name


def _strip_product_suffix(name: str) -> tuple[str, str]:
    for suffix, product in (
        ("_ensemble_mean", "ensemble_mean"),
        ("_anomaly", "anomaly"),
        ("_diff", "difference"),
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)], product
    return name, "raw"


def _first_manifest_match(text: str, candidates: tuple[str, ...]) -> str:
    matches = [candidate for candidate in candidates if candidate in text]
    return max(matches, key=len) if matches else ""


def infer_result_metadata(
    path: Path,
    value_name: str,
    attrs: dict[str, object] | None,
    data_type: str,
) -> ResultMetadata:
    attrs = attrs or {}
    manifest = _run_manifest(path)
    section = _result_section(path)
    measure, product_type = _strip_product_suffix(value_name)
    parts = path.stem.split("__")
    stem = path.stem

    item = str(attrs.get("item") or "")
    references = ()
    evaluation = manifest.get("evaluation", {})
    if isinstance(evaluation, dict):
        references_config = manifest.get("reference", {})
        variables = evaluation.get("variables", [])
        items = tuple(str(value) for value in variables) if isinstance(variables, list) else ()
    else:
        references_config, items = {}, ()
    if not item:
        item = parts[0] if len(parts) > 1 else _first_manifest_match(stem, items)
    if not item:
        item = stem.split("_stn_", 1)[0].split("_ref_", 1)[0]

    reference = str(attrs.get("ref_source") or "")
    if isinstance(references_config, dict):
        sources = references_config.get("sources", {})
        if isinstance(sources, dict):
            source = sources.get(item)
            if isinstance(source, str):
                references = (source,)
            elif isinstance(source, list):
                references = tuple(str(value) for value in source)
    if not reference and "ref" in parts:
        index = parts.index("ref")
        reference = parts[index + 1] if index + 1 < len(parts) else ""
    if not reference and "stn" in parts:
        index = parts.index("stn")
        reference = parts[index + 1] if index + 1 < len(parts) else ""
    if not reference:
        reference = _first_manifest_match(stem, references)

    simulations_config = manifest.get("simulation", {})
    known_simulations = tuple(str(value) for value in simulations_config) if isinstance(simulations_config, dict) else ()
    simulations: list[str] = []
    attr_simulation = str(attrs.get("sim_source") or "")
    if attr_simulation:
        simulations.append(attr_simulation)
    if "sim" in parts:
        index = parts.index("sim")
        if index + 1 < len(parts):
            simulations.append(parts[index + 1])
    if "vs" in parts:
        index = parts.index("vs")
        if data_type == "grid" and index >= 1:
            simulations.append(parts[index - 1])
        elif data_type == "station" and "stn" in parts:
            stn_index = parts.index("stn")
            if stn_index + 2 < len(parts):
                simulations.append(parts[stn_index + 2])
        if index + 1 < len(parts):
            simulations.append(parts[index + 1])
    for simulation in known_simulations:
        if simulation in stem:
            simulations.append(simulation)
    simulations = list(dict.fromkeys(value for value in simulations if value))

    metrics = _manifest_names(manifest, "metrics") or tuple(IMPLEMENTED_METRICS)
    scores = _manifest_names(manifest, "scores") or tuple(IMPLEMENTED_SCORES)
    attr_measure = attrs.get("metric") or attrs.get("score")
    if attr_measure:
        measure = str(attr_measure)
    if measure in scores or section == "scores":
        measure_type = "score"
    elif measure in metrics or section == "metrics":
        measure_type = "metric"
    else:
        measure_type = "score" if measure in IMPLEMENTED_SCORES else "metric"

    if product_type == "raw":
        if "ensemble_mean" in stem:
            product_type = "ensemble_mean"
        elif "anomaly" in stem:
            product_type = "anomaly"
        elif "diff" in stem or "vs" in parts:
            product_type = "difference"

    unit = "" if measure_type == "score" else str(attrs.get("units") or "")
    if measure_type == "metric" and not unit:
        unit = _manifest_display_unit(
            path, item, reference, tuple(simulations), product_type
        )
    return ResultMetadata(
        item=item,
        reference=reference,
        simulations=tuple(simulations),
        measure=measure,
        measure_type=measure_type,
        product_type=product_type,
        data_type=data_type,
        unit=unit,
        section=section,
    )


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("The selected OpenBench result contains no finite values.")
    return float(finite.min()), float(finite.max())


def _openbench_limits(values: np.ndarray, metadata: ResultMetadata, option: dict[str, object]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if option.get("vmin_max_on"):
        return float(option["vmin"]), float(option["vmax"])
    measure = metadata.measure
    if metadata.product_type in {"anomaly", "difference"}:
        if measure in {
            "bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF", "NSE", "KGE",
            "KGESS", "correlation", "kappa_coeff", "rSpearman",
        }:
            low, high = np.quantile(finite, [0.05, 0.95])
            low, high = math.floor(float(low)), math.ceil(float(high))
            if measure == "percent_bias":
                low, high = max(low, -100), min(high, 100)
            return float(low), float(high)
        return _finite_range(finite)
    if metadata.measure_type == "score":
        return 0.0, 1.0

    low, high = np.quantile(finite, [0.05, 0.95])
    if measure in {"bias", "percent_bias", "rSD", "PBIAS_HF", "PBIAS_LF"}:
        low, high = rounded_limits(float(low), float(high))
        if measure == "percent_bias":
            low, high = max(low, -100), min(high, 100)
        return float(low), float(high)
    if measure in {"NSE", "KGE", "KGESS", "KGEln", "dr", "cp", "APFB", "correlation", "kappa_coeff", "rSpearman"}:
        return -1.0, 1.0
    if measure in {"LNSE", "ubNSE", "rNSE", "wNSE", "wsNSE"}:
        return float(math.floor(low)), 1.0
    if measure in {"RMSE", "CRMSD", "MSE", "ubRMSE", "nRMSE", "mean_absolute_error", "ssq", "ve", "absolute_percent_bias"}:
        return 0.0, float(rounded_limits(0.0, float(high))[1])
    return 0.0, 1.0


def resolve_openbench_plot_style(
    values: np.ndarray,
    metadata: ResultMetadata,
    args: argparse.Namespace,
) -> OpenBenchPlotStyle:
    figures = build_fig_nml()
    if metadata.product_type in {"anomaly", "difference"}:
        option = figures["Comparison"]["Diff_Plot"].copy()
        if not option.get("cmap"):
            option["cmap"] = "RdBu_r"
    else:
        key = "make_stn_plot_index" if metadata.data_type == "station" else "make_geo_plot_index"
        option = figures[key].copy()

    if args.color_limit is not None:
        vmin, vmax = -float(args.color_limit), float(args.color_limit)
    else:
        vmin, vmax = _openbench_limits(values, metadata, option)
    if not np.isfinite([vmin, vmax]).all() or vmin >= vmax:
        center = float(vmin) if np.isfinite(vmin) else 0.0
        vmin, vmax = center - 0.5, center + 0.5

    index_name = "" if metadata.product_type in {"anomaly", "difference"} else metadata.measure
    cmap, ticks, norm, levels, extend = get_index(vmin, vmax, option.get("cmap", "Spectral"), index_name)
    actual_min, actual_max = _finite_range(values)
    if metadata.product_type not in {"anomaly", "difference"}:
        extend = colorbar_extend(actual_min, actual_max, float(ticks[0]), float(ticks[-1]))
    option.update({"vmin": float(ticks[0]), "vmax": float(ticks[-1]), "extend": extend})

    title = str(option.get("title") or "")
    if metadata.product_type == "anomaly":
        simulation = metadata.simulations[0] if metadata.simulations else ""
        title = f"{metadata.item} {metadata.measure} anomaly for {simulation}".strip()
    elif metadata.product_type == "difference":
        simulations = metadata.simulations[:2]
        comparison = " vs ".join(simulations)
        title = f"{metadata.item} {metadata.measure} difference {comparison}".strip()

    if metadata.measure_type == "score":
        colorbar_label = metadata.measure.replace("_", "\n")
    else:
        unit_label = process_unit(metadata.unit, metadata.unit, metadata.measure)
        colorbar_label = metadata.measure.replace("_", "\n") + "\n" + unit_label
    if metadata.product_type in {"anomaly", "difference"} and metadata.measure_type == "metric":
        colorbar_label = process_unit(metadata.unit, "", metadata.measure)

    return OpenBenchPlotStyle(
        option=option,
        cmap=cmap,
        ticks=np.asarray(ticks, dtype=float),
        norm=norm,
        levels=np.asarray(levels, dtype=float),
        extend=extend,
        title=title,
        colorbar_label=colorbar_label,
    )


def _add_overlay_title(
    style: OpenBenchPlotStyle,
    metadata: ResultMetadata,
    region_name: str | None,
    dam_year_filter_label: str | None,
) -> OpenBenchPlotStyle:
    if style.title:
        return style
    region_prefix = f"{region_name} " if region_name else ""
    measure = metadata.measure.replace("_", " ")
    product = " Ensemble Mean" if metadata.product_type == "ensemble_mean" else ""
    title = f"{region_prefix}{metadata.item} {measure}{product}, River Network, and Dams"
    details = []
    if metadata.reference:
        details.append(f"Reference: {metadata.reference}")
    if metadata.simulations:
        label = "Simulation" if len(metadata.simulations) == 1 else "Simulations"
        details.append(f"{label}: {' vs '.join(metadata.simulations)}")
    if dam_year_filter_label:
        details.append(dam_year_filter_label)
    if details:
        title += "\n" + "  |  ".join(details)
    return replace(style, title=title)


def read_grid(path: Path, requested_variable: str | None = None) -> GridData:
    if not path.is_file():
        raise FileNotFoundError(f"Grid file not found: {path}")

    with xr.open_dataset(path) as dataset:
        missing_coordinates = {"lat", "lon"}.difference(dataset.coords)
        if missing_coordinates:
            raise ValueError(
                f"Grid NetCDF is missing coordinates: {sorted(missing_coordinates)}"
            )
        if dataset["lat"].ndim != 1 or dataset["lon"].ndim != 1:
            raise ValueError("Coordinates 'lat' and 'lon' must both be one-dimensional.")

        if requested_variable is None:
            candidates = [
                name
                for name, variable in dataset.data_vars.items()
                if variable.ndim == 2 and set(variable.dims) == {"lat", "lon"}
            ]
            if len(candidates) > 1:
                filename_matches = [name for name in candidates if name in path.stem]
                if len(filename_matches) == 1:
                    candidates = filename_matches
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one two-dimensional lat/lon data variable in {path}, "
                    f"found {candidates}. Select one with --value-variable."
                )
            variable_name = candidates[0]
        else:
            variable_name = requested_variable
            if variable_name not in dataset.data_vars:
                raise ValueError(
                    f"Requested variable {variable_name!r} is not present in {path}. "
                    f"Available variables: {list(dataset.data_vars)}"
                )

        variable = dataset[variable_name]
        if variable.ndim != 2 or set(variable.dims) != {"lat", "lon"}:
            raise ValueError(
                f"Variable {variable_name!r} must be two-dimensional with lat/lon dimensions; "
                f"found dimensions {variable.dims}."
            )
        longitude = np.asarray(dataset["lon"].values, dtype=float)
        latitude = np.asarray(dataset["lat"].values, dtype=float)
        values = np.asarray(variable.transpose("lat", "lon").values, dtype=float)
        attrs = dict(variable.attrs)

    _validate_coordinate(longitude, "lon", -180.0, 180.0)
    _validate_coordinate(latitude, "lat", -90.0, 90.0)
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        values = values[:, ::-1]
    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        values = values[::-1, :]
    values[~np.isfinite(values)] = np.nan
    if not np.isfinite(values).any():
        raise ValueError(f"Variable {variable_name!r} in {path} contains no finite grid values.")
    return GridData(longitude, latitude, values, variable_name, attrs)


def _validate_coordinate(
    coordinate: np.ndarray,
    name: str,
    minimum: float,
    maximum: float,
) -> None:
    if coordinate.size < 2:
        raise ValueError(f"Coordinate {name!r} must contain at least two values.")
    if not np.isfinite(coordinate).all():
        raise ValueError(f"Coordinate {name!r} contains non-finite values.")
    if np.any((coordinate < minimum) | (coordinate > maximum)):
        raise ValueError(f"Coordinate {name!r} must stay within [{minimum}, {maximum}].")
    differences = np.diff(coordinate)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"Coordinate {name!r} must be strictly monotonic.")


def coordinate_edges(coordinate: np.ndarray) -> np.ndarray:
    midpoints = (coordinate[:-1] + coordinate[1:]) / 2.0
    first = coordinate[0] - (coordinate[1] - coordinate[0]) / 2.0
    last = coordinate[-1] + (coordinate[-1] - coordinate[-2]) / 2.0
    return np.concatenate(([first], midpoints, [last]))


def _normalized_longitudes(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0


def load_basin_mask(path: Path) -> BasinMask:
    if not path.is_file():
        raise FileNotFoundError(f"Basin mask file not found: {path}")
    with xr.open_dataset(path) as dataset:
        if "basin_mask" not in dataset.data_vars:
            raise ValueError(f"Basin mask file does not contain a 'basin_mask' variable: {path}")
        source = dataset["basin_mask"]
        if source.ndim != 2 or set(source.dims) != {"lat", "lon"}:
            raise ValueError("Basin mask must be two-dimensional with lat/lon dimensions.")
        if "lat" not in source.coords or "lon" not in source.coords:
            raise ValueError("Basin mask must provide lat/lon coordinates.")
        if source["lat"].ndim != 1 or source["lon"].ndim != 1:
            raise ValueError("Basin mask lat/lon coordinates must be one-dimensional.")
        mask = (source.fillna(0) > 0).load()

    mask = mask.assign_coords(lon=_normalized_longitudes(mask["lon"].values))
    if np.unique(mask["lat"].values).size != mask.sizes["lat"]:
        raise ValueError("Basin mask latitude coordinates must be unique.")
    if np.unique(mask["lon"].values).size != mask.sizes["lon"]:
        raise ValueError("Basin mask longitude coordinates must be unique after normalization.")
    mask = mask.sortby("lat").sortby("lon")
    active = mask.where(mask, drop=True)
    if active.size == 0:
        raise ValueError(f"Basin mask contains no positive cells: {path}")

    lon_spacing = float(np.median(np.diff(mask["lon"].values)))
    lat_spacing = float(np.median(np.diff(mask["lat"].values)))
    extent = (
        float(active["lon"].min()) - abs(lon_spacing) / 2.0,
        float(active["lon"].max()) + abs(lon_spacing) / 2.0,
        float(active["lat"].min()) - abs(lat_spacing) / 2.0,
        float(active["lat"].max()) + abs(lat_spacing) / 2.0,
    )
    return BasinMask(mask, extent, path)


def basin_mask_for_grid(basin_mask: BasinMask, grid: GridData) -> np.ndarray:
    lon_edges = coordinate_edges(grid.longitude)
    lat_edges = coordinate_edges(grid.latitude)
    west, east, south, north = basin_mask.extent
    if (
        lon_edges[0] > west + 1.0e-6
        or lon_edges[-1] < east - 1.0e-6
        or lat_edges[0] > south + 1.0e-6
        or lat_edges[-1] < north - 1.0e-6
    ):
        raise ValueError(
            "Grid extent does not cover the complete OpenBench basin mask: "
            f"grid=({lon_edges[0]}, {lon_edges[-1]}, {lat_edges[0]}, {lat_edges[-1]}), "
            f"mask={basin_mask.extent}"
        )
    mapped = basin_mask.data.astype(np.uint8).interp(
        lat=xr.DataArray(grid.latitude, dims="lat"),
        lon=xr.DataArray(_normalized_longitudes(grid.longitude), dims="lon"),
        method="nearest",
    )
    return np.asarray(mapped.fillna(0).values, dtype=bool)


def apply_basin_mask(grid: GridData, basin_mask: BasinMask) -> GridData:
    inside = basin_mask_for_grid(basin_mask, grid)
    values = np.where(inside, grid.values, np.nan)
    if not np.isfinite(values).any():
        raise ValueError(
            f"Variable {grid.variable!r} has no finite values inside basin mask {basin_mask.path}."
        )
    return GridData(grid.longitude, grid.latitude, values, grid.variable, grid.attrs)


def filter_points_to_basin(points: pd.DataFrame, basin_mask: BasinMask) -> pd.DataFrame:
    if points.empty:
        return points.copy()
    inside = points_inside_basin(
        points["lon"].to_numpy(float),
        points["lat"].to_numpy(float),
        basin_mask,
    )
    return points.loc[inside].copy()


def points_inside_basin(
    longitude: np.ndarray,
    latitude: np.ndarray,
    basin_mask: BasinMask,
) -> np.ndarray:
    selected = basin_mask.data.sel(
        lon=xr.DataArray(_normalized_longitudes(longitude), dims="point"),
        lat=xr.DataArray(np.asarray(latitude, dtype=float), dims="point"),
        method="nearest",
    )
    return np.asarray(selected.values, dtype=bool)


def load_river_network(
    routing_path: Path,
    detail: str,
    extent: tuple[float, float, float, float],
    basin_mask: BasinMask | None,
) -> RiverNetwork | None:
    config = RIVER_DETAIL_CONFIG[detail]
    if detail == "none":
        return None
    if not routing_path.is_file():
        raise FileNotFoundError(f"Routing file not found: {routing_path}")

    with h5py.File(routing_path, "r") as h5:
        sequence_raw = routing_numeric_1d(h5, "seq")
        next_raw = routing_numeric_1d(h5, "seq_next")
        longitude = routing_numeric_1d(h5, "seq_lon")
        latitude = routing_numeric_1d(h5, "seq_lat")
        upstream_area = routing_numeric_1d(h5, "topo_uparea")
        units = h5["topo_uparea"].attrs.get("units", b"")

    lengths = {
        len(sequence_raw),
        len(next_raw),
        len(longitude),
        len(latitude),
        len(upstream_area),
    }
    if len(lengths) != 1:
        raise ValueError(
            "Routing variables seq, seq_next, seq_lon, seq_lat, and topo_uparea "
            "must have the same length."
        )
    if isinstance(units, bytes):
        units = units.decode("utf-8", errors="replace")
    if str(units).strip().lower() not in {"m2", "m^2", "m²"}:
        raise ValueError(f"Routing variable topo_uparea must use square metres; found units={units!r}.")

    valid_sequence = np.isfinite(sequence_raw) & (sequence_raw > 0)
    sequence_indices = np.flatnonzero(valid_sequence)
    sequence = sequence_raw[sequence_indices].astype(np.int64)
    order = np.argsort(sequence)
    sorted_sequence = sequence[order]
    if np.any(np.diff(sorted_sequence) == 0):
        raise ValueError("Routing variable seq contains duplicate positive sequence IDs.")
    sorted_indices = sequence_indices[order]

    destination = np.full(len(sequence_raw), -1, dtype=np.int64)
    finite_next = np.flatnonzero(np.isfinite(next_raw) & (next_raw > 0))
    next_sequence = next_raw[finite_next].astype(np.int64)
    positions = np.searchsorted(sorted_sequence, next_sequence)
    in_range = positions < len(sorted_sequence)
    matched_sources = finite_next[in_range]
    matched_positions = positions[in_range]
    matches = sorted_sequence[matched_positions] == next_sequence[in_range]
    destination[matched_sources[matches]] = sorted_indices[matched_positions[matches]]

    source_indices = np.arange(len(sequence_raw))
    candidate = destination >= 0
    candidate &= destination != source_indices
    safe_destination = np.maximum(destination, 0)
    candidate &= np.isfinite(longitude) & np.isfinite(latitude)
    candidate &= np.isfinite(longitude[safe_destination]) & np.isfinite(latitude[safe_destination])
    candidate &= np.isfinite(upstream_area) & (upstream_area >= 0)

    minimum_uparea_km2 = config["minimum_uparea_km2"]
    if minimum_uparea_km2 is not None:
        candidate &= upstream_area >= float(minimum_uparea_km2) * 1_000_000.0
    candidate_count = int(candidate.sum())

    west, east, south, north = extent
    destination_longitude = longitude[safe_destination]
    destination_latitude = latitude[safe_destination]
    candidate &= (
        (longitude >= west)
        & (longitude <= east)
        & (latitude >= south)
        & (latitude <= north)
        & (destination_longitude >= west)
        & (destination_longitude <= east)
        & (destination_latitude >= south)
        & (destination_latitude <= north)
    )

    selected_sources = np.flatnonzero(candidate)
    selected_destinations = destination[selected_sources]
    if basin_mask is not None and selected_sources.size:
        source_inside = points_inside_basin(
            longitude[selected_sources], latitude[selected_sources], basin_mask
        )
        destination_inside = points_inside_basin(
            longitude[selected_destinations], latitude[selected_destinations], basin_mask
        )
        inside = source_inside & destination_inside
        selected_sources = selected_sources[inside]
        selected_destinations = selected_destinations[inside]

    if selected_sources.size == 0:
        raise ValueError(
            f"No valid CoLM river segments remain for detail={detail!r}, extent={extent}."
        )
    segments = np.stack(
        (
            np.column_stack((longitude[selected_sources], latitude[selected_sources])),
            np.column_stack((longitude[selected_destinations], latitude[selected_destinations])),
        ),
        axis=1,
    )
    network = RiverNetwork(
        segments=segments,
        detail=detail,
        minimum_uparea_km2=minimum_uparea_km2,
        linewidth=float(config["linewidth"]),
        source_path=routing_path,
        candidate_count=candidate_count,
    )
    threshold_label = "all valid areas" if minimum_uparea_km2 is None else f"≥{minimum_uparea_km2:,.0f} km²"
    print(
        f"Loaded CoLM river network: {routing_path} "
        f"(detail={detail}, threshold={threshold_label}, candidates={candidate_count:,}, "
        f"segments={len(segments):,})"
    )
    return network


def draw_basin_mask_boundary(ax, basin_mask: BasinMask, projection) -> None:
    mask = basin_mask.data
    active_rows = np.flatnonzero(np.asarray(mask.any(dim="lon").values, dtype=bool))
    active_columns = np.flatnonzero(np.asarray(mask.any(dim="lat").values, dtype=bool))
    row_slice = slice(max(0, active_rows[0] - 1), min(mask.sizes["lat"], active_rows[-1] + 2))
    column_slice = slice(max(0, active_columns[0] - 1), min(mask.sizes["lon"], active_columns[-1] + 2))
    cropped = mask.isel(lat=row_slice, lon=column_slice)
    ax.contour(
        cropped["lon"].values,
        cropped["lat"].values,
        np.asarray(cropped.values, dtype=float),
        levels=[0.5],
        colors=["black"],
        linewidths=1.2,
        transform=projection,
        zorder=4.5,
    )


def automatic_extent_from_grids(
    paths: list[Path],
    requested_variable: str | None = None,
) -> tuple[float, float, float, float]:
    west = np.inf
    east = -np.inf
    south = np.inf
    north = -np.inf
    for path in paths:
        grid = read_grid(path, requested_variable)
        lon_edges = coordinate_edges(grid.longitude)
        lat_edges = coordinate_edges(grid.latitude)
        west = min(west, float(lon_edges[0]))
        east = max(east, float(lon_edges[-1]))
        south = min(south, float(lat_edges[0]))
        north = max(north, float(lat_edges[-1]))
    lon_padding = max((east - west) * EXTENT_PADDING_FRACTION, MIN_EXTENT_PADDING_DEGREES)
    lat_padding = max((north - south) * EXTENT_PADDING_FRACTION, MIN_EXTENT_PADDING_DEGREES)
    return (
        max(-180.0, west - lon_padding),
        min(180.0, east + lon_padding),
        max(-90.0, south - lat_padding),
        min(90.0, north + lat_padding),
    )


def read_dams(path: Path) -> pd.DataFrame:
    dams = pd.read_csv(path)
    required = {"lon", "lat", "TotalVol_mcm", "year"}
    missing = required.difference(dams.columns)
    if missing:
        raise ValueError(f"Dam CSV is missing columns: {sorted(missing)}")
    dams = dams.copy()
    for column in required:
        dams[column] = pd.to_numeric(dams[column], errors="coerce")
    dams = dams.dropna(subset=["lon", "lat"])
    if dams.empty:
        raise ValueError("No valid dams remain after filtering invalid coordinates.")
    return dams


def _add_openbench_colorbar(fig, ax, mappable, style: OpenBenchPlotStyle):
    option = style.option
    orientation = str(option.get("colorbar_position", "vertical"))
    if option.get("colorbar_position_set"):
        bounds = [
            float(option["colorbar_left"]),
            float(option["colorbar_bottom"]),
            float(option["colorbar_width"]),
            float(option["colorbar_height"]),
        ]
    else:
        left, bottom, width, height = ax.get_position().bounds
        if orientation == "horizontal":
            offset = 0.15 if option.get("xticklabel") else 0.1
            bounds = [left + width / 8.0, bottom - offset, width * 0.75, 0.03]
        else:
            bounds = [left + width + 0.01, bottom, 0.015, height]
    colorbar = fig.colorbar(
        mappable,
        cax=fig.add_axes(bounds),
        ticks=style.ticks,
        spacing="uniform",
        extend=style.extend,
        orientation=orientation,
    )
    colorbar.set_label(
        style.colorbar_label,
        rotation=0,
        fontsize=16,
        weight="bold",
        labelpad=10,
        ha="left",
        va="bottom",
    )
    colorbar.ax.tick_params(labelsize=float(option["xtick"]))
    colorbar.solids.set_edgecolor("face")
    return colorbar


def plot_grid_map(
    grid: GridData,
    dams: pd.DataFrame,
    grid_path: Path,
    output_path: Path,
    extent: tuple[float, float, float, float],
    style: OpenBenchPlotStyle,
    dpi: int,
    region_name: str | None,
    dam_year_filter_label: str | None,
    basin_mask: BasinMask | None,
    river_network: RiverNetwork | None,
) -> None:
    option = style.option
    matplotlib.rc("font", family=option["font"])
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(float(option["x_wise"]), float(option["y_wise"])))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1ec", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#ffffff", zorder=0)

    grid_values = np.ma.masked_invalid(grid.values)
    if option.get("show_method") == "interpolate":
        grid_layer = ax.contourf(
            grid.longitude,
            grid.latitude,
            grid_values,
            levels=style.levels,
            cmap=style.cmap,
            norm=style.norm,
            extend=style.extend,
            transform=projection,
            zorder=1.2,
        )
    else:
        grid_layer = ax.pcolormesh(
            coordinate_edges(grid.longitude),
            coordinate_edges(grid.latitude),
            grid_values,
            cmap=style.cmap,
            norm=style.norm,
            shading="flat",
            transform=projection,
            zorder=1.2,
        )

    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#e6f2f8",
        edgecolor="#77a9c3",
        linewidth=0.35,
        zorder=2.4,
    )
    if river_network is not None:
        river_lines = LineCollection(
            river_network.segments,
            colors="#377fa8",
            linewidths=river_network.linewidth,
            alpha=0.9,
            transform=projection,
            zorder=3,
        )
        ax.add_collection(river_lines)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="0.35", linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="0.2", linewidth=0.75, zorder=3)
    if basin_mask is not None:
        draw_basin_mask_boundary(ax, basin_mask, projection)

    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.45,
        color="0.55",
        alpha=0.55,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 10}
    gridlines.ylabel_style = {"size": 10}
    gridlines.xlocator = MaxNLocator(7)
    gridlines.ylocator = MaxNLocator(6)

    year_points = None
    if not dams.empty:
        dam_sizes = dam_marker_sizes(dams["TotalVol_mcm"])
        storage_values = pd.to_numeric(
            dams["TotalVol_mcm"], errors="coerce"
        ).fillna(0).clip(lower=0).to_numpy(float)
        valid_year = dams["year"].between(1800, 2030)
        if valid_year.any():
            year_min = float(dams.loc[valid_year, "year"].min())
            year_max = float(dams.loc[valid_year, "year"].max())
            if year_min == year_max:
                year_min -= 0.5
                year_max += 0.5
            year_norm = Normalize(vmin=year_min, vmax=year_max)
            for lower, upper, marker, _ in DAM_STORAGE_CLASSES:
                selected = (
                    valid_year.to_numpy()
                    & (storage_values >= lower)
                    & (storage_values < upper)
                )
                if not selected.any():
                    continue
                points = ax.scatter(
                    dams.loc[selected, "lon"],
                    dams.loc[selected, "lat"],
                    c=dams.loc[selected, "year"],
                    s=dam_sizes[selected],
                    cmap="viridis",
                    norm=year_norm,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.96,
                    transform=projection,
                    zorder=5,
                )
                if year_points is None:
                    year_points = points
        invalid_year = (~valid_year).to_numpy()
        for lower, upper, marker, _ in DAM_STORAGE_CLASSES:
            selected = (
                invalid_year
                & (storage_values >= lower)
                & (storage_values < upper)
            )
            if selected.any():
                ax.scatter(
                    dams.loc[selected, "lon"],
                    dams.loc[selected, "lat"],
                    s=dam_sizes[selected],
                    color="0.65",
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    transform=projection,
                    zorder=5,
                )

    if year_points is not None:
        year_bar = fig.colorbar(
            year_points,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            shrink=0.67,
            aspect=30,
        )
        year_bar.set_label("Dam construction year", fontsize=11)
        year_bar.ax.tick_params(labelsize=9)

    _add_openbench_colorbar(fig, ax, grid_layer, style)

    legend_handles: list[object] = []
    if basin_mask is not None:
        legend_handles.append(
            Line2D([], [], color="black", linewidth=1.2, label="OpenBench basin-mask boundary"),
        )
    if river_network is not None:
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#377fa8",
                linewidth=max(1.0, river_network.linewidth),
                label=river_network.legend_label,
            )
        )
    if not dams.empty:
        for _, _, marker, label in DAM_STORAGE_CLASSES:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="none",
                    markerfacecolor="#5ec962",
                    markeredgecolor="black",
                    markersize=8,
                    label=label,
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.7",
        fontsize=8.8,
        title="Map layers and dam-size scale",
        title_fontsize=9.2,
        borderpad=0.7,
    )

    ax.set_xlabel(str(option.get("xticklabel", "")), fontsize=float(option["xtick"]) + 1)
    ax.set_ylabel(str(option.get("yticklabel", "")), fontsize=float(option["ytick"]) + 1)
    ax.set_title(
        style.title,
        fontsize=float(option["title_size"]),
        fontweight="bold",
        pad=14,
    )
    river_summary = (
        f"CoLM rivers: {len(river_network.segments):,} segments ({river_network.detail})"
        if river_network is not None
        else "CoLM rivers: none"
    )
    fig.text(
        0.5,
        0.045,
        f"Additional layers — {river_summary}  |  Triangles: dams "
        f"(n={len(dams)}, size = log-scaled total storage)",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_grid_file(
    grid_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    dams: pd.DataFrame,
    extent: tuple[float, float, float, float],
    basin_mask: BasinMask | None,
    river_network: RiverNetwork | None,
) -> None:
    grid = read_grid(grid_path, args.value_variable)
    if basin_mask is not None:
        grid = apply_basin_mask(grid, basin_mask)
    visible_dams, dam_year_filter_label = filter_dams_by_year(
        dams,
        args.dam_built_before,
        args.dam_built_by,
    )
    visible_dams = filter_dams_to_extent(visible_dams, extent)
    if basin_mask is not None:
        visible_dams = filter_points_to_basin(visible_dams, basin_mask)
    metadata = infer_result_metadata(grid_path, grid.variable, grid.attrs, "grid")
    style = resolve_openbench_plot_style(grid.values, metadata, args)
    region_name = infer_region_name(grid_path, args.region_name)
    style = _add_overlay_title(
        style, metadata, region_name, dam_year_filter_label
    )
    dpi = args.dpi if args.dpi is not None else int(style.option["dpi"])
    plot_grid_map(
        grid,
        visible_dams,
        grid_path,
        output_path,
        extent,
        style,
        dpi,
        region_name,
        dam_year_filter_label,
        basin_mask,
        river_network,
    )
    valid_count = int(np.isfinite(grid.values).sum())
    print(
        f"Wrote {output_path} "
        f"(variable={grid.variable}, product={metadata.product_type}, valid_cells={valid_count}, "
        f"dams={len(visible_dams)}, rivers={0 if river_network is None else len(river_network.segments)}, "
        f"river_detail={args.river_detail}, extent={tuple(round(value, 6) for value in extent)})"
    )


def evaluation_output(evaluation_name: str) -> Path:
    return OUTPUT_ROOT / f"{evaluation_name}_dams_rivers_metric_diff.png"


def evaluation_output_dir(evaluation_name: str) -> Path:
    return OUTPUT_ROOT / evaluation_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay dams, rivers, and basin boundaries on spatial OpenBench results."
    )
    parser.add_argument(
        "--evaluation-name",
        default=EVALUATION_NAME,
        help=f"OpenBench evaluation name used to derive defaults (default: {EVALUATION_NAME}).",
    )
    parser.add_argument(
        "--input-suffix",
        type=Path,
        default=DEFAULT_INPUT_SUFFIX,
        help=f"Input path below the evaluation directory (default: {DEFAULT_INPUT_SUFFIX}).",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--evaluation",
        "--grid-file",
        dest="evaluation",
        type=Path,
        help="One OpenBench station CSV or grid NetCDF instead of the default batch directory.",
    )
    input_group.add_argument(
        "--evaluation-dir",
        "--grid-dir",
        dest="evaluation_dir",
        type=Path,
        help="Batch input directory containing CSV and/or NetCDF files.",
    )
    parser.add_argument("--dams", type=Path, default=DEFAULT_DAMS)
    parser.add_argument("--output", type=Path, help="Output PNG for single-file mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for batch mode (default: derived from --evaluation-name).",
    )
    parser.add_argument(
        "--pattern",
        "--station-pattern",
        dest="station_pattern",
        default="*.csv",
        help="Station CSV filename pattern used in batch mode (default: *.csv).",
    )
    parser.add_argument(
        "--grid-pattern",
        default="*.nc",
        help="Grid NetCDF filename pattern used in batch mode (default: *.nc).",
    )
    parser.add_argument(
        "--value-column",
        help="Station metric column; by default a unique supported metric column is selected.",
    )
    parser.add_argument(
        "--value-variable",
        help="Grid variable; by default the unique two-dimensional lat/lon variable is selected.",
    )
    parser.add_argument(
        "--region-name",
        help="Region label for the title; by default it is inferred from the OpenBench case directory.",
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        help="Manual map extent; by default it is calculated from the OpenBench station coordinates.",
    )
    parser.add_argument(
        "--routing-file",
        type=Path,
        default=DEFAULT_ROUTING_FILE,
        help="CaMa/CoLM routing NetCDF used to delineate the upstream basin.",
    )
    parser.add_argument(
        "--basin-cache-dir",
        type=Path,
        default=DEFAULT_BASIN_CACHE_DIR,
        help="Directory for cached upstream routing cells.",
    )
    parser.add_argument(
        "--basin-mask",
        type=Path,
        help="OpenBench basin-mask NetCDF used for grid data (default: derived from region).",
    )
    parser.add_argument(
        "--no-basin",
        action="store_true",
        help="Do not calculate or draw the basin footprint and boundary.",
    )
    parser.add_argument(
        "--river-detail",
        choices=tuple(RIVER_DETAIL_CONFIG),
        default=DEFAULT_RIVER_DETAIL,
        help=f"River detail level (default: {DEFAULT_RIVER_DETAIL}).",
    )
    parser.add_argument(
        "--color-limit",
        "--kge-limit",
        dest="color_limit",
        type=float,
        help="Manual symmetric color limit; by default OpenBench's native rule is used.",
    )
    dam_year_group = parser.add_mutually_exclusive_group()
    dam_year_group.add_argument(
        "--dam-built-before",
        type=int,
        metavar="YEAR",
        help="Only show dams with a valid construction year strictly before YEAR.",
    )
    dam_year_group.add_argument(
        "--dam-built-by",
        type=int,
        metavar="YEAR",
        help="Only show dams with a valid construction year less than or equal to YEAR.",
    )
    dam_year_group.add_argument(
        "--all-dam-years",
        action="store_true",
        help="Disable the default construction-year filter and show all dams.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="Output DPI override; by default the corresponding OpenBench plotting YAML is used.",
    )
    args = parser.parse_args()
    if args.evaluation is None and args.evaluation_dir is None:
        args.evaluation_dir = openbench_input_dir(args.evaluation_name, args.input_suffix)
    if args.output is None:
        args.output = evaluation_output(args.evaluation_name)
    if args.output_dir is None:
        args.output_dir = evaluation_output_dir(args.evaluation_name)
    if args.dam_built_before is None and args.dam_built_by is None and not args.all_dam_years:
        input_path = args.evaluation if args.evaluation is not None else args.evaluation_dir
        manifest_year, manifest_path = dam_built_before_from_manifest(input_path)
        if manifest_year is not None:
            args.dam_built_before = manifest_year
            print(
                f"Dam construction-year cutoff: < {manifest_year} "
                f"(evaluation end year from {manifest_path})"
            )
        else:
            args.dam_built_before = (
                DEFAULT_DAM_BUILT_BEFORE
                if DEFAULT_DAM_BUILT_BEFORE is not None
                else FALLBACK_DAM_BUILT_BEFORE
            )
            print(
                f"No run_manifest.json found above {input_path}; "
                f"using fallback dam construction-year cutoff "
                f"< {args.dam_built_before}."
            )
    return args


def _attach_station_coordinates(evaluation: pd.DataFrame, path: Path, value_column: str) -> pd.DataFrame:
    if {"lon", "lat"}.issubset(evaluation.columns) or {"ref_lon", "ref_lat"}.issubset(evaluation.columns):
        return evaluation
    if "ID" not in evaluation.columns:
        raise ValueError("non-spatial OpenBench result: no coordinates or station ID")

    metadata = infer_result_metadata(path, value_column, {}, "station")
    case_root = _case_root(path)
    if case_root is None:
        raise ValueError("Cannot recover station coordinates without a nearby run_manifest.json")
    prefix = f"{metadata.item}_stn_"
    if metadata.reference:
        prefix += f"{metadata.reference}_"
    candidates = [
        candidate
        for folder in (case_root / "metrics", case_root / "scores")
        for candidate in folder.glob(f"{prefix}*_evaluations.csv")
    ]
    coordinate_frames = []
    for candidate in candidates:
        header = pd.read_csv(candidate, nrows=0).columns
        if {"ID", "ref_lon", "ref_lat"}.issubset(header):
            frame = pd.read_csv(candidate, usecols=["ID", "ref_lon", "ref_lat"], dtype={"ID": "string"})
            frame = frame.rename(columns={"ref_lon": "lon", "ref_lat": "lat"})
            coordinate_frames.append(frame)
        elif {"ID", "lon", "lat"}.issubset(header):
            coordinate_frames.append(
                pd.read_csv(candidate, usecols=["ID", "lon", "lat"], dtype={"ID": "string"})
            )
    if not coordinate_frames:
        raise ValueError(f"No station evaluation table was found to recover coordinates for {path.name}")

    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    coordinates["ID"] = coordinates["ID"].astype("string")
    coordinates[["lon", "lat"]] = coordinates[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
    conflicts = coordinates.groupby("ID", dropna=False)[["lon", "lat"]].nunique(dropna=True).max(axis=1)
    if (conflicts > 1).any():
        ids = conflicts[conflicts > 1].index.astype(str).tolist()[:5]
        raise ValueError(f"Station IDs have conflicting coordinates: {ids}")
    coordinates = coordinates.dropna(subset=["ID", "lon", "lat"]).drop_duplicates("ID")
    result = evaluation.copy()
    result["ID"] = result["ID"].astype("string")
    result = result.merge(coordinates, on="ID", how="left", validate="many_to_one")
    if result[["lon", "lat"]].isna().all().all():
        raise ValueError("No station IDs could be matched to OpenBench evaluation coordinates")
    return result


def read_inputs(
    evaluation_path: Path,
    dam_path: Path,
    requested_value_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    evaluation = pd.read_csv(evaluation_path, dtype={"ID": "string"})
    dams = pd.read_csv(dam_path)
    evaluation = _attach_station_coordinates(evaluation, evaluation_path, requested_value_column)

    value_column = requested_value_column
    if {"lon", "lat"}.issubset(evaluation.columns):
        pass
    elif {"ref_lon", "ref_lat"}.issubset(evaluation.columns):
        evaluation = evaluation.rename(columns={"ref_lon": "lon", "ref_lat": "lat"})
    else:
        raise ValueError(
            "Evaluation CSV must contain either lon/lat or ref_lon/ref_lat coordinates."
        )

    evaluation_columns = {"lon", "lat", value_column}
    dam_columns = {"lon", "lat", "TotalVol_mcm", "year"}
    missing_evaluation = evaluation_columns.difference(evaluation.columns)
    missing_dams = dam_columns.difference(dams.columns)
    if missing_evaluation:
        raise ValueError(f"Evaluation CSV is missing columns: {sorted(missing_evaluation)}")
    if missing_dams:
        raise ValueError(f"Dam CSV is missing columns: {sorted(missing_dams)}")

    evaluation = evaluation.copy()
    dams = dams.copy()
    for column in evaluation_columns:
        evaluation[column] = pd.to_numeric(evaluation[column], errors="coerce")
    for column in dam_columns:
        dams[column] = pd.to_numeric(dams[column], errors="coerce")

    evaluation = evaluation.dropna(subset=["lon", "lat", value_column])
    dams = dams.dropna(subset=["lon", "lat"])
    if evaluation.empty:
        raise ValueError("No valid evaluation stations remain after filtering.")
    if dams.empty:
        raise ValueError("No valid dams remain after filtering.")
    return evaluation, dams, value_column


def automatic_extent(
    longitude: pd.Series | np.ndarray,
    latitude: pd.Series | np.ndarray,
) -> tuple[float, float, float, float]:
    lon = pd.to_numeric(pd.Series(longitude), errors="coerce").to_numpy(float)
    lat = pd.to_numeric(pd.Series(latitude), errors="coerce").to_numpy(float)
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (lon >= -180.0)
        & (lon <= 180.0)
        & (lat >= -90.0)
        & (lat <= 90.0)
    )
    if not valid.any():
        raise ValueError("No finite OpenBench station coordinates are available to calculate the map extent.")

    west = float(np.min(lon[valid]))
    east = float(np.max(lon[valid]))
    south = float(np.min(lat[valid]))
    north = float(np.max(lat[valid]))
    lon_padding = max((east - west) * EXTENT_PADDING_FRACTION, MIN_EXTENT_PADDING_DEGREES)
    lat_padding = max((north - south) * EXTENT_PADDING_FRACTION, MIN_EXTENT_PADDING_DEGREES)
    return (
        max(-180.0, west - lon_padding),
        min(180.0, east + lon_padding),
        max(-90.0, south - lat_padding),
        min(90.0, north + lat_padding),
    )


def automatic_extent_from_files(
    station_results: list[tuple[Path, str]],
) -> tuple[float, float, float, float]:
    longitude = []
    latitude = []
    for evaluation_path, value_column in station_results:
        evaluation = pd.read_csv(evaluation_path, dtype={"ID": "string"})
        evaluation = _attach_station_coordinates(evaluation, evaluation_path, value_column)
        if {"ref_lon", "ref_lat"}.issubset(evaluation.columns):
            evaluation = evaluation.rename(columns={"ref_lon": "lon", "ref_lat": "lat"})
        longitude.append(evaluation["lon"])
        latitude.append(evaluation["lat"])
    return automatic_extent(pd.concat(longitude, ignore_index=True), pd.concat(latitude, ignore_index=True))


def _haversine_km(
    longitude: np.ndarray,
    latitude: np.ndarray,
    target_lon: float,
    target_lat: float,
) -> np.ndarray:
    lon = np.deg2rad(longitude)
    lat = np.deg2rad(latitude)
    lon0 = np.deg2rad(target_lon)
    lat0 = np.deg2rad(target_lat)
    delta_lon = (lon - lon0 + np.pi) % (2.0 * np.pi) - np.pi
    delta_lat = lat - lat0
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat) * np.cos(lat0) * np.sin(delta_lon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _upstream_cell_indices(
    sequence: np.ndarray,
    next_sequence: np.ndarray,
    outlet_sequence: int,
) -> np.ndarray:
    order_by_sequence = np.argsort(sequence)
    sorted_sequence = sequence[order_by_sequence]
    outlet_position = int(np.searchsorted(sorted_sequence, outlet_sequence))
    if outlet_position >= len(sorted_sequence) or sorted_sequence[outlet_position] != outlet_sequence:
        raise ValueError(f"Outlet sequence {outlet_sequence} was not found in the routing file.")

    order_by_next = np.argsort(next_sequence)
    sorted_next = next_sequence[order_by_next]
    visited = np.zeros(len(sequence), dtype=bool)
    outlet_index = int(order_by_sequence[outlet_position])
    visited[outlet_index] = True
    pending = deque([outlet_sequence])

    while pending:
        downstream = pending.popleft()
        left = int(np.searchsorted(sorted_next, downstream, side="left"))
        right = int(np.searchsorted(sorted_next, downstream, side="right"))
        if left == right:
            continue
        upstream_indices = order_by_next[left:right]
        upstream_indices = upstream_indices[~visited[upstream_indices]]
        if upstream_indices.size == 0:
            continue
        visited[upstream_indices] = True
        pending.extend(sequence[upstream_indices].tolist())
    return np.flatnonzero(visited)


def _basin_grid(longitude: np.ndarray, latitude: np.ndarray) -> dict[str, np.ndarray | float]:
    rounded_lon = np.round(longitude.astype(float), 6)
    rounded_lat = np.round(latitude.astype(float), 6)
    unique_lon = np.unique(rounded_lon)
    unique_lat = np.unique(rounded_lat)
    lon_steps = np.diff(unique_lon)
    lat_steps = np.diff(unique_lat)
    lon_steps = lon_steps[lon_steps > 1.0e-6]
    lat_steps = lat_steps[lat_steps > 1.0e-6]
    if lon_steps.size == 0 or lat_steps.size == 0:
        raise ValueError("The upstream routing cells do not span a two-dimensional basin.")
    delta_lon = float(np.min(lon_steps))
    delta_lat = float(np.min(lat_steps))
    lon0 = float(unique_lon.min())
    lat0 = float(unique_lat.min())
    nx = int(np.rint((unique_lon.max() - lon0) / delta_lon)) + 1
    ny = int(np.rint((unique_lat.max() - lat0) / delta_lat)) + 1
    if nx * ny > 25_000_000:
        raise ValueError(
            f"Basin raster would contain {nx * ny:,} cells; check seq_lon/seq_lat spacing."
        )
    x_index = np.rint((rounded_lon - lon0) / delta_lon).astype(int)
    y_index = np.rint((rounded_lat - lat0) / delta_lat).astype(int)
    mask = np.zeros((ny, nx), dtype=bool)
    valid = (x_index >= 0) & (x_index < nx) & (y_index >= 0) & (y_index < ny)
    mask[y_index[valid], x_index[valid]] = True
    return {
        "mask": mask,
        "longitude": lon0 + np.arange(nx) * delta_lon,
        "latitude": lat0 + np.arange(ny) * delta_lat,
        "delta_lon": delta_lon,
        "delta_lat": delta_lat,
    }


def prepare_basin_layer(
    routing_file: Path,
    cache_dir: Path,
    region_name: str | None,
) -> dict[str, object] | None:
    region_key = re.sub(r"[_-]+", " ", region_name or "").strip().lower()
    outlet = OUTLET_CONFIGS.get(region_key)
    if outlet is None:
        print(f"No built-in basin outlet for region={region_name or 'unspecified'}; basin layer skipped.")
        return None

    cache_name = re.sub(r"[^a-z0-9]+", "_", region_key).strip("_")
    cache_path = cache_dir / f"{cache_name}_upstream_cells.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            basin_lon = cached["longitude"]
            basin_lat = cached["latitude"]
            route_lon = float(cached["route_lon"])
            route_lat = float(cached["route_lat"])
            outlet_sequence = int(cached["outlet_sequence"])
            distance_km = float(cached["distance_km"])
        cache_status = "loaded"
    else:
        if not routing_file.exists():
            raise FileNotFoundError(f"Routing file not found: {routing_file}")
        with h5py.File(routing_file, "r") as h5:
            sequence_raw = routing_numeric_1d(h5, "seq")
            next_raw = routing_numeric_1d(h5, "seq_next")
            longitude_raw = routing_numeric_1d(h5, "seq_lon")
            latitude_raw = routing_numeric_1d(h5, "seq_lat")
        lengths = {len(sequence_raw), len(next_raw), len(longitude_raw), len(latitude_raw)}
        if len(lengths) != 1:
            raise ValueError("Routing variables seq, seq_next, seq_lon, and seq_lat have different lengths.")
        valid = (
            np.isfinite(sequence_raw)
            & np.isfinite(longitude_raw)
            & np.isfinite(latitude_raw)
            & (sequence_raw > 0)
            & (longitude_raw >= -180.0)
            & (longitude_raw <= 180.0)
            & (latitude_raw >= -90.0)
            & (latitude_raw <= 90.0)
        )
        sequence = sequence_raw[valid].astype(np.int64)
        longitude = longitude_raw[valid]
        latitude = latitude_raw[valid]
        next_sequence = np.full(len(sequence), -1, dtype=np.int64)
        finite_next = np.isfinite(next_raw[valid]) & (next_raw[valid] > 0)
        next_sequence[finite_next] = next_raw[valid][finite_next].astype(np.int64)

        distances = _haversine_km(
            longitude,
            latitude,
            float(outlet["lon"]),
            float(outlet["lat"]),
        )
        outlet_index = int(np.argmin(distances))
        outlet_sequence = int(sequence[outlet_index])
        route_lon = float(longitude[outlet_index])
        route_lat = float(latitude[outlet_index])
        distance_km = float(distances[outlet_index])
        upstream_indices = _upstream_cell_indices(sequence, next_sequence, outlet_sequence)
        basin_lon = longitude[upstream_indices]
        basin_lat = latitude[upstream_indices]
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as cache_file:
            np.savez_compressed(
                cache_file,
                longitude=basin_lon,
                latitude=basin_lat,
                route_lon=route_lon,
                route_lat=route_lat,
                outlet_sequence=outlet_sequence,
                distance_km=distance_km,
            )
        cache_status = "created"

    grid = _basin_grid(basin_lon, basin_lat)
    grid.update(
        {
            "region": region_name,
            "station": outlet["station"],
            "grdc_id": outlet["grdc_id"],
            "route_lon": route_lon,
            "route_lat": route_lat,
            "outlet_sequence": outlet_sequence,
            "distance_km": distance_km,
        }
    )
    print(
        f"Basin cache {cache_status}: {cache_path} | outlet={outlet['station']} "
        f"(GRDC {outlet['grdc_id']}), route_seq={outlet_sequence}, "
        f"route=({route_lon:.5f}, {route_lat:.5f}), distance={distance_km:.2f} km, "
        f"upstream_cells={len(basin_lon):,}"
    )
    return grid


def draw_basin_layer(ax, basin_layer: dict[str, object], projection) -> None:
    longitude = np.asarray(basin_layer["longitude"], dtype=float)
    latitude = np.asarray(basin_layer["latitude"], dtype=float)
    delta_lon = float(basin_layer["delta_lon"])
    delta_lat = float(basin_layer["delta_lat"])
    mask = np.asarray(basin_layer["mask"], dtype=bool)
    longitude_edges = np.concatenate(
        [longitude - delta_lon / 2.0, [longitude[-1] + delta_lon / 2.0]]
    )
    latitude_edges = np.concatenate(
        [latitude - delta_lat / 2.0, [latitude[-1] + delta_lat / 2.0]]
    )
    basin_values = np.ma.masked_where(~mask, np.ones(mask.shape, dtype=float))
    ax.pcolormesh(
        longitude_edges,
        latitude_edges,
        basin_values,
        cmap=ListedColormap(["#f2b56b"]),
        vmin=0.0,
        vmax=1.0,
        shading="flat",
        alpha=0.24,
        transform=projection,
        zorder=0.6,
    )
    padded_mask = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    padded_lon = longitude[0] + (np.arange(mask.shape[1] + 2) - 1) * delta_lon
    padded_lat = latitude[0] + (np.arange(mask.shape[0] + 2) - 1) * delta_lat
    ax.contour(
        padded_lon,
        padded_lat,
        padded_mask,
        levels=[0.5],
        colors=["#9b5f24"],
        linewidths=1.1,
        alpha=0.9,
        transform=projection,
        zorder=1.5,
    )


def plot_combined_map(
    evaluation: pd.DataFrame,
    dams: pd.DataFrame,
    value_column: str,
    style: OpenBenchPlotStyle,
    output_path: Path,
    extent: tuple[float, float, float, float],
    dpi: int,
    river_detail: str,
    dam_year_filter_label: str | None = None,
    basin_layer: dict[str, object] | None = None,
    basin_mask: BasinMask | None = None,
) -> None:
    option = style.option
    matplotlib.rc("font", family=option["font"])
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(float(option["x_wise"]), float(option["y_wise"])))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1ec", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#ffffff", zorder=0)
    if basin_layer is not None:
        draw_basin_layer(ax, basin_layer, projection)
    if basin_mask is not None:
        draw_basin_mask_boundary(ax, basin_mask, projection)
    ax.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="#e6f2f8",
        edgecolor="#77a9c3",
        linewidth=0.35,
        zorder=1,
    )
    river_config = RIVER_DETAIL_CONFIG[river_detail]
    if river_detail != "none":
        ax.add_feature(
            cfeature.RIVERS.with_scale(RIVER_FEATURE_SCALES[river_detail]),
            edgecolor="#579bc4",
            linewidth=float(river_config["linewidth"]),
            alpha=0.85,
            zorder=2,
        )
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="0.48", linewidth=0.45, zorder=3)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="0.25", linewidth=0.7, zorder=3)

    gridlines = ax.gridlines(
        crs=projection,
        draw_labels=True,
        linewidth=0.45,
        color="0.55",
        alpha=0.55,
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 10}
    gridlines.ylabel_style = {"size": 10}
    gridlines.xlocator = MaxNLocator(7)
    gridlines.ylocator = MaxNLocator(6)

    station_points = ax.scatter(
        evaluation["lon"],
        evaluation["lat"],
        c=evaluation[value_column],
        s=float(option.get("markersize", 15)),
        cmap=style.cmap,
        norm=style.norm,
        marker=option["marker"],
        edgecolors="black",
        linewidths=0.45,
        alpha=0.92,
        transform=projection,
        zorder=6,
    )

    year_points = None
    if not dams.empty:
        dam_sizes = dam_marker_sizes(dams["TotalVol_mcm"])
        storage_values = pd.to_numeric(
            dams["TotalVol_mcm"], errors="coerce"
        ).fillna(0).clip(lower=0).to_numpy(float)
        valid_year = dams["year"].between(1800, 2030)
        if valid_year.any():
            year_min = float(dams.loc[valid_year, "year"].min())
            year_max = float(dams.loc[valid_year, "year"].max())
            year_norm = Normalize(vmin=year_min, vmax=year_max)
            for lower, upper, marker, _ in DAM_STORAGE_CLASSES:
                selected = (
                    valid_year.to_numpy()
                    & (storage_values >= lower)
                    & (storage_values < upper)
                )
                if not selected.any():
                    continue
                points = ax.scatter(
                    dams.loc[selected, "lon"],
                    dams.loc[selected, "lat"],
                    c=dams.loc[selected, "year"],
                    s=dam_sizes[selected],
                    cmap="viridis",
                    norm=year_norm,
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    alpha=0.96,
                    transform=projection,
                    zorder=4,
                )
                if year_points is None:
                    year_points = points
        invalid_year = (~valid_year).to_numpy()
        for lower, upper, marker, _ in DAM_STORAGE_CLASSES:
            selected = (
                invalid_year
                & (storage_values >= lower)
                & (storage_values < upper)
            )
            if selected.any():
                ax.scatter(
                    dams.loc[selected, "lon"],
                    dams.loc[selected, "lat"],
                    s=dam_sizes[selected],
                    color="0.65",
                    marker=marker,
                    edgecolors="black",
                    linewidths=0.7,
                    transform=projection,
                    zorder=4,
                )

    if year_points is not None:
        year_bar = fig.colorbar(
            year_points,
            ax=ax,
            orientation="horizontal",
            pad=0.08,
            shrink=0.67,
            aspect=30,
        )
        year_bar.set_label("Dam construction year", fontsize=11)
        year_bar.ax.tick_params(labelsize=9)

    _add_openbench_colorbar(fig, ax, station_points, style)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label=f"OpenBench station (n={len(evaluation)})",
        ),
    ]
    if river_detail != "none":
        legend_handles.append(
            Line2D(
                [],
                [],
                color="#579bc4",
                linewidth=max(1.0, float(river_config["linewidth"])),
                label=f"River network ({river_detail})",
            )
        )
    if basin_layer is not None:
        legend_handles.insert(
            0,
            Patch(
                facecolor="#f2b56b",
                edgecolor="#9b5f24",
                alpha=0.55,
                label=f"Upstream basin ({basin_layer['station']} outlet)",
            ),
        )
    elif basin_mask is not None:
        legend_handles.insert(
            0,
            Line2D(
                [],
                [],
                color="black",
                linewidth=1.2,
                label="OpenBench basin-mask boundary",
            ),
        )
    if not dams.empty:
        for _, _, marker, label in DAM_STORAGE_CLASSES:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker=marker,
                    linestyle="none",
                    markerfacecolor="#5ec962",
                    markeredgecolor="black",
                    markersize=8,
                    label=label,
                )
            )
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="0.7",
        fontsize=8.8,
        title="Map layers and dam-size scale",
        title_fontsize=9.2,
        borderpad=0.7,
    )

    ax.set_xlabel(str(option.get("xticklabel", "")), fontsize=float(option["xtick"]) + 1)
    ax.set_ylabel(str(option.get("yticklabel", "")), fontsize=float(option["ytick"]) + 1)
    ax.set_title(
        style.title,
        fontsize=float(option["title_size"]),
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.045,
        f"Additional layers — Triangles: dams "
        f"(n={len(dams)}, size = log-scaled total storage)",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_one(
    evaluation_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    value_column: str,
    shared_extent: tuple[float, float, float, float] | None = None,
    basin_layer: dict[str, object] | None = None,
    basin_mask: BasinMask | None = None,
) -> None:
    evaluation, dams, value_column = read_inputs(evaluation_path, args.dams, value_column)
    if basin_mask is not None:
        evaluation = filter_points_to_basin(evaluation, basin_mask)
        if evaluation.empty:
            raise ValueError(
                f"No evaluation stations remain inside basin mask {basin_mask.path}."
            )
    extent = shared_extent or automatic_extent(evaluation["lon"], evaluation["lat"])
    dams, dam_year_filter_label = filter_dams_by_year(
        dams,
        args.dam_built_before,
        args.dam_built_by,
    )
    dams = filter_dams_to_extent(dams, extent)
    if basin_mask is not None:
        dams = filter_points_to_basin(dams, basin_mask)
    metadata = infer_result_metadata(evaluation_path, value_column, {}, "station")
    style = resolve_openbench_plot_style(evaluation[value_column].to_numpy(float), metadata, args)
    style = _add_overlay_title(
        style,
        metadata,
        infer_region_name(evaluation_path, args.region_name),
        dam_year_filter_label,
    )
    dpi = args.dpi if args.dpi is not None else int(style.option["dpi"])
    plot_combined_map(
        evaluation,
        dams,
        value_column,
        style,
        output_path,
        extent,
        dpi,
        args.river_detail,
        dam_year_filter_label,
        basin_layer,
        basin_mask,
    )
    print(
        f"Wrote {output_path} "
        f"(metric={value_column}, product={metadata.product_type}, stations={len(evaluation)}, dams={len(dams)}, "
        f"river_detail={args.river_detail}, region={infer_region_name(evaluation_path, args.region_name) or 'unspecified'}, "
        f"extent={tuple(round(value, 6) for value in extent)})"
    )


def station_metric_columns(
    path: Path,
    requested_column: str | None,
) -> tuple[list[str], list[str]]:
    """Return plottable station metrics and metrics skipped for having no values."""
    evaluation = pd.read_csv(path, dtype={"ID": "string"})
    columns = evaluation.columns
    if {"lon", "lat"}.issubset(columns):
        lon_column, lat_column = "lon", "lat"
    elif {"ref_lon", "ref_lat"}.issubset(columns):
        lon_column, lat_column = "ref_lon", "ref_lat"
    elif "ID" in columns:
        lon_column = lat_column = None
    else:
        raise ValueError("non-spatial OpenBench result: no coordinates or station ID")

    if requested_column is not None:
        if requested_column not in columns:
            raise ValueError(f"requested column {requested_column!r} is absent")
        candidates = [requested_column]
    else:
        supported_suffixes = ("_diff", "_anomaly", "_ensemble_mean")
        legacy_candidates = [
            column for column in columns if column.endswith(supported_suffixes)
        ]
        section = _result_section(path)
        manifest = _run_manifest(path)
        if section == "metrics":
            enabled = set(_manifest_names(manifest, "metrics") or IMPLEMENTED_METRICS)
            candidates = [column for column in columns if column in enabled]
        elif section == "scores":
            enabled = set(_manifest_names(manifest, "scores") or IMPLEMENTED_SCORES)
            candidates = [
                column for column in columns if column in enabled and column != "nSpatialScore"
            ]
        elif legacy_candidates:
            candidates = legacy_candidates
        else:
            raise ValueError(
                "non-spatial OpenBench result: no enabled metric, score, anomaly, "
                "difference, or ensemble-mean column"
            )

    if lon_column is None:
        valid_coordinates = evaluation["ID"].notna()
    else:
        coordinates = evaluation[[lon_column, lat_column]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid_coordinates = coordinates.notna().all(axis=1)
    valid_metrics: list[str] = []
    empty_metrics: list[str] = []
    for column in candidates:
        values = pd.to_numeric(evaluation[column], errors="coerce")
        if (valid_coordinates & values.notna()).any():
            valid_metrics.append(column)
        else:
            empty_metrics.append(column)
    if not valid_metrics and requested_column is not None:
        raise ValueError(f"metric column {requested_column!r} contains no finite station values")
    if not valid_metrics and not empty_metrics:
        raise ValueError("no station metric columns were found")
    return valid_metrics, empty_metrics


def process_station_files(
    paths: list[Path],
    output_paths: list[Path],
    args: argparse.Namespace,
    manual_extent: tuple[float, float, float, float] | None,
    skip_errors: bool,
) -> tuple[int, int]:
    tasks: list[tuple[Path, Path, str]] = []
    skipped = 0
    for path, output_path in zip(paths, output_paths):
        try:
            metrics, empty_metrics = station_metric_columns(path, args.value_column)
        except (OSError, ValueError, KeyError) as exc:
            if not skip_errors:
                raise ValueError(f"Invalid station input {path}: {exc}") from exc
            skipped += 1
            print(f"Skipping station candidate {path}: {exc}")
            continue
        for metric in empty_metrics:
            print(f"Skipping empty station metric {metric!r} in {path}")
        for metric in metrics:
            metadata = infer_result_metadata(path, metric, {}, "station")
            if skip_errors or len(metrics) > 1:
                result_tag = "__".join(
                    (metadata.measure_type, metadata.product_type, metadata.measure)
                )
                safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_tag).strip("_")
                metric_output = (
                    output_path.parent
                    / f"{path.stem}__{safe_metric}__dams_rivers.png"
                )
            else:
                metric_output = output_path
            tasks.append((path, metric_output, metric))
    if not tasks:
        return 0, skipped

    region_name = infer_region_name(tasks[0][0], args.region_name)
    basin_layer = None
    basin_mask = None
    if not args.no_basin:
        basin_path = args.basin_mask
        if basin_path is None and region_name is not None:
            basin_path = BASIN_MASKS_BY_REGION.get(region_name)
        if basin_path is not None:
            if not basin_path.is_file():
                raise FileNotFoundError(
                    f"Basin mask file not found: {basin_path}. "
                    "Use --basin-mask PATH to select an existing mask or --no-basin to disable it."
                )
            basin_mask = load_basin_mask(basin_path)
            print(
                f"Loaded OpenBench basin mask for station filtering: {basin_mask.path} "
                f"(active_cells={int(basin_mask.data.sum())}, extent={basin_mask.extent})"
            )
        else:
            basin_layer = prepare_basin_layer(
                args.routing_file, args.basin_cache_dir, region_name
            )
    shared_extent = (
        basin_mask.extent
        if basin_mask is not None
        else manual_extent
        or automatic_extent_from_files(
            list(dict.fromkeys((path, metric) for path, _, metric in tasks))
        )
    )

    completed = 0
    for path, output_path, metric in tasks:
        try:
            render_one(
                path,
                output_path,
                args,
                metric,
                shared_extent,
                basin_layer,
                basin_mask,
            )
            completed += 1
        except (OSError, ValueError, KeyError) as exc:
            if not skip_errors:
                raise
            skipped += 1
            print(f"Skipping station candidate {path}: {exc}")
    return completed, skipped


def process_grid_files(
    paths: list[Path],
    output_paths: list[Path],
    args: argparse.Namespace,
    manual_extent: tuple[float, float, float, float] | None,
    skip_errors: bool,
) -> tuple[int, int]:
    valid_pairs: list[tuple[Path, Path]] = []
    skipped = 0
    for path, output_path in zip(paths, output_paths):
        try:
            grid = read_grid(path, args.value_variable)
            metadata = infer_result_metadata(path, grid.variable, grid.attrs, "grid")
            if metadata.measure == "nSpatialScore":
                raise ValueError("non-spatial OpenBench result: nSpatialScore is spatially constant")
            if skip_errors:
                result_tag = "__".join(
                    (metadata.measure_type, metadata.product_type, metadata.measure)
                )
                safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", result_tag).strip("_")
                output_path = output_path.parent / f"{path.stem}__{safe_tag}__dams_rivers.png"
            valid_pairs.append((path, output_path))
        except (OSError, ValueError, KeyError) as exc:
            if not skip_errors:
                raise
            skipped += 1
            print(f"Skipping grid candidate {path}: {exc}")
    if not valid_pairs:
        return 0, skipped

    region_name = args.region_name or infer_region_name(valid_pairs[0][0], None)
    basin_path = args.basin_mask
    if basin_path is None and region_name is not None:
        basin_path = BASIN_MASKS_BY_REGION.get(region_name)
    basin_mask = None
    if not args.no_basin and basin_path is not None:
        if not basin_path.is_file():
            raise FileNotFoundError(
                f"Basin mask file not found: {basin_path}. "
                "Use --basin-mask PATH to select an existing mask or --no-basin to disable it."
            )
        basin_mask = load_basin_mask(basin_path)
        print(
            f"Loaded OpenBench basin mask: {basin_mask.path} "
            f"(active_cells={int(basin_mask.data.sum())}, extent={basin_mask.extent})"
        )
    elif not args.no_basin:
        print(f"No basin mask is configured for region={region_name!r}; basin masking is disabled.")

    grid_paths = [path for path, _ in valid_pairs]
    extent = (
        basin_mask.extent
        if basin_mask is not None
        else manual_extent or automatic_extent_from_grids(grid_paths, args.value_variable)
    )
    river_network = load_river_network(
        args.routing_file, args.river_detail, extent, basin_mask
    )
    dams = read_dams(args.dams)
    completed = 0
    for path, output_path in valid_pairs:
        try:
            render_grid_file(
                path, output_path, args, dams, extent, basin_mask, river_network
            )
            completed += 1
        except (OSError, ValueError, KeyError) as exc:
            if not skip_errors:
                raise
            skipped += 1
            print(f"Skipping grid candidate {path}: {exc}")
    return completed, skipped


def main() -> None:
    args = parse_args()
    if OPENBENCH_SOURCE_ROOT is not None:
        print(f"OpenBench source selected from result path: {OPENBENCH_SOURCE_ROOT}")
    region_name = args.region_name or region_name_from_evaluation(args.evaluation_name)
    print(f"OpenBench evaluation: {args.evaluation_name} (region={region_name})")
    manual_extent = validate_extent(tuple(args.extent)) if args.extent is not None else None

    if args.evaluation_dir is None:
        path = args.evaluation
        suffix = path.suffix.lower()
        if suffix == ".csv":
            process_station_files([path], [args.output], args, manual_extent, False)
        elif suffix in {".nc", ".nc4", ".netcdf"}:
            process_grid_files([path], [args.output], args, manual_extent, False)
        else:
            raise ValueError(
                f"Cannot identify input type from extension {path.suffix!r}; "
                "expected a station CSV or grid NetCDF."
            )
        return

    station_paths = sorted(args.evaluation_dir.glob(args.station_pattern))
    grid_paths = sorted(args.evaluation_dir.glob(args.grid_pattern))
    if not station_paths and not grid_paths:
        raise FileNotFoundError(
            f"No files matching {args.station_pattern!r} or {args.grid_pattern!r} "
            f"were found in {args.evaluation_dir}."
        )

    station_dir = args.output_dir / "station"
    grid_dir = args.output_dir / "grid"
    station_outputs = [
        station_dir / f"{path.stem}__dams_rivers.png" for path in station_paths
    ]
    grid_outputs = [grid_dir / f"{path.stem}__dams_rivers.png" for path in grid_paths]
    station_completed, station_skipped = process_station_files(
        station_paths, station_outputs, args, manual_extent, True
    )
    grid_completed, grid_skipped = process_grid_files(
        grid_paths, grid_outputs, args, manual_extent, True
    )
    print(
        "Completed mixed OpenBench plotting: "
        f"station={station_completed}, grid={grid_completed}, "
        f"skipped={station_skipped + grid_skipped}, output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
