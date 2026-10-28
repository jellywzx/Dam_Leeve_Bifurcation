#!/usr/bin/env python3
"""Plot one or many OpenBench station metrics with rivers and dams."""

from __future__ import annotations

import argparse
import re
from collections import deque
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from openbench_dams_rivers_common import (
    DEFAULT_DAMS,
    DEFAULT_DAM_BUILT_BEFORE,
    DEFAULT_INPUT_SUFFIX,
    DEFAULT_RIVER_DETAIL,
    DEFAULT_ROUTING_FILE,
    EXTENT_PADDING_FRACTION,
    MIN_EXTENT_PADDING_DEGREES,
    OUTPUT_ROOT,
    RIVER_DETAIL_CONFIG,
    dam_marker_sizes,
    filter_dams_by_year,
    filter_dams_to_extent,
    infer_region_name,
    openbench_input_dir,
    region_name_from_evaluation,
    storage_marker_size,
    validate_extent,
)

EVALUATION_NAME = "LowerMississippi_inundation_D"
RIVER_FEATURE_SCALES = {
    "all": "10m",
    "fine": "10m",
    "medium": "50m",
    "coarse": "110m",
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


def evaluation_output(evaluation_name: str) -> Path:
    return OUTPUT_ROOT / f"{evaluation_name}_dams_rivers_metric_diff.png"


def evaluation_output_dir(evaluation_name: str) -> Path:
    return OUTPUT_ROOT / evaluation_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine OpenBench metric differences, river network, and dam information."
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
        type=Path,
        help="One OpenBench station difference CSV instead of the default batch directory.",
    )
    input_group.add_argument(
        "--evaluation-dir",
        type=Path,
        help="Batch input directory (default: derived from --evaluation-name).",
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
        default="*__diff.csv",
        help="Input filename pattern used in batch mode (default: *__diff.csv).",
    )
    parser.add_argument(
        "--value-column",
        help="Metric column to plot; by default the single column ending in '_diff' is used.",
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
        help="Manual symmetric color limit; default is the robust per-metric limit.",
    )
    parser.add_argument(
        "--robust-quantile",
        type=float,
        default=0.98,
        help="Absolute-value quantile used for automatic color limits (default: 0.98).",
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
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.evaluation is None and args.evaluation_dir is None:
        args.evaluation_dir = openbench_input_dir(args.evaluation_name, args.input_suffix)
    if args.output is None:
        args.output = evaluation_output(args.evaluation_name)
    if args.output_dir is None:
        args.output_dir = evaluation_output_dir(args.evaluation_name)
    if args.dam_built_before is None and args.dam_built_by is None and not args.all_dam_years:
        args.dam_built_before = DEFAULT_DAM_BUILT_BEFORE
    return args


def read_inputs(
    evaluation_path: Path,
    dam_path: Path,
    requested_value_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    evaluation = pd.read_csv(evaluation_path)
    dams = pd.read_csv(dam_path)

    if requested_value_column is None:
        value_columns = [column for column in evaluation.columns if column.endswith("_diff")]
        if len(value_columns) != 1:
            raise ValueError(
                f"Expected exactly one '*_diff' column in {evaluation_path}, found {value_columns}. "
                "Use --value-column to select one explicitly."
            )
        value_column = value_columns[0]
    else:
        value_column = requested_value_column

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


def automatic_color_limit(values: pd.Series, quantile: float) -> float:
    if not 0 < quantile <= 1:
        raise ValueError("--robust-quantile must be greater than 0 and at most 1.")
    absolute = np.abs(pd.to_numeric(values, errors="coerce").dropna().to_numpy(float))
    if absolute.size == 0:
        raise ValueError("The selected metric contains no finite values.")
    limit = float(np.quantile(absolute, quantile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.nanmax(absolute))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return limit


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


def automatic_extent_from_files(evaluation_paths: list[Path]) -> tuple[float, float, float, float]:
    longitude = []
    latitude = []
    for evaluation_path in evaluation_paths:
        coordinates = pd.read_csv(evaluation_path, usecols=["lon", "lat"])
        longitude.append(coordinates["lon"])
        latitude.append(coordinates["lat"])
    return automatic_extent(pd.concat(longitude, ignore_index=True), pd.concat(latitude, ignore_index=True))


def labels_from_input(evaluation_path: Path, value_column: str) -> tuple[str, str, str]:
    metric = value_column[: -len("_diff")].replace("_", " ")
    parts = evaluation_path.stem.split("__")
    if "vs" not in parts:
        return metric, "first simulation", "second simulation"
    versus_index = parts.index("vs")
    if versus_index < 2 or versus_index + 1 >= len(parts):
        return metric, "first simulation", "second simulation"
    return metric, parts[versus_index - 2], parts[versus_index + 1]


def _routing_numeric_1d(h5: h5py.File, name: str) -> np.ndarray:
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
            sequence_raw = _routing_numeric_1d(h5, "seq")
            next_raw = _routing_numeric_1d(h5, "seq_next")
            longitude_raw = _routing_numeric_1d(h5, "seq_lon")
            latitude_raw = _routing_numeric_1d(h5, "seq_lat")
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
    metric_label: str,
    region_name: str | None,
    simulation_1: str,
    simulation_2: str,
    output_path: Path,
    extent: tuple[float, float, float, float],
    color_limit: float,
    dpi: int,
    river_detail: str,
    dam_year_filter_label: str | None = None,
    basin_layer: dict[str, object] | None = None,
) -> None:
    if color_limit <= 0:
        raise ValueError("The color limit must be positive.")

    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1ec", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#ffffff", zorder=0)
    if basin_layer is not None:
        draw_basin_layer(ax, basin_layer, projection)
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

    metric_norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    station_points = ax.scatter(
        evaluation["lon"],
        evaluation["lat"],
        c=evaluation[value_column],
        s=34,
        cmap="RdBu_r",
        norm=metric_norm,
        marker="o",
        edgecolors="black",
        linewidths=0.45,
        alpha=0.92,
        transform=projection,
        zorder=6,
    )

    year_points = None
    if not dams.empty:
        dam_sizes = dam_marker_sizes(dams["TotalVol_mcm"])
        valid_year = dams["year"].between(1800, 2030)
        if valid_year.any():
            year_min = float(dams.loc[valid_year, "year"].min())
            year_max = float(dams.loc[valid_year, "year"].max())
            year_norm = Normalize(vmin=year_min, vmax=year_max)
            year_points = ax.scatter(
                dams.loc[valid_year, "lon"],
                dams.loc[valid_year, "lat"],
                c=dams.loc[valid_year, "year"],
                s=dam_sizes[valid_year.to_numpy()],
                cmap="viridis",
                norm=year_norm,
                marker="^",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.96,
                transform=projection,
                zorder=4,
            )
        if (~valid_year).any():
            ax.scatter(
                dams.loc[~valid_year, "lon"],
                dams.loc[~valid_year, "lat"],
                s=dam_sizes[(~valid_year).to_numpy()],
                color="0.65",
                marker="^",
                edgecolors="black",
                linewidths=0.7,
                transform=projection,
                zorder=4,
            )

    metric_values = evaluation[value_column].to_numpy(float)
    extend = "neither"
    if np.nanmin(metric_values) < -color_limit and np.nanmax(metric_values) > color_limit:
        extend = "both"
    elif np.nanmin(metric_values) < -color_limit:
        extend = "min"
    elif np.nanmax(metric_values) > color_limit:
        extend = "max"
    metric_bar = fig.colorbar(
        station_points,
        ax=ax,
        orientation="horizontal",
        pad=0.075,
        fraction=0.052,
        shrink=0.72,
        extend=extend,
    )
    metric_bar.set_label(
        f"{metric_label} difference ({simulation_1} − {simulation_2})",
        fontsize=11,
    )
    metric_bar.ax.tick_params(labelsize=9)

    if year_points is not None:
        year_bar = fig.colorbar(year_points, ax=ax, orientation="vertical", pad=0.018, shrink=0.67)
        year_bar.set_label("Dam construction year", fontsize=11)
        year_bar.ax.tick_params(labelsize=9)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label=f"GRDC station (n={len(evaluation)})",
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
    if not dams.empty:
        maximum_storage = float(dams["TotalVol_mcm"].fillna(0).clip(lower=0).max())
        for storage in (100, 1000, 10000):
            size = storage_marker_size(storage, maximum_storage)
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="^",
                    linestyle="none",
                    markerfacecolor="#5ec962",
                    markeredgecolor="black",
                    markersize=np.sqrt(size),
                    label=f"Dam storage {storage:,} million m³",
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

    comparison_line = f"OpenBench: {simulation_1} versus {simulation_2}"
    if dam_year_filter_label is not None:
        comparison_line += f"  |  {dam_year_filter_label}"
    region_prefix = f"{region_name} " if region_name else ""
    ax.set_title(
        f"{region_prefix}Streamflow {metric_label} Difference, River Network, and Dams\n"
        f"{comparison_line}",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.012,
        f"Circles: GRDC evaluation stations  |  Triangles: dams "
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
    shared_extent: tuple[float, float, float, float] | None = None,
    basin_layer: dict[str, object] | None = None,
) -> None:
    evaluation, dams, value_column = read_inputs(evaluation_path, args.dams, args.value_column)
    extent = shared_extent or automatic_extent(evaluation["lon"], evaluation["lat"])
    dams, dam_year_filter_label = filter_dams_by_year(
        dams,
        args.dam_built_before,
        args.dam_built_by,
    )
    dams = filter_dams_to_extent(dams, extent)
    metric_label, simulation_1, simulation_2 = labels_from_input(evaluation_path, value_column)
    region_name = infer_region_name(evaluation_path, args.region_name)
    color_limit = args.color_limit
    if color_limit is None:
        color_limit = automatic_color_limit(evaluation[value_column], args.robust_quantile)
    plot_combined_map(
        evaluation,
        dams,
        value_column,
        metric_label,
        region_name,
        simulation_1,
        simulation_2,
        output_path,
        extent,
        color_limit,
        args.dpi,
        args.river_detail,
        dam_year_filter_label,
        basin_layer,
    )
    print(
        f"Wrote {output_path} "
        f"(metric={value_column}, limit=±{color_limit:.6g}, stations={len(evaluation)}, dams={len(dams)}, "
        f"river_detail={args.river_detail}, region={region_name or 'unspecified'}, "
        f"extent={tuple(round(value, 6) for value in extent)})"
    )


def main() -> None:
    args = parse_args()
    print(
        f"OpenBench evaluation: {args.evaluation_name} "
        f"(region={args.region_name or region_name_from_evaluation(args.evaluation_name)})"
    )
    manual_extent = validate_extent(tuple(args.extent)) if args.extent is not None else None
    if args.evaluation_dir is None:
        region_name = infer_region_name(args.evaluation, args.region_name)
        basin_layer = None
        if not args.no_basin:
            basin_layer = prepare_basin_layer(args.routing_file, args.basin_cache_dir, region_name)
        render_one(args.evaluation, args.output, args, manual_extent, basin_layer)
        return

    input_files = sorted(args.evaluation_dir.glob(args.pattern))
    if not input_files:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} were found in {args.evaluation_dir}."
        )
    shared_extent = manual_extent or automatic_extent_from_files(input_files)
    region_name = infer_region_name(input_files[0], args.region_name)
    basin_layer = None
    if not args.no_basin:
        basin_layer = prepare_basin_layer(args.routing_file, args.basin_cache_dir, region_name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for evaluation_path in input_files:
        output_path = args.output_dir / f"{evaluation_path.stem}__dams_rivers.png"
        render_one(evaluation_path, output_path, args, shared_extent, basin_layer)
    print(f"Completed {len(input_files)} metric maps in {args.output_dir}")


if __name__ == "__main__":
    main()
