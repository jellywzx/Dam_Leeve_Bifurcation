#!/usr/bin/env python3
"""Plot OpenBench gridded metric differences with rivers and dams.

Edit ``CONFIG`` below or override its values with command-line arguments.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
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

EVALUATION_NAME = "Amazon_inundation_D"
BASIN_MASK_ROOT = Path(
    "/share/home/dq134/wzx/openbench/basin_mask/outputs/basin_masks_06min"
)
BASIN_MASKS_BY_REGION = {
    "Amazon": BASIN_MASK_ROOT / "amazon/amazon_basin.nc",
    "Lower Mississippi": BASIN_MASK_ROOT / "lower_mississippi/lower_mississippi_basin.nc",
    "Missouri": BASIN_MASK_ROOT / "missouri/missouri_basin.nc",
    "Mekong": BASIN_MASK_ROOT / "mekong/mekong_basin.nc",
}
@dataclass(frozen=True)
class ScriptConfig:
    """User-editable settings for this script."""

    evaluation_name: str
    input_suffix: Path
    grid_file: Path | None
    grid_dir: Path | None
    pattern: str
    value_variable: str | None
    output: Path
    output_dir: Path
    dpi: int
    region_name: str | None
    extent: tuple[float, float, float, float] | None
    basin_mask: Path | None
    no_basin: bool
    routing_file: Path
    river_detail: str
    color_limit: float | None
    robust_quantile: float
    dams: Path
    dam_built_before: int | None
    dam_built_by: int | None
    all_dam_years: bool


def evaluation_paths(evaluation_name: str, input_suffix: Path) -> dict[str, object]:
    """Build all evaluation-dependent defaults from one evaluation name."""
    evaluation_name = evaluation_name.strip()
    if not evaluation_name:
        raise ValueError("The evaluation name must not be empty.")
    region_name = region_name_from_evaluation(evaluation_name)
    basin_mask = BASIN_MASKS_BY_REGION.get(region_name)
    return {
        "grid_file": None,
        "grid_dir": openbench_input_dir(evaluation_name, input_suffix),
        "output": OUTPUT_ROOT / f"{evaluation_name}_grid_diff_dams_rivers.png",
        "output_dir": OUTPUT_ROOT / f"{evaluation_name}_grid",
        "region_name": region_name,
        "basin_mask": basin_mask,
        "no_basin": basin_mask is None,
    }


EVALUATION_DEFAULTS = evaluation_paths(EVALUATION_NAME, DEFAULT_INPUT_SUFFIX)


# ============================ USER CONFIGURATION ============================
# Change values here; no command-line arguments are needed.
CONFIG = ScriptConfig(
    evaluation_name=EVALUATION_NAME,
    input_suffix=DEFAULT_INPUT_SUFFIX,
    # Input: use either one grid_file or a grid_dir, and set the other to None.
    grid_file=EVALUATION_DEFAULTS["grid_file"],
    grid_dir=EVALUATION_DEFAULTS["grid_dir"],
    pattern="*__diff.nc",
    value_variable=None,  # None: automatically select the only 2-D lat/lon variable.
    # Output: output is used for grid_file; output_dir is used for grid_dir.
    output=EVALUATION_DEFAULTS["output"],
    output_dir=EVALUATION_DEFAULTS["output_dir"],
    dpi=300,
    # Map and OpenBench basin mask.
    region_name=EVALUATION_DEFAULTS["region_name"],
    extent=None,  # Used only when no_basin=True; None selects an automatic extent.
    basin_mask=EVALUATION_DEFAULTS["basin_mask"],
    no_basin=EVALUATION_DEFAULTS["no_basin"],
    # CoLM river network: "all", "fine", "medium", "coarse", or "none".
    routing_file=DEFAULT_ROUTING_FILE,
    river_detail=DEFAULT_RIVER_DETAIL,
    # Symmetric grid color scale; None uses robust_quantile automatically.
    color_limit=None,
    robust_quantile=0.98,
    # Dams. Set only one year option; all_dam_years=True requires both years=None.
    dams=DEFAULT_DAMS,
    dam_built_before=DEFAULT_DAM_BUILT_BEFORE,
    dam_built_by=None,
    all_dam_years=False,
)
# ============================================================================


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


def parse_args() -> ScriptConfig:
    """Return ``CONFIG`` with explicitly supplied CLI options applied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-name",
        help="OpenBench evaluation name used to derive input, output, region, and basin defaults.",
    )
    parser.add_argument(
        "--input-suffix",
        type=Path,
        help=f"Input path below the evaluation directory (default: {DEFAULT_INPUT_SUFFIX}).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--grid-file", type=Path, help="Plot one OpenBench NetCDF grid.")
    source.add_argument("--grid-dir", type=Path, help="Plot matching grids in this directory.")
    parser.add_argument("--pattern", help="Glob used with --grid-dir (default: %(default)s).")
    parser.add_argument("--value-variable", help="NetCDF variable to plot (default: auto-detect).")
    parser.add_argument("--output", type=Path, help="Output PNG used with --grid-file.")
    parser.add_argument("--output-dir", type=Path, help="Output directory used with --grid-dir.")
    parser.add_argument("--dpi", type=int, help="Output resolution in dots per inch.")
    parser.add_argument("--region-name", help="Map-title region name (default: derive from evaluation).")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        help="Map extent; used only without a basin mask.",
    )
    parser.add_argument("--basin-mask", type=Path, help="OpenBench basin-mask NetCDF file.")
    basin = parser.add_mutually_exclusive_group()
    basin.add_argument("--no-basin", dest="no_basin", action="store_true", help="Disable basin masking.")
    basin.add_argument("--use-basin", dest="no_basin", action="store_false", help="Enable basin masking.")
    parser.add_argument("--routing-file", type=Path, help="CoLM routing NetCDF file.")
    parser.add_argument(
        "--river-detail",
        choices=tuple(RIVER_DETAIL_CONFIG),
        help=f"River detail level (default: {DEFAULT_RIVER_DETAIL}).",
    )
    parser.add_argument("--color-limit", type=float, help="Positive symmetric color-scale limit.")
    parser.add_argument("--robust-quantile", type=float, help="Quantile used for automatic color limits.")
    parser.add_argument("--dams", type=Path, help="Dam CSV file.")
    years = parser.add_mutually_exclusive_group()
    years.add_argument("--dam-built-before", type=int, metavar="YEAR")
    years.add_argument("--dam-built-by", type=int, metavar="YEAR")
    years.add_argument("--all-dam-years", action="store_true", help="Do not filter dams by year.")

    parser.set_defaults(**CONFIG.__dict__)
    namespace = parser.parse_args()
    values = vars(namespace)

    # Selecting a CLI source or year policy replaces the corresponding CONFIG choice.
    cli_arguments = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            cli_arguments.add(token.split("=", 1)[0])

    if "--evaluation-name" in cli_arguments:
        derived = evaluation_paths(namespace.evaluation_name, namespace.input_suffix)
        evaluation_options = {
            "grid_dir": "--grid-dir",
            "output": "--output",
            "output_dir": "--output-dir",
            "region_name": "--region-name",
            "basin_mask": "--basin-mask",
            "no_basin": None,
        }
        for field, option in evaluation_options.items():
            if option is None or option not in cli_arguments:
                values[field] = derived[field]
    elif "--input-suffix" in cli_arguments and "--grid-dir" not in cli_arguments:
        values["grid_dir"] = openbench_input_dir(namespace.evaluation_name, namespace.input_suffix)

    if "--grid-file" in cli_arguments:
        values["grid_dir"] = None
    elif "--grid-dir" in cli_arguments:
        values["grid_file"] = None

    basin_options = {"--no-basin", "--use-basin"}
    if "--basin-mask" in cli_arguments and not cli_arguments & basin_options:
        values["no_basin"] = False
    elif "--no-basin" in cli_arguments:
        values["no_basin"] = True
    elif "--use-basin" in cli_arguments:
        values["no_basin"] = False
    if values["basin_mask"] is None:
        values["no_basin"] = True

    year_options = {"--dam-built-before", "--dam-built-by", "--all-dam-years"}
    if cli_arguments & year_options:
        values["dam_built_before"] = (
            namespace.dam_built_before if "--dam-built-before" in cli_arguments else None
        )
        values["dam_built_by"] = namespace.dam_built_by if "--dam-built-by" in cli_arguments else None
        values["all_dam_years"] = "--all-dam-years" in cli_arguments
    if values["extent"] is not None:
        values["extent"] = tuple(values["extent"])
    return ScriptConfig(**values)


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
            if len(candidates) != 1:
                raise ValueError(
                    f"Expected exactly one two-dimensional lat/lon data variable in {path}, "
                    f"found {candidates}. Set CONFIG.value_variable explicitly."
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
        sequence_raw = _routing_numeric_1d(h5, "seq")
        next_raw = _routing_numeric_1d(h5, "seq_next")
        longitude = _routing_numeric_1d(h5, "seq_lon")
        latitude = _routing_numeric_1d(h5, "seq_lat")
        upstream_area = _routing_numeric_1d(h5, "topo_uparea")
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


def automatic_color_limit(values: np.ndarray, quantile: float) -> float:
    if not 0 < quantile <= 1:
        raise ValueError("CONFIG.robust_quantile must be greater than 0 and at most 1.")
    absolute = np.abs(values[np.isfinite(values)])
    if absolute.size == 0:
        raise ValueError("The selected grid variable contains no finite values.")
    limit = float(np.quantile(absolute, quantile))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(absolute))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return limit


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


def labels_from_grid(path: Path, grid: GridData) -> tuple[str, str, str, str, str]:
    parts = path.stem.split("__")
    item = str(grid.attrs.get("item") or (parts[0] if parts else "OpenBench variable"))
    reference = str(grid.attrs.get("ref_source") or "reference")
    metric = str(grid.attrs.get("metric") or grid.attrs.get("score") or grid.variable)
    if metric.endswith("_diff"):
        metric = metric[: -len("_diff")]
    simulation_1 = "first simulation"
    simulation_2 = "second simulation"
    if "ref" in parts:
        ref_index = parts.index("ref")
        if ref_index + 1 < len(parts):
            reference = parts[ref_index + 1]
    if "vs" in parts:
        versus_index = parts.index("vs")
        if versus_index >= 1:
            simulation_1 = parts[versus_index - 1]
        if versus_index + 1 < len(parts):
            simulation_2 = parts[versus_index + 1]
        if versus_index + 2 < len(parts):
            metric = parts[versus_index + 2]
    return (
        item.replace("_", " "),
        reference.replace("_", " "),
        metric.replace("_", " "),
        simulation_1,
        simulation_2,
    )


def plot_combined_map(
    grid: GridData,
    dams: pd.DataFrame,
    grid_path: Path,
    output_path: Path,
    extent: tuple[float, float, float, float],
    color_limit: float,
    dpi: int,
    region_name: str | None,
    dam_year_filter_label: str | None,
    basin_mask: BasinMask | None,
    river_network: RiverNetwork | None,
) -> None:
    if color_limit <= 0 or not np.isfinite(color_limit):
        raise ValueError("The color limit must be finite and positive.")

    item, reference, metric, simulation_1, simulation_2 = labels_from_grid(grid_path, grid)
    projection = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14.0, 9.0))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_extent(extent, crs=projection)

    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#f3f1ec", zorder=0)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#ffffff", zorder=0)

    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    grid_values = np.ma.masked_invalid(grid.values)
    grid_layer = ax.pcolormesh(
        coordinate_edges(grid.longitude),
        coordinate_edges(grid.latitude),
        grid_values,
        cmap="RdBu_r",
        norm=norm,
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
        valid_year = dams["year"].between(1800, 2030)
        if valid_year.any():
            year_min = float(dams.loc[valid_year, "year"].min())
            year_max = float(dams.loc[valid_year, "year"].max())
            if year_min == year_max:
                year_min -= 0.5
                year_max += 0.5
            year_points = ax.scatter(
                dams.loc[valid_year, "lon"],
                dams.loc[valid_year, "lat"],
                c=dams.loc[valid_year, "year"],
                s=dam_sizes[valid_year.to_numpy()],
                cmap="viridis",
                norm=Normalize(vmin=year_min, vmax=year_max),
                marker="^",
                edgecolors="black",
                linewidths=0.7,
                alpha=0.96,
                transform=projection,
                zorder=5,
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
                zorder=5,
            )

    finite_values = grid.values[np.isfinite(grid.values)]
    below = float(np.min(finite_values)) < -color_limit
    above = float(np.max(finite_values)) > color_limit
    extend = "both" if below and above else "min" if below else "max" if above else "neither"
    metric_bar = fig.colorbar(
        grid_layer,
        ax=ax,
        orientation="horizontal",
        pad=0.075,
        fraction=0.052,
        shrink=0.72,
        extend=extend,
    )
    metric_bar.set_label(
        f"{metric} difference ({simulation_1} − {simulation_2})",
        fontsize=11,
    )
    metric_bar.ax.tick_params(labelsize=9)

    if year_points is not None:
        year_bar = fig.colorbar(year_points, ax=ax, orientation="vertical", pad=0.018, shrink=0.67)
        year_bar.set_label("Dam construction year", fontsize=11)
        year_bar.ax.tick_params(labelsize=9)

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

    comparison_line = f"Reference: {reference}  |  {simulation_1} versus {simulation_2}"
    if dam_year_filter_label is not None:
        comparison_line += f"  |  {dam_year_filter_label}"
    region_prefix = f"{region_name} " if region_name else ""
    ax.set_title(
        f"{region_prefix}{item} {metric} Difference, River Network, and Dams\n"
        f"{comparison_line}",
        fontsize=16,
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
        0.012,
        f"Shading: OpenBench basin-masked grid-cell difference  |  {river_summary}  |  Triangles: dams "
        f"(n={len(dams)}, size = log-scaled total storage)",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_one(
    grid_path: Path,
    output_path: Path,
    args: ScriptConfig,
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
    color_limit = args.color_limit
    if color_limit is None:
        color_limit = automatic_color_limit(grid.values, args.robust_quantile)
    plot_combined_map(
        grid,
        visible_dams,
        grid_path,
        output_path,
        extent,
        color_limit,
        args.dpi,
        infer_region_name(grid_path, args.region_name),
        dam_year_filter_label,
        basin_mask,
        river_network,
    )
    valid_count = int(np.isfinite(grid.values).sum())
    print(
        f"Wrote {output_path} "
        f"(variable={grid.variable}, valid_cells={valid_count}, limit=±{color_limit:.6g}, "
        f"dams={len(visible_dams)}, rivers={0 if river_network is None else len(river_network.segments)}, "
        f"river_detail={args.river_detail}, extent={tuple(round(value, 6) for value in extent)})"
    )


def main() -> None:
    args = parse_args()
    print(f"OpenBench evaluation: {args.evaluation_name} (region={args.region_name})")
    if args.grid_file is None and args.grid_dir is None:
        raise ValueError("Set either CONFIG.grid_file or CONFIG.grid_dir.")
    if args.grid_file is not None and args.grid_dir is not None:
        raise ValueError(
            "CONFIG.grid_file and CONFIG.grid_dir are mutually exclusive; set one of them to None."
        )
    if args.river_detail not in RIVER_DETAIL_CONFIG:
        raise ValueError(
            f"CONFIG.river_detail must be one of {tuple(RIVER_DETAIL_CONFIG)}; "
            f"found {args.river_detail!r}."
        )
    year_settings = sum(
        (
            args.dam_built_before is not None,
            args.dam_built_by is not None,
            args.all_dam_years,
        )
    )
    if year_settings > 1:
        raise ValueError(
            "CONFIG.dam_built_before, CONFIG.dam_built_by, and "
            "CONFIG.all_dam_years are mutually exclusive."
        )
    if args.dpi <= 0:
        raise ValueError("CONFIG.dpi must be positive.")
    if args.color_limit is not None and (
        not np.isfinite(args.color_limit) or args.color_limit <= 0
    ):
        raise ValueError("CONFIG.color_limit must be finite and positive.")
    if not 0 < args.robust_quantile <= 1:
        raise ValueError("CONFIG.robust_quantile must be greater than 0 and at most 1.")

    manual_extent = validate_extent(tuple(args.extent)) if args.extent is not None else None
    if args.grid_file is not None:
        grid_paths = [args.grid_file]
        output_paths = [args.output]
    else:
        grid_paths = sorted(args.grid_dir.glob(args.pattern))
        if not grid_paths:
            raise FileNotFoundError(
                f"No files matching {args.pattern!r} were found in {args.grid_dir}."
            )
        output_paths = [
            args.output_dir / f"{grid_path.stem}__dams_rivers.png" for grid_path in grid_paths
        ]

    if args.no_basin and args.basin_mask is None:
        print(
            f"No basin mask is configured for region={args.region_name!r}; "
            "basin masking is disabled."
        )
    basin_mask = None if args.no_basin else load_basin_mask(args.basin_mask)
    extent = (
        basin_mask.extent
        if basin_mask is not None
        else manual_extent or automatic_extent_from_grids(grid_paths, args.value_variable)
    )
    if basin_mask is not None:
        print(
            f"Loaded OpenBench basin mask: {basin_mask.path} "
            f"(active_cells={int(basin_mask.data.sum())}, extent={basin_mask.extent})"
        )
    river_network = load_river_network(
        args.routing_file,
        args.river_detail,
        extent,
        basin_mask,
    )
    dams = read_dams(args.dams)

    for grid_path, output_path in zip(grid_paths, output_paths):
        render_one(grid_path, output_path, args, dams, extent, basin_mask, river_network)
    if len(grid_paths) > 1:
        print(f"Completed {len(grid_paths)} grid metric maps in {args.output_dir}")


if __name__ == "__main__":
    main()
