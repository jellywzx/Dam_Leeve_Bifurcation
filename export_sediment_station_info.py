#!/usr/bin/env python3
"""Export the stations actually used by OpenBench sediment evaluations.

使用说明
--------
本脚本需要在计算节点 ``node111`` 上运行，并需要将 OpenBench 源码目录加入
``PYTHONPATH``。例如：

.. code-block:: bash

   ssh node111
   cd /share/home/dq134/wzx/openbench/postprocess
   export PYTHONPATH=/share/home/dq134/wzx/openbench/openbench_1015/src:${PYTHONPATH:-}
   python export_sediment_station_info.py \\
       --config /path/to/openbench_config.yaml

如果配置文件中的评估结果目录无法自动定位，可显式指定 ``--case-dir``；输出
目录默认是 ``<case-dir>/data/station_info``，也可通过 ``--output-dir`` 修改：

.. code-block:: bash

   python export_sediment_station_info.py \\
       --config /path/to/openbench_config.yaml \\
       --case-dir /path/to/openbench_evaluation_case \\
       --output-dir /path/to/station_info

使用 ``--variable``、``--reference`` 或 ``--simulation`` 可以筛选输出；查看
全部参数请运行 ``python export_sediment_station_info.py --help``。

The evaluation CSV is the source of truth for the selected stations.  The
reference NetCDF is opened only to read one-dimensional station metadata; no
time-series variable is loaded.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import logging
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd

from openbench.util.filenames import join_filename_components
from openbench.util.station_ids import read_station_csv, station_id_key


LOG = logging.getLogger("export_sediment_station_info")
SEDIMENT_VARIABLES = (
    "Discharge_For_Sediment",
    "Suspended_Sediment_Concentration",
    "Suspended_Sediment_Load",
)
_SOURCE_SUFFIX = {
    "Discharge_For_Sediment": "Streamflow",
    "Suspended_Sediment_Concentration": "Suspended_Sediment_Concentration",
    "Suspended_Sediment_Load": "Suspended_Sediment_Load",
}


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in frame.columns}
    return next((lookup[name.casefold()] for name in names if name.casefold() in lookup), None)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _candidate_result_names(variable: str, reference: str, simulation: str) -> list[str]:
    legacy = f"{variable}_stn_{reference}_{simulation}_evaluations.csv"
    safe = f"{join_filename_components(variable, 'stn', reference, simulation, 'evaluations')}.csv"
    return list(dict.fromkeys((legacy, safe)))


def find_evaluated_station_file(case_dir: Path, variable: str, reference: str, simulation: str) -> Path:
    """Find successful station results, then fall back to the final station list."""
    names = _candidate_result_names(variable, reference, simulation)
    for category in ("metrics", "scores"):
        for name in names:
            path = case_dir / category / name
            if path.is_file():
                return path

    exact_lists = (
        case_dir / f"stn_{reference}_{simulation}_list.csv",
        case_dir / f"stn_{reference}_{simulation}_list.txt",
    )
    for path in exact_lists:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No evaluated station file for {variable} / {reference} / {simulation} under {case_dir}. "
        "Run the OpenBench evaluation first."
    )


def _station_dimension(ds, station_id_var: str) -> tuple[str, str]:
    lookup = {str(name).casefold(): str(name) for name in ds.variables}
    station_key = lookup.get(station_id_var.casefold()) or lookup.get("station")
    if station_key is None:
        raise ValueError(f"Reference dataset has no station identifier variable {station_id_var!r}")
    variable = ds[station_key]
    if variable.ndim != 1:
        raise ValueError(f"Station identifier {station_key!r} must be one-dimensional")
    return station_key, str(variable.dims[0])


def _one_dimensional_values(ds, station_dim: str, names: Iterable[str], size: int, default=None):
    lookup = {str(name).casefold(): str(name) for name in ds.variables}
    for name in names:
        key = lookup.get(name.casefold())
        if key is not None and ds[key].dims == (station_dim,):
            return np.asarray(ds[key].values)
    return np.full(size, default, dtype=object)


def read_reference_station_metadata(
    dataset_path: Path,
    *,
    variable: str,
    grid_resolution: float | None,
    station_id_var: str = "station",
) -> pd.DataFrame:
    """Read station-only metadata without touching any time-series variable."""
    import xarray as xr

    with xr.open_dataset(dataset_path, decode_times=False, cache=False) as ds:
        station_key, station_dim = _station_dimension(ds, station_id_var)
        station_ids = np.asarray(ds[station_key].values)
        size = len(station_ids)

        resolution_suffix = None
        if grid_resolution is not None:
            try:
                from openbench.data.station_matcher import get_resolution_suffix

                resolution_suffix = get_resolution_suffix(grid_resolution)
            except (TypeError, ValueError):
                LOG.warning("No CaMA metadata suffix for grid resolution %r", grid_resolution)

        source_suffix = _SOURCE_SUFFIX.get(variable, variable)
        source_names = (f"source_dataset_{source_suffix}", "source_dataset")
        cama_lon_names = (f"cama_lon_{resolution_suffix}",) if resolution_suffix else ()
        cama_lat_names = (f"cama_lat_{resolution_suffix}",) if resolution_suffix else ()
        cama_area_names = (f"cama_uparea_{resolution_suffix}",) if resolution_suffix else ()
        alloc_error_names = (f"cama_alloc_err_{resolution_suffix}",) if resolution_suffix else ()

        frame = pd.DataFrame(
            {
                "station_id": [str(value) for value in station_ids],
                "station_name": _one_dimensional_values(ds, station_dim, ("station_name",), size, ""),
                "river": _one_dimensional_values(ds, station_dim, ("river",), size, ""),
                "country": _one_dimensional_values(ds, station_dim, ("country",), size, ""),
                "country_code": _one_dimensional_values(ds, station_dim, ("country_code",), size, ""),
                "data_source": _one_dimensional_values(ds, station_dim, source_names, size, ""),
                "source_station_id": _one_dimensional_values(
                    ds, station_dim, ("source_station_id", "source_station_uid"), size, ""
                ),
                "original_lon": _one_dimensional_values(ds, station_dim, ("lon", "longitude"), size, np.nan),
                "original_lat": _one_dimensional_values(ds, station_dim, ("lat", "latitude"), size, np.nan),
                "matched_lon": _one_dimensional_values(ds, station_dim, cama_lon_names, size, np.nan),
                "matched_lat": _one_dimensional_values(ds, station_dim, cama_lat_names, size, np.nan),
                "upstream_area_km2": _one_dimensional_values(
                    ds, station_dim, ("upstream_area", "area"), size, np.nan
                ),
                "matched_upstream_area_km2": _one_dimensional_values(
                    ds, station_dim, cama_area_names, size, np.nan
                ),
                "allocation_error": _one_dimensional_values(ds, station_dim, alloc_error_names, size, np.nan),
            }
        )
    return frame


def _numeric_column(frame: pd.DataFrame, names: Iterable[str]) -> np.ndarray:
    column = _first_column(frame, names)
    if column is None:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def merge_evaluated_with_metadata(evaluated: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach metadata with unambiguous ID matching and coordinate fallback."""
    id_column = _first_column(evaluated, ("ID", "station_id", "site_id", "site"))
    if id_column is None:
        raise ValueError("Evaluated station table has no ID/station_id column")

    evaluated = evaluated.copy()
    evaluated_ids = evaluated[id_column].fillna("").astype(str)
    metadata = metadata.copy()
    metadata["_key"] = metadata["station_id"].map(station_id_key)
    key_to_indices: dict[str, list[int]] = {}
    for index, key in enumerate(metadata["_key"]):
        key_to_indices.setdefault(key, []).append(index)

    eval_lon = _numeric_column(evaluated, ("ref_lon", "matched_lon", "lon", "longitude"))
    eval_lat = _numeric_column(evaluated, ("ref_lat", "matched_lat", "lat", "latitude"))
    meta_lon = pd.to_numeric(metadata["matched_lon"], errors="coerce").to_numpy(dtype=float)
    meta_lat = pd.to_numeric(metadata["matched_lat"], errors="coerce").to_numpy(dtype=float)
    raw_lon = pd.to_numeric(metadata["original_lon"], errors="coerce").to_numpy(dtype=float)
    raw_lat = pd.to_numeric(metadata["original_lat"], errors="coerce").to_numpy(dtype=float)

    selected: list[int | None] = []
    statuses: list[str] = []
    for row_index, station_id in enumerate(evaluated_ids):
        key = station_id_key(station_id)
        matches = key_to_indices.get(key, [])
        if len(matches) == 1:
            selected.append(matches[0])
            statuses.append("id")
            continue

        # Source-qualified duplicate IDs are written as SOURCE_<id>.
        suffix_key = station_id_key(station_id.split("_", 1)[-1]) if "_" in station_id else ""
        suffix_matches = key_to_indices.get(suffix_key, []) if suffix_key else []
        candidates = matches or suffix_matches or list(range(len(metadata)))
        lon, lat = eval_lon[row_index], eval_lat[row_index]
        if np.isfinite(lon) and np.isfinite(lat):
            candidate_array = np.asarray(candidates, dtype=int)
            distances = np.minimum(
                np.hypot(meta_lon[candidate_array] - lon, meta_lat[candidate_array] - lat),
                np.hypot(raw_lon[candidate_array] - lon, raw_lat[candidate_array] - lat),
            )
            finite = np.isfinite(distances)
            if finite.any():
                best_position = int(np.nanargmin(distances))
                best_distance = float(distances[best_position])
                tied = int(np.count_nonzero(np.isclose(distances, best_distance, atol=1e-6, rtol=0)))
                if best_distance <= 1e-4 and tied == 1:
                    selected.append(int(candidate_array[best_position]))
                    statuses.append("qualified_id_and_coordinates" if suffix_matches else "coordinates")
                    continue
        selected.append(None)
        statuses.append("ambiguous" if matches or suffix_matches else "not_found")

    public_columns = [column for column in metadata.columns if column != "_key"]
    attached_rows = []
    for index in selected:
        attached_rows.append({column: metadata.iloc[index][column] for column in public_columns} if index is not None else {})
    attached = pd.DataFrame(attached_rows, columns=public_columns)

    result = attached.copy()
    result["station_id"] = evaluated_ids.to_numpy()
    result["matched_lon"] = np.where(np.isfinite(eval_lon), eval_lon, result["matched_lon"])
    result["matched_lat"] = np.where(np.isfinite(eval_lat), eval_lat, result["matched_lat"])
    start_column = _first_column(evaluated, ("use_syear", "start_year", "syear"))
    end_column = _first_column(evaluated, ("use_eyear", "end_year", "eyear"))
    result["use_start_year"] = evaluated[start_column].to_numpy() if start_column else np.nan
    result["use_end_year"] = evaluated[end_column].to_numpy() if end_column else np.nan
    status_column = _first_column(evaluated, ("status",))
    result["evaluation_status"] = evaluated[status_column].to_numpy() if status_column else "ok"
    result["metadata_match_status"] = statuses
    return result


def plot_station_map(
    stations: pd.DataFrame,
    path: Path,
    *,
    title: str,
    extent: tuple[float, float, float, float] | None,
    dpi: int,
    label_fontsize: float,
    map_features: Mapping | None = None,
) -> int:
    """Plot every station and full ID. Returns the number of text labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = pd.to_numeric(stations["matched_lon"], errors="coerce").to_numpy(dtype=float)
    lat = pd.to_numeric(stations["matched_lat"], errors="coerce").to_numpy(dtype=float)
    fallback_lon = pd.to_numeric(stations["original_lon"], errors="coerce").to_numpy(dtype=float)
    fallback_lat = pd.to_numeric(stations["original_lat"], errors="coerce").to_numpy(dtype=float)
    lon = np.where(np.isfinite(lon), lon, fallback_lon)
    lat = np.where(np.isfinite(lat), lat, fallback_lat)
    valid = np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise ValueError("No finite station coordinates are available for plotting")

    station_count = int(valid.sum())
    side = min(24.0, max(10.0, 9.0 + np.sqrt(station_count) * 0.22))
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise RuntimeError("Cartopy is required to draw the station information map") from exc

    fig = plt.figure(figsize=(side * 1.25, side))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2f0e9", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf3f8", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#555555")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#888888")
    from openbench.visualization._geo_axes import add_rivers

    add_rivers(ax, map_features or {})
    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        xpad = max(1.0, (np.nanmax(lon[valid]) - np.nanmin(lon[valid])) * 0.08)
        ypad = max(1.0, (np.nanmax(lat[valid]) - np.nanmin(lat[valid])) * 0.08)
        ax.set_extent(
            [np.nanmin(lon[valid]) - xpad, np.nanmax(lon[valid]) + xpad,
             np.nanmin(lat[valid]) - ypad, np.nanmax(lat[valid]) + ypad],
            crs=ccrs.PlateCarree(),
        )
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.35, color="gray", alpha=0.5, linestyle=":")
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax.scatter(lon[valid], lat[valid], s=24, color="#d7301f", edgecolor="white", linewidth=0.45, zorder=5)

    offsets = ((5, 5), (5, -9), (-5, 5), (-5, -9), (9, 0), (-9, 0), (0, 8), (0, -11))
    labels = 0
    for sequence, index in enumerate(np.flatnonzero(valid)):
        dx, dy = offsets[sequence % len(offsets)]
        ax.annotate(
            str(stations.iloc[index]["station_id"]),
            xy=(lon[index], lat[index]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=label_fontsize,
            color="#202020",
            arrowprops={"arrowstyle": "-", "lw": 0.3, "color": "#777777"},
            zorder=6,
        )
        labels += 1
    ax.set_title(f"{title}\nSelected stations: {station_count}", fontsize=13, weight="bold")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(temporary, dpi=dpi, bbox_inches="tight", format=path.suffix.lstrip("."))
        os.replace(temporary, path)
    finally:
        plt.close(fig)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return labels


def _case_dir(cfg, config_path: Path, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    output = Path(str(cfg.project.output_dir)).expanduser()
    candidates = [output / cfg.project.name]
    if not output.is_absolute():
        candidates.append(config_path.parent / output / cfg.project.name)
    existing = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
    return existing.resolve()


def _effective_reference(cfg, resolved):
    from openbench.config.adapter import _apply_reference_override, _source_override, reference_data_dir

    override = _source_override(cfg, resolved.source_name) or _source_override(cfg, resolved.resolved_name)
    ref_ds, var_map = _apply_reference_override(
        resolved.ref_ds, resolved.var_map, resolved.var_name, override
    )
    root, _directory = reference_data_dir(cfg, ref_ds, var_map, override)
    return ref_ds, root


def export(args: argparse.Namespace) -> list[tuple[Path, list[Path]]]:
    from openbench.cli.run import _expand_config_paths
    from openbench.config.adapter import build_fig_nml
    from openbench.config import load_config
    from openbench.config.resolver import resolve_all_references
    from openbench.data.registry.manager import get_registry
    from openbench.data.station_matcher import resolve_station_dataset

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(str(config_path))
    map_features = build_fig_nml().get("map_features", {})
    _expand_config_paths(cfg)
    case_dir = _case_dir(cfg, config_path, args.case_dir)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else case_dir / "data" / "station_info"
    wanted_variables = tuple(args.variables or SEDIMENT_VARIABLES)
    unknown = [name for name in wanted_variables if name not in SEDIMENT_VARIABLES]
    if unknown:
        raise ValueError(f"Not a built-in sediment variable: {', '.join(unknown)}")

    resolved_refs = resolve_all_references(cfg, get_registry(), strict=False)
    outputs: list[tuple[Path, list[Path]]] = []
    for resolved in resolved_refs:
        if resolved.status != "ok" or resolved.var_name not in wanted_variables:
            continue
        if args.reference and args.reference not in {resolved.source_name, resolved.resolved_name}:
            continue
        ref_ds, root = _effective_reference(cfg, resolved)
        matching = getattr(ref_ds, "station_matching", None)
        if matching is None:
            LOG.warning("Skipping %s: reference %s has no station_matching", resolved.var_name, resolved.resolved_name)
            continue
        dataset_path = resolve_station_dataset(root, matching.dataset_file)
        if dataset_path is None:
            raise FileNotFoundError(f"Station reference dataset not found below {root}: {matching.dataset_file}")

        for simulation, sim_entry in cfg.simulation.items():
            if args.simulation and simulation != args.simulation:
                continue
            evaluated_path = None
            for output_reference in dict.fromkeys((resolved.resolved_name, resolved.source_name)):
                try:
                    evaluated_path = find_evaluated_station_file(
                        case_dir, resolved.var_name, output_reference, simulation
                    )
                    break
                except FileNotFoundError:
                    continue
            if evaluated_path is None:
                raise FileNotFoundError(
                    f"No evaluated station file for {resolved.var_name} / {resolved.source_name} / "
                    f"{simulation} under {case_dir}. Run the OpenBench evaluation first."
                )
            evaluated = read_station_csv(evaluated_path)
            metadata = read_reference_station_metadata(
                dataset_path,
                variable=resolved.var_name,
                grid_resolution=sim_entry.grid_res or cfg.project.grid_res,
                station_id_var=matching.station_id_var,
            )
            stations = merge_evaluated_with_metadata(evaluated, metadata)
            stem = join_filename_components(
                resolved.var_name, "ref", resolved.resolved_name, "sim", simulation
            )
            csv_path = output_dir / f"{stem}__stations.csv"
            _atomic_csv(stations, csv_path)

            formats = ("png", "pdf") if args.format == "both" else (args.format,)
            figure_paths = []
            extent = None
            if cfg.project.lon_range is not None and cfg.project.lat_range is not None:
                extent = (
                    float(cfg.project.lon_range[0]), float(cfg.project.lon_range[1]),
                    float(cfg.project.lat_range[0]), float(cfg.project.lat_range[1]),
                )
            for image_format in formats:
                figure_path = output_dir / f"{stem}__station_map.{image_format}"
                plot_station_map(
                    stations,
                    figure_path,
                    title=f"{resolved.var_name} | {resolved.resolved_name} | {simulation}",
                    extent=extent,
                    dpi=args.dpi,
                    label_fontsize=args.label_fontsize,
                    map_features=map_features,
                )
                figure_paths.append(figure_path)
            outputs.append((csv_path, figure_paths))
    if not outputs:
        raise RuntimeError("No matching sediment evaluation outputs were found")
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="OpenBench YAML configuration")
    parser.add_argument("--variable", dest="variables", action="append", help="Sediment variable (repeatable)")
    parser.add_argument("--reference", help="Reference source filter")
    parser.add_argument("--simulation", help="Simulation label filter")
    parser.add_argument("--case-dir", help="Override the OpenBench case output directory")
    parser.add_argument("--output-dir", help="Output directory (default: <case>/data/station_info)")
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="png")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--label-fontsize", type=float, default=6.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    if args.label_fontsize <= 0:
        raise SystemExit("--label-fontsize must be positive")
    outputs = export(args)
    for csv_path, figures in outputs:
        print(csv_path)
        for figure in figures:
            print(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
