#!/usr/bin/env python3
"""Build Lower Mississippi levee-analysis masks from the CoLM 6-arc-minute network.

Without an authoritative Mississippi Alluvial Plain polygon, the script writes
candidate diagnostics only and deliberately does not create the formal target
``lower_mississippi_levee_effect_mask.nc``.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from pathlib import Path
import sys

import numpy as np
import xarray as xr


DEFAULT_ROUTING = Path(
    "/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment/"
    "grid_routing_data_06min.nc"
)
DEFAULT_OUTPUT = Path("outputs/lower_mississippi_levee_effect")
CAIRO_UNIT_ID = 1492387
MOUTH_UNIT_ID = 1537441
TARGET_NLON = 1440
TARGET_NLAT = 720
TARGET_RESOLUTION = 0.25
MAPPING_CHUNK_UNITS = 18000

REQUIRED_VARIABLES = (
    "lon", "lat", "seq", "seq_lon", "seq_lat", "seq_x", "seq_y",
    "seq_next", "topo_area", "topo_uparea", "inpmat_x", "inpmat_y",
    "inpmat_area", "levee_frc", "levee_hgt", "dam_seq",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-file", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument(
        "--alluvial-boundary", type=Path,
        help="Authoritative Mississippi Alluvial Plain polygon (Shapefile/GPKG/GeoJSON).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_1d(ds: xr.Dataset, name: str, dtype: np.dtype) -> np.ndarray:
    values = np.asarray(ds[name].values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; found {ds[name].dims}")
    return values.astype(dtype, copy=False)


def build_topology(nxt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nunit = len(nxt)
    ids = np.arange(1, nunit + 1)
    if np.any(nxt > nunit) or np.any(nxt == ids):
        raise ValueError("seq_next contains an out-of-range ID or self-loop")

    degree = np.bincount(nxt[nxt > 0], minlength=nunit + 1)[1:]
    queue = deque(np.flatnonzero(degree == 0).tolist())
    counts = np.ones(nunit, dtype=np.int64)
    seen = 0
    while queue:
        unit = queue.popleft()
        seen += 1
        downstream = int(nxt[unit])
        if downstream > 0:
            target = downstream - 1
            counts[target] += counts[unit]
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    if seen != nunit:
        raise ValueError(f"seq_next contains cycles affecting {nunit - seen} units")

    first_upstream = np.full(nunit, -1, dtype=np.int64)
    sibling = np.full(nunit, -1, dtype=np.int64)
    for unit, downstream in enumerate(nxt):
        if downstream > 0:
            target = int(downstream) - 1
            sibling[unit] = first_upstream[target]
            first_upstream[target] = unit
    return counts, first_upstream, sibling


def upstream_mask(
    outlet_id: int, first_upstream: np.ndarray, sibling: np.ndarray
) -> np.ndarray:
    nunit = len(first_upstream)
    if not 1 <= outlet_id <= nunit:
        raise ValueError(f"Unit ID {outlet_id} is outside 1..{nunit}")
    selected = np.zeros(nunit, dtype=bool)
    stack = [outlet_id - 1]
    while stack:
        unit = stack.pop()
        if selected[unit]:
            continue
        selected[unit] = True
        child = first_upstream[unit]
        while child >= 0:
            if not selected[child]:
                stack.append(int(child))
            child = sibling[child]
    return selected


def trace_to_terminal(start_id: int, nxt: np.ndarray) -> tuple[int, int, int]:
    unit = start_id - 1
    steps = 0
    while nxt[unit] > 0:
        unit = int(nxt[unit]) - 1
        steps += 1
        if steps > len(nxt):
            raise ValueError("Downstream trace exceeded network size")
    return unit + 1, int(nxt[unit]), steps


def load_boundary_native(path: Path, shape: tuple[int, int]) -> np.ndarray:
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_origin
    except ImportError as exc:
        raise ImportError("Boundary processing requires geopandas and rasterio") from exc

    if not path.is_file():
        raise FileNotFoundError(f"Alluvial Plain boundary not found: {path}")
    frame = gpd.read_file(path)
    if frame.empty or frame.crs is None:
        raise ValueError("Alluvial Plain boundary is empty or has no CRS")
    frame = frame.to_crs("EPSG:4326")
    geometries = [geom for geom in frame.geometry if geom is not None and not geom.is_empty]
    if not geometries:
        raise ValueError("Alluvial Plain boundary has no usable polygon geometry")
    if any(geom.geom_type not in {"Polygon", "MultiPolygon"} for geom in geometries):
        raise ValueError("Alluvial Plain boundary must contain only polygon geometry")

    return rasterize(
        ((geom, 1) for geom in geometries),
        out_shape=shape,
        transform=from_origin(-180.0, 90.0, 0.1, 0.1),
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def aggregate_positive(native: np.ndarray) -> np.ndarray:
    """Area-overlap aggregate a 0.1-degree binary mask to 0.25 degrees."""
    try:
        from rasterio.crs import CRS
        from rasterio.transform import from_origin
        from rasterio.warp import reproject, Resampling
    except ImportError as exc:
        raise ImportError("0.25-degree aggregation requires rasterio") from exc

    coverage = np.zeros((TARGET_NLAT, TARGET_NLON), dtype=np.float32)
    reproject(
        source=native.astype(np.uint8, copy=False),
        destination=coverage,
        src_transform=from_origin(-180.0, 90.0, 0.1, 0.1),
        src_crs=CRS.from_epsg(4326),
        dst_transform=from_origin(-180.0, 90.0, TARGET_RESOLUTION, TARGET_RESOLUTION),
        dst_crs=CRS.from_epsg(4326),
        resampling=Resampling.average,
    )
    return (coverage > 0).astype(np.uint8)


def mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    lon = -180.0 + (cols + 0.5) * TARGET_RESOLUTION
    lat = 90.0 - (rows + 0.5) * TARGET_RESOLUTION
    return float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())


def format_bbox(bbox: tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return "none"
    west, east, south, north = bbox
    return f"lon {west:.3f}..{east:.3f}, lat {south:.3f}..{north:.3f}"


def map_units(
    ds: xr.Dataset,
    hydrologic_units: np.ndarray,
    effective_levee: np.ndarray,
    boundary_native: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    nunit = hydrologic_units.size
    shape = (1800, 3600)
    maps = {
        "hydrologic": np.zeros(shape, dtype=bool),
        "levee_parameter": np.zeros(shape, dtype=bool),
        "hydrologic_levee": np.zeros(shape, dtype=bool),
    }
    alluvial_unit = np.zeros(nunit, dtype=bool) if boundary_native is not None else None

    for start in range(0, nunit, MAPPING_CHUNK_UNITS):
        stop = min(start + MAPPING_CHUNK_UNITS, nunit)
        xx = np.asarray(ds["inpmat_x"].isel(nseqmax=slice(start, stop)).values)
        yy = np.asarray(ds["inpmat_y"].isel(nseqmax=slice(start, stop)).values)
        area = np.asarray(ds["inpmat_area"].isel(nseqmax=slice(start, stop)).values)
        if xx.shape != yy.shape or xx.shape != area.shape or xx.shape[0] != stop - start:
            raise ValueError("inpmat_x/y/area have incompatible shapes")
        good = (
            (xx >= 1) & (xx <= 3600) & (yy >= 1) & (yy <= 1800)
            & np.isfinite(area) & (area > 0)
        )

        selections = {
            "hydrologic": hydrologic_units[start:stop],
            "levee_parameter": effective_levee[start:stop],
            "hydrologic_levee": (
                hydrologic_units[start:stop] & effective_levee[start:stop]
            ),
        }
        for name, selected in selections.items():
            picked = good & selected[:, None]
            maps[name][yy[picked] - 1, xx[picked] - 1] = True

        if boundary_native is not None:
            inside = np.zeros_like(good)
            inside[good] = boundary_native[yy[good] - 1, xx[good] - 1]
            alluvial_unit[start:stop] = np.any(inside, axis=1)

    return maps, alluvial_unit


def write_unit_csv(
    path: Path,
    hydrologic: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    nxt: np.ndarray,
    topo_area: np.ndarray,
    topo_uparea: np.ndarray,
    levee_frc: np.ndarray,
    levee_hgt: np.ndarray,
    raw_levee: np.ndarray,
    reservoir_conflict: np.ndarray,
    effective_levee: np.ndarray,
    alluvial_unit: np.ndarray | None,
) -> None:
    fields = [
        "unit_id_1based", "seq_lon", "seq_lat", "seq_x", "seq_y", "seq_next",
        "topo_area_km2", "topo_uparea_km2", "levee_frc_unprotected_fraction",
        "levee_hgt_m", "raw_levee_valid", "reservoir_conflict",
        "reservoir_aware_levee_valid", "alluvial_boundary_overlap",
        "alluvial_analysis_unit", "levee_core_unit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for unit in np.flatnonzero(hydrologic):
            if alluvial_unit is None:
                overlap: bool | str = ""
                analysis: bool | str = ""
                core: bool | str = ""
            else:
                overlap = bool(alluvial_unit[unit])
                analysis = bool(alluvial_unit[unit])
                core = bool(alluvial_unit[unit] and effective_levee[unit])
            writer.writerow([
                unit + 1, f"{lon[unit]:.7f}", f"{lat[unit]:.7f}",
                int(sx[unit]), int(sy[unit]), int(nxt[unit]),
                f"{topo_area[unit] / 1e6:.6f}",
                f"{topo_uparea[unit] / 1e6:.6f}",
                f"{levee_frc[unit]:.10g}", f"{levee_hgt[unit]:.10g}",
                int(raw_levee[unit]), int(reservoir_conflict[unit]),
                int(effective_levee[unit]), overlap, analysis, core,
            ])


def write_netcdf(
    path: Path,
    target_masks: dict[str, np.ndarray],
    routing_file: Path,
    boundary_file: Path | None,
    counts: dict[str, int],
) -> None:
    lon = -180.0 + (np.arange(TARGET_NLON) + 0.5) * TARGET_RESOLUTION
    lat = 90.0 - (np.arange(TARGET_NLAT) + 0.5) * TARGET_RESOLUTION
    descriptions = {
        "cairo_to_mouth_hydrologic_mask": (
            "Cairo-to-Mississippi-mouth incremental hydrologic footprint"
        ),
        "levee_parameter_mask": (
            "reservoir-aware CoLM units with valid levee parameters"
        ),
        "hydrologic_levee_candidate_mask": (
            "valid levee-parameter footprint inside the Cairo-to-mouth hydrologic range; "
            "candidate only because the Alluvial Plain boundary is unavailable"
        ),
        "alluvial_analysis_mask": (
            "Mississippi Alluvial Plain intersected with the Cairo-to-mouth hydrologic range"
        ),
        "levee_core_mask": (
            "alluvial analysis area intersected with valid CoLM levee parameters; "
            "potential direct-action units, not an observed impact extent"
        ),
    }
    variables = {}
    for name, mask in target_masks.items():
        values = np.asarray(mask, dtype=np.uint8)
        if values.shape != (TARGET_NLAT, TARGET_NLON):
            raise ValueError(f"{name} has incorrect shape {values.shape}")
        if not np.all((values == 0) | (values == 1)):
            raise ValueError(f"{name} contains values other than 0 and 1")
        variables[name] = (
            ("lat", "lon"), values,
            {
                "long_name": descriptions[name],
                "flag_values": np.array([0, 1], dtype=np.uint8),
                "flag_meanings": "outside inside",
            },
        )

    status = "formal" if boundary_file is not None else "candidate_missing_alluvial_boundary"
    dataset = xr.Dataset(
        variables,
        coords={
            "lon": ("lon", lon, {"standard_name": "longitude", "units": "degrees_east"}),
            "lat": ("lat", lat, {"standard_name": "latitude", "units": "degrees_north"}),
        },
        attrs={
            "title": "Lower Mississippi levee-effect analysis masks",
            "product_status": status,
            "source_routing_file": str(routing_file),
            "alluvial_boundary_file": str(boundary_file) if boundary_file else "NOT AVAILABLE",
            "cairo_downstream_unit_id": CAIRO_UNIT_ID,
            "mississippi_mouth_unit_id": MOUTH_UNIT_ID,
            "hydrologic_formula": f"U({MOUTH_UNIT_ID}) minus U({CAIRO_UNIT_ID})",
            "levee_raw_criterion": (
                "finite(levee_frc,levee_hgt) and levee_hgt>0 and 0<=levee_frc<1"
            ),
            "levee_effective_criterion": "raw criterion and unit_id not in positive dam_seq",
            "native_mapping": "original inpmat_x/inpmat_y/inpmat_area on 0.1 degree grid",
            "target_aggregation": (
                "area-overlap average from 0.1 to 0.25 degrees; binary is positive overlap"
            ),
            "interpretation_warning": (
                "levee masks identify potential direct parameter action, not an observed "
                "or simulated levee-effect extent"
            ),
            **{f"count_{key}": int(value) for key, value in counts.items()},
        },
    )
    encoding = {
        name: {"dtype": "u1", "zlib": True, "complevel": 4, "_FillValue": None}
        for name in variables
    }
    dataset.to_netcdf(path, engine="netcdf4", encoding=encoding)


def write_preview(
    path: Path,
    masks: dict[str, np.ndarray],
    lon_units: np.ndarray,
    lat_units: np.ndarray,
    hydrologic: np.ndarray,
    effective_levee: np.ndarray,
    candidate: bool,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    hydro = masks["cairo_to_mouth_hydrologic_mask"]
    extent = (-180.0, 180.0, -90.0, 90.0)
    ax.imshow(
        np.ma.masked_where(hydro == 0, hydro), origin="upper", extent=extent,
        cmap=ListedColormap(["#b9d9f3"]), interpolation="nearest", alpha=0.7,
    )
    if candidate:
        overlay = masks["hydrologic_levee_candidate_mask"]
        title = "Lower Mississippi levee candidate — missing Alluvial Plain boundary"
    else:
        overlay = masks["levee_core_mask"]
        title = "Lower Mississippi alluvial analysis and core levee masks"
    ax.imshow(
        np.ma.masked_where(overlay == 0, overlay), origin="upper", extent=extent,
        cmap=ListedColormap(["#d62728"]), interpolation="nearest", alpha=0.9,
    )
    selected = hydrologic & effective_levee
    ax.scatter(lon_units[selected], lat_units[selected], s=2, c="black", alpha=0.35,
               label="effective levee unit centers")
    ax.scatter([-89.15, -89.45], [36.95, 28.95], c=["#ff7f0e", "#2ca02c"],
               s=45, marker="x", linewidths=2, label="Cairo / mouth controls")
    ax.set_xlim(-108.0, -87.0)
    ax.set_ylim(28.0, 40.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    routing_file: Path,
    boundary_file: Path | None,
    stats: dict[str, int],
    bboxes: dict[str, tuple[float, float, float, float] | None],
    terminal_steps: int,
    source_before: tuple[int, int],
    source_after: tuple[int, int],
    nc_name: str,
    levee_summary: dict[str, float | int],
) -> None:
    candidate = boundary_file is None
    status_text = (
        "候选诊断：本地没有可靠的 Mississippi Alluvial Plain GIS 边界，正式目标文件未生成。"
        if candidate else
        "正式产品：使用了所提供的权威 Mississippi Alluvial Plain 多边形。"
    )
    verification = (
        "Alluvial Plain 与 levee core 尚待真实边界验证。"
        if candidate else
        "Alluvial Plain 和 levee core 已使用所提供边界完成数据验证。"
    )
    lines = [
        "# Lower Mississippi levee-effect mask diagnostic",
        "",
        f"**状态：** {status_text}",
        "",
        "## 输入与判据",
        "",
        f"- Routing file: `{routing_file}`",
        "- 原生 routing/inpmat 网格：3600 × 1800，0.1°（6 arc minutes）。",
        "- 输出分析网格：1440 × 720，0.25°。",
        f"- Cairo 下游 unit: `{CAIRO_UNIT_ID}`；Mississippi mouth unit: `{MOUTH_UNIT_ID}`。",
        f"- Cairo 沿 `seq_next` 到河口：{terminal_steps} steps。",
        "- 原始 levee 有效判据：`finite(frc,hgt) & hgt>0 & 0<=frc<1`。",
        "- Reservoir-aware 判据：再排除正 `dam_seq` 对应的 unit。",
        "- CoLM 中 `levee_frc` 是未保护比例，`levee_hgt` 单位为 m。",
        f"- `levee_frc` shape: ({stats['total_units']:,},)，范围 "
        f"{levee_summary['frc_min']:.6g}..{levee_summary['frc_max']:.6g}，"
        f"非有限值 {levee_summary['frc_nonfinite']:,}。",
        f"- `levee_hgt` shape: ({stats['total_units']:,},)，范围 "
        f"{levee_summary['hgt_min']:.6g}..{levee_summary['hgt_max']:.6g} m，"
        f"非有限值 {levee_summary['hgt_nonfinite']:,}。",
        f"- Global raw-valid levee units: {stats['global_raw_levee_units']:,}；"
        f"reservoir conflicts: {stats['global_reservoir_conflicts']:,}；"
        f"reservoir-aware valid units: {stats['global_effective_levee_units']:,}。",
        "",
        "## 统计",
        "",
        f"- Cairo–mouth hydrologic units: {stats['hydrologic_units']:,}",
        f"- Hydrologic raw-valid levee units: {stats['hydrologic_raw_levee_units']:,}",
        f"- Hydrologic reservoir conflicts: {stats['hydrologic_reservoir_conflicts']:,}",
        f"- Hydrologic reservoir-aware levee units: {stats['hydrologic_effective_levee_units']:,}",
        f"- Native 0.1° hydrologic-levee cells: {stats['native_hydrologic_levee_cells']:,}",
        f"- Target 0.25° hydrologic-levee cells: {stats['target_hydrologic_levee_cells']:,}",
        f"- Hydrologic 0.25° extent: {format_bbox(bboxes['hydrologic'])}",
        f"- Hydrologic-levee 0.25° extent: {format_bbox(bboxes['hydrologic_levee'])}",
    ]
    if not candidate:
        lines.extend([
            f"- Alluvial analysis units: {stats['alluvial_analysis_units']:,}",
            f"- Core levee units: {stats['levee_core_units']:,}",
            f"- Alluvial analysis cells: {stats['alluvial_analysis_cells']:,}",
            f"- Core levee cells: {stats['levee_core_cells']:,}",
            f"- Alluvial analysis extent: {format_bbox(bboxes['alluvial_analysis'])}",
            f"- Core levee extent: {format_bbox(bboxes['levee_core'])}",
        ])
    lines.extend([
        "",
        "## 输出与验证状态",
        "",
        f"- NetCDF: `{nc_name}`",
        "- 所有输出掩膜均为 0/1，并与 0.25° 全球网格一致。",
        "- unit 到原生网格的映射严格使用 `inpmat_x/y/area` 的有效正权重。",
        "- 0.1° 到 0.25° 使用面积重叠聚合，正重叠记为 1。",
        f"- 原始 routing 文件大小/mtime_ns 前后保持一致：{source_before == source_after}。",
        f"- {verification}",
        "- `levee_core_mask`（若生成）仅表示潜在直接参数作用单元，不代表实验测得的影响范围。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    routing_file = args.routing_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    boundary_file = (
        args.alluvial_boundary.expanduser().resolve() if args.alluvial_boundary else None
    )
    if not routing_file.is_file():
        raise FileNotFoundError(f"Routing file not found: {routing_file}")
    if boundary_file is not None and not boundary_file.is_file():
        raise FileNotFoundError(f"Alluvial Plain boundary not found: {boundary_file}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stat = routing_file.stat()
    source_before = (source_stat.st_size, source_stat.st_mtime_ns)

    with xr.open_dataset(routing_file, decode_cf=False, mask_and_scale=False) as ds:
        missing = [name for name in REQUIRED_VARIABLES if name not in ds]
        if missing:
            raise ValueError(f"Routing file is missing required variables: {missing}")
        if ds.sizes.get("nx") != 3600 or ds.sizes.get("ny") != 1800:
            raise ValueError(f"Expected a 3600x1800 6-minute grid; found {dict(ds.sizes)}")

        seq = read_1d(ds, "seq", np.int64)
        nunit = seq.size
        if not np.array_equal(seq, np.arange(1, nunit + 1)):
            raise ValueError("seq is not the expected contiguous 1-based unit ID")
        lon = read_1d(ds, "seq_lon", np.float64)
        lat = read_1d(ds, "seq_lat", np.float64)
        sx = read_1d(ds, "seq_x", np.int64)
        sy = read_1d(ds, "seq_y", np.int64)
        nxt = read_1d(ds, "seq_next", np.int64)
        topo_area = read_1d(ds, "topo_area", np.float64)
        topo_uparea = read_1d(ds, "topo_uparea", np.float64)
        levee_frc = read_1d(ds, "levee_frc", np.float64)
        levee_hgt = read_1d(ds, "levee_hgt", np.float64)
        vectors = (lon, lat, sx, sy, nxt, topo_area, topo_uparea, levee_frc, levee_hgt)
        if any(len(vector) != nunit for vector in vectors):
            raise ValueError("Routing vectors do not share the nseqmax dimension")
        if not np.all((sx >= 1) & (sx <= 3600) & (sy >= 1) & (sy <= 1800)):
            raise ValueError("seq_x/seq_y contain indices outside the native grid")
        if str(ds["levee_hgt"].attrs.get("units", "")).strip() != "m":
            raise ValueError("levee_hgt units are not confirmed as meters")

        counts, first_upstream, sibling = build_topology(nxt)
        cairo_upstream = upstream_mask(CAIRO_UNIT_ID, first_upstream, sibling)
        mouth_upstream = upstream_mask(MOUTH_UNIT_ID, first_upstream, sibling)
        if np.any(cairo_upstream & ~mouth_upstream):
            raise ValueError("Cairo upstream set is not nested inside the mouth set")
        hydrologic = mouth_upstream & ~cairo_upstream
        if int(cairo_upstream.sum()) != int(counts[CAIRO_UNIT_ID - 1]):
            raise ValueError("Cairo reverse traversal count disagrees with topology")
        if int(mouth_upstream.sum()) != int(counts[MOUTH_UNIT_ID - 1]):
            raise ValueError("Mouth reverse traversal count disagrees with topology")
        terminal_id, terminal_code, terminal_steps = trace_to_terminal(CAIRO_UNIT_ID, nxt)
        if terminal_id != MOUTH_UNIT_ID or terminal_code != -9:
            raise ValueError(
                f"Cairo trace reaches {terminal_id} (code {terminal_code}), not the approved mouth"
            )

        raw_levee = (
            np.isfinite(levee_frc) & np.isfinite(levee_hgt)
            & (levee_hgt > 0) & (levee_frc >= 0) & (levee_frc < 1)
        )
        dam_seq = read_1d(ds, "dam_seq", np.int64)
        positive_dams = np.unique(dam_seq[(dam_seq >= 1) & (dam_seq <= nunit)])
        is_reservoir = np.isin(seq, positive_dams)
        reservoir_conflict = raw_levee & is_reservoir
        effective_levee = raw_levee & ~is_reservoir

        boundary_native = (
            load_boundary_native(boundary_file, (1800, 3600))
            if boundary_file is not None else None
        )
        native_maps, alluvial_unit = map_units(
            ds, hydrologic, effective_levee, boundary_native
        )

    target_masks = {
        "cairo_to_mouth_hydrologic_mask": aggregate_positive(native_maps["hydrologic"]),
        "levee_parameter_mask": aggregate_positive(native_maps["levee_parameter"]),
    }
    hydrologic_levee_target = aggregate_positive(native_maps["hydrologic_levee"])
    if boundary_native is None:
        target_masks["hydrologic_levee_candidate_mask"] = hydrologic_levee_target
        nc_name = "lower_mississippi_levee_effect_candidate.nc"
        png_name = "lower_mississippi_levee_effect_candidate.png"
    else:
        analysis_native = boundary_native & native_maps["hydrologic"]
        core_native = analysis_native & native_maps["levee_parameter"]
        target_masks["alluvial_analysis_mask"] = aggregate_positive(analysis_native)
        target_masks["levee_core_mask"] = aggregate_positive(core_native)
        nc_name = "lower_mississippi_levee_effect_mask.nc"
        png_name = "lower_mississippi_levee_effect_mask.png"

    stats = {
        "total_units": int(nunit),
        "global_raw_levee_units": int(raw_levee.sum()),
        "global_reservoir_conflicts": int(reservoir_conflict.sum()),
        "global_effective_levee_units": int(effective_levee.sum()),
        "hydrologic_units": int(hydrologic.sum()),
        "hydrologic_raw_levee_units": int((hydrologic & raw_levee).sum()),
        "hydrologic_reservoir_conflicts": int((hydrologic & reservoir_conflict).sum()),
        "hydrologic_effective_levee_units": int((hydrologic & effective_levee).sum()),
        "native_hydrologic_cells": int(native_maps["hydrologic"].sum()),
        "native_hydrologic_levee_cells": int(native_maps["hydrologic_levee"].sum()),
        "target_hydrologic_cells": int(target_masks["cairo_to_mouth_hydrologic_mask"].sum()),
        "target_global_levee_cells": int(target_masks["levee_parameter_mask"].sum()),
        "target_hydrologic_levee_cells": int(hydrologic_levee_target.sum()),
    }
    if alluvial_unit is not None:
        stats.update({
            "alluvial_analysis_units": int((hydrologic & alluvial_unit).sum()),
            "levee_core_units": int((hydrologic & alluvial_unit & effective_levee).sum()),
            "alluvial_analysis_cells": int(target_masks["alluvial_analysis_mask"].sum()),
            "levee_core_cells": int(target_masks["levee_core_mask"].sum()),
        })
        if stats["levee_core_units"] == 0 or stats["levee_core_cells"] == 0:
            raise ValueError("The supplied boundary produces an empty levee core")

    nc_path = output_dir / nc_name
    write_netcdf(nc_path, target_masks, routing_file, boundary_file, stats)
    csv_path = output_dir / "lower_mississippi_levee_effect_unit_statistics.csv"
    write_unit_csv(
        csv_path, hydrologic, lon, lat, sx, sy, nxt, topo_area, topo_uparea,
        levee_frc, levee_hgt, raw_levee, reservoir_conflict, effective_levee,
        alluvial_unit,
    )
    png_path = output_dir / png_name
    write_preview(
        png_path, target_masks, lon, lat, hydrologic, effective_levee,
        candidate=boundary_file is None,
    )

    source_stat = routing_file.stat()
    source_after = (source_stat.st_size, source_stat.st_mtime_ns)
    if source_before != source_after:
        raise RuntimeError("Source routing file size or mtime changed during processing")
    bboxes = {
        "hydrologic": mask_bbox(target_masks["cairo_to_mouth_hydrologic_mask"]),
        "hydrologic_levee": mask_bbox(hydrologic_levee_target),
    }
    if boundary_file is not None:
        bboxes["alluvial_analysis"] = mask_bbox(target_masks["alluvial_analysis_mask"])
        bboxes["levee_core"] = mask_bbox(target_masks["levee_core_mask"])
    report_path = output_dir / "diagnostic_report.md"
    write_report(
        report_path, routing_file, boundary_file, stats, bboxes, terminal_steps,
        source_before, source_after, nc_name,
        {
            "frc_min": float(np.nanmin(levee_frc)),
            "frc_max": float(np.nanmax(levee_frc)),
            "frc_nonfinite": int((~np.isfinite(levee_frc)).sum()),
            "hgt_min": float(np.nanmin(levee_hgt)),
            "hgt_max": float(np.nanmax(levee_hgt)),
            "hgt_nonfinite": int((~np.isfinite(levee_hgt)).sum()),
        },
    )

    formal_path = output_dir / "lower_mississippi_levee_effect_mask.nc"
    if boundary_file is None and formal_path.exists():
        raise RuntimeError(
            "Candidate run found a formal target file in the output directory; "
            "remove or relocate it to avoid confusing product status"
        )
    print(f"Wrote {nc_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    print("Status:", "formal" if boundary_file else "candidate_missing_alluvial_boundary")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        sys.exit(f"ERROR: {exc}")
