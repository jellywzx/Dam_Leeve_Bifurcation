#!/usr/bin/env python3
"""Unified production workflow for Amazon, Missouri, Lower Mississippi and Mekong."""

from collections import deque
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np

from basin_mask_config import BasinRunConfig
# On node111, initialize matplotlib before xarray loads the system libstdc++
# through a NetCDF backend; otherwise matplotlib's compiled extension cannot load.
from basin_mask_plot import plot_basin_mask

import xarray as xr


PRODUCT_TITLES = {
    "amazon": "Amazon basin",
    "missouri": "Missouri basin before Mississippi confluence",
    "lower_mississippi": "Lower Mississippi incremental basin (Cairo to mouth)",
    "mekong": "Mekong main-mouth basin",
    "amazon_mainstem_mouth": "Amazon mainstem-mouth basin",
    "missouri_hermann": "Missouri basin upstream of Hermann",
    "missouri_before_confluence": "Missouri basin before Mississippi confluence",
    "cairo_below_confluence": "Mississippi-Ohio basin upstream of Cairo boundary",
    "mississippi_vicksburg": "Mississippi basin upstream of Vicksburg",
    "mississippi_main_mouth": "Mississippi main-mouth basin",
    "mississippi_cairo_to_vicksburg_added_catchments": (
        "Mississippi incremental basin (Cairo to Vicksburg)"
    ),
    "mississippi_cairo_to_mouth_added_catchments": (
        "Mississippi incremental basin (Cairo to mouth)"
    ),
    "mekong_stungtreng": "Mekong basin upstream of Stung Treng",
    "mekong_main_mouth": "Mekong main-mouth basin",
}


def _read_vector(ds: xr.Dataset, name: str, dtype: np.dtype) -> np.ndarray:
    if name not in ds:
        raise ValueError(f"Routing file is missing required variable {name!r}")
    values = np.asarray(ds[name].values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; found {ds[name].dims}")
    return values.astype(dtype, copy=False)


def build_topology(nxt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upstream counts and a compact reverse adjacency list."""
    nunit = len(nxt)
    ids = np.arange(1, nunit + 1)
    if np.any(nxt > nunit) or np.any(nxt == ids):
        raise ValueError("seq_next contains an out-of-range ID or self-loop")

    indegree = np.bincount(nxt[nxt > 0], minlength=nunit + 1)[1:]
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    counts = np.ones(nunit, dtype=np.int64)
    seen = 0
    while queue:
        unit = queue.popleft()
        seen += 1
        downstream = int(nxt[unit])
        if downstream > 0:
            target = downstream - 1
            counts[target] += counts[unit]
            indegree[target] -= 1
            if indegree[target] == 0:
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
    outlet_id: int,
    first_upstream: np.ndarray,
    sibling: np.ndarray,
) -> np.ndarray:
    """Select an outlet and every unit that drains to it through seq_next."""
    nunit = len(first_upstream)
    if not 1 <= int(outlet_id) <= nunit:
        raise ValueError(f"Outlet ID {outlet_id} is outside 1..{nunit}")
    selected = np.zeros(nunit, dtype=bool)
    stack = [int(outlet_id) - 1]
    while stack:
        unit = stack.pop()
        if selected[unit]:
            continue
        selected[unit] = True
        upstream = int(first_upstream[unit])
        while upstream >= 0:
            if not selected[upstream]:
                stack.append(upstream)
            upstream = int(sibling[upstream])
    return selected


def _read_map_chunk(
    ds: xr.Dataset,
    name: str,
    nunit: int,
    start: int,
    stop: int,
) -> np.ndarray:
    variable = ds[name]
    if variable.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; found {variable.dims}")
    axes = [axis for axis, size in enumerate(variable.shape) if size == nunit]
    if len(axes) != 1:
        raise ValueError(f"Cannot identify unit axis in {name} shape {variable.shape}")
    axis = axes[0]
    values = np.asarray(
        variable.isel({variable.dims[axis]: slice(start, stop)}).values
    )
    return values if axis == 0 else values.T


def _map_selected_weights(
    ds: xr.Dataset,
    selections: Mapping[str, np.ndarray],
    nlon: int,
    nlat: int,
    chunk_units: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Accumulate all selected mapping weights in one inpmat pass."""
    nunit = next(iter(selections.values())).size
    total_weight = np.zeros((nlat, nlon), dtype=np.float64)
    selected_weight = {
        name: np.zeros((nlat, nlon), dtype=np.float64) for name in selections
    }

    for start in range(0, nunit, chunk_units):
        stop = min(start + chunk_units, nunit)
        xx = _read_map_chunk(ds, "inpmat_x", nunit, start, stop).astype(np.int64)
        yy = _read_map_chunk(ds, "inpmat_y", nunit, start, stop).astype(np.int64)
        weight = _read_map_chunk(ds, "inpmat_area", nunit, start, stop).astype(
            np.float64
        )
        if xx.shape != yy.shape or xx.shape != weight.shape:
            raise ValueError("inpmat_x/y/area shapes do not match")
        valid = (
            (xx >= 1)
            & (xx <= nlon)
            & (yy >= 1)
            & (yy <= nlat)
            & np.isfinite(weight)
            & (weight > 0)
        )
        np.add.at(total_weight, (yy[valid] - 1, xx[valid] - 1), weight[valid])
        for name, selected in selections.items():
            chosen = valid & selected[start:stop, None]
            np.add.at(
                selected_weight[name],
                (yy[chosen] - 1, xx[chosen] - 1),
                weight[chosen],
            )

    for name, numerator in selected_weight.items():
        if np.any(numerator > total_weight * (1 + 1e-8) + 1e-7):
            raise ValueError(f"{name}: selected mapping weights exceed total weights")
    return total_weight, selected_weight


def map_selected_units(
    ds: xr.Dataset,
    selections: Mapping[str, np.ndarray],
    nlon: int,
    nlat: int,
    chunk_units: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Map every selected unit set in one pass through the inpmat arrays."""
    total_weight, selected_weight = _map_selected_weights(
        ds, selections, nlon, nlat, chunk_units
    )
    mapped = {}
    for name, numerator in selected_weight.items():
        share = np.divide(
            numerator,
            total_weight,
            out=np.zeros_like(numerator),
            where=total_weight > 0,
        )
        mapped[name] = (share, numerator)
    return mapped


def _write_geotiff(path: Path, mask: np.ndarray, nlon: int, nlat: int) -> bool:
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        print("  INFO: rasterio is unavailable; GeoTIFF skipped")
        return False
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=nlon,
        height=nlat,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(-180, 90, 360 / nlon, 180 / nlat),
        compress="deflate",
    ) as output:
        output.write(mask.astype(np.uint8), 1)
    print(f"  GeoTIFF -> {path}")
    return True


def _export_product(
    name: str,
    selected: np.ndarray,
    outlet_ids: List[int],
    definition: str,
    config: BasinRunConfig,
    output_dir: Path,
    sx: np.ndarray,
    sy: np.ndarray,
    nxt: np.ndarray,
    unit_lon: np.ndarray,
    unit_lat: np.ndarray,
    topo_uparea: np.ndarray,
    fraction: np.ndarray,
    selected_weight: np.ndarray,
) -> Dict[str, object]:
    folder = output_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    lon = -180.0 + (np.arange(config.nlon) + 0.5) * 360.0 / config.nlon
    lat = 90.0 - (np.arange(config.nlat) + 0.5) * 180.0 / config.nlat
    binary = fraction > 0
    centers = np.zeros((config.nlat, config.nlon), dtype=np.int32)
    np.add.at(centers, (sy[selected] - 1, sx[selected] - 1), 1)

    dataset = xr.Dataset(
        {
            "basin_mask": (
                ("lat", "lon"),
                binary.astype(np.uint8),
                {"long_name": "positive CoLM inpmat-weight footprint", "units": "1"},
            ),
            "basin_mapping_weight_share": (
                ("lat", "lon"),
                fraction.astype(np.float32),
                {"long_name": "selected share of positive CoLM mapping weights", "units": "1"},
            ),
            "upstream_river_unit_count": (
                ("lat", "lon"),
                centers,
                {"long_name": "selected unit-catchment centers in grid cell", "units": "1"},
            ),
            "selected_inpmat_weight_sum": (
                ("lat", "lon"),
                selected_weight.astype(np.float32),
                {"long_name": "sum of selected original inpmat_area weights", "units": "1"},
            ),
        },
        coords={"lon": ("lon", lon), "lat": ("lat", lat)},
        attrs={
            "basin_name": name,
            "resolution_label": config.label,
            "definition": definition,
            "source_routing_file": str(config.routing_file),
            "network_definition": "CoLM primary seq_next upstream tree",
            "bifurcation_caveat": "Dynamic bifurcation paths are not part of this static mask",
            "selected_unit_count": int(selected.sum()),
            "selected_grid_cell_count": int(binary.sum()),
            "outlet_or_boundary_unit_ids": ",".join(map(str, outlet_ids)),
            "mapping_weight_caveat": "inpmat_area has units 1 and is not physical area",
        },
    )
    nc_path = folder / f"{name}_basin.nc"
    dataset.to_netcdf(nc_path)
    print(f"  NetCDF -> {nc_path}")

    csv_path = folder / f"{name}_upstream_units.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "unit_id_1based",
                "seq_x",
                "seq_y",
                "seq_next",
                "seq_lon",
                "seq_lat",
                "topo_uparea_km2",
            ]
        )
        for index in np.flatnonzero(selected):
            writer.writerow(
                [
                    int(index + 1),
                    int(sx[index]),
                    int(sy[index]),
                    int(nxt[index]),
                    f"{unit_lon[index]:.7f}",
                    f"{unit_lat[index]:.7f}",
                    f"{topo_uparea[index] / 1e6:.6f}",
                ]
            )
    print(f"  Unit table -> {csv_path}")

    summary = {
        "basin_name": name,
        "resolution": config.label,
        "definition": definition,
        "source_routing_file": str(config.routing_file),
        "outlet_or_boundary_unit_ids": outlet_ids,
        "selected_unit_count": int(selected.sum()),
        "selected_grid_cell_count": int(binary.sum()),
        "netcdf": str(nc_path),
    }
    summary_path = folder / f"{name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tif_path = folder / f"{name}_basin_mask.tif"
    _write_geotiff(tif_path, binary, config.nlon, config.nlat)
    points = [(float(unit_lon[unit_id - 1]), float(unit_lat[unit_id - 1])) for unit_id in outlet_ids]
    plot_basin_mask(
        fraction,
        lon,
        lat,
        folder / f"{name}_preview.png",
        f"{PRODUCT_TITLES[name]} ({config.label})",
        points,
    )
    dataset.close()
    return summary


def _validate_config(
    config: BasinRunConfig,
    ds: xr.Dataset,
    nunit: int,
    required_ids: set,
) -> None:
    missing = required_ids - set(config.outlet_ids)
    if missing:
        raise ValueError(f"{config.label}: missing configured outlet IDs: {sorted(missing)}")
    invalid = {name: value for name, value in config.outlet_ids.items() if not 1 <= value <= nunit}
    if invalid:
        raise ValueError(f"{config.label}: outlet IDs outside routing range: {invalid}")
    actual = (ds.sizes.get("nx"), ds.sizes.get("ny"))
    expected = (config.nlon, config.nlat)
    if actual != expected:
        raise ValueError(f"{config.label}: routing grid is {actual}, expected {expected}")


def _require_nested(
    masks: Mapping[str, np.ndarray],
    inner: str,
    outer: str,
) -> None:
    if np.any(masks[inner] & ~masks[outer]):
        raise ValueError(f"{inner} upstream set is not nested in {outer}")


def _standard_products(endpoint_masks, ids):
    _require_nested(
        endpoint_masks, "cairo_below_confluence", "mississippi_main_mouth"
    )
    selections = {
        "amazon": endpoint_masks["amazon_main_mouth"],
        "missouri": endpoint_masks["missouri_before_confluence"],
        "lower_mississippi": (
            endpoint_masks["mississippi_main_mouth"]
            & ~endpoint_masks["cairo_below_confluence"]
        ),
        "mekong": endpoint_masks["mekong_main_mouth"],
    }
    specs = {
        "amazon": ([ids["amazon_main_mouth"]], "U(Amazon main mouth)"),
        "missouri": (
            [ids["missouri_before_confluence"]],
            "U(Missouri unit immediately before Mississippi confluence)",
        ),
        "lower_mississippi": (
            [ids["mississippi_main_mouth"], ids["cairo_below_confluence"]],
            "U(Mississippi main mouth) minus U(just below Cairo confluence); "
            "not USGS HUC08",
        ),
        "mekong": ([ids["mekong_main_mouth"]], "U(Mekong main-network mouth)"),
    }
    return ("amazon", "missouri", "lower_mississippi", "mekong"), selections, specs


def _part_products(endpoint_masks, ids):
    nesting = (
        ("missouri_hermann", "missouri_before_confluence"),
        ("cairo_below_confluence", "mississippi_vicksburg"),
        ("mississippi_vicksburg", "mississippi_main_mouth"),
        ("mekong_stungtreng", "mekong_main_mouth"),
    )
    for inner, outer in nesting:
        _require_nested(endpoint_masks, inner, outer)

    cairo = endpoint_masks["cairo_below_confluence"]
    cairo_to_vicksburg = endpoint_masks["mississippi_vicksburg"] & ~cairo
    cairo_to_mouth = endpoint_masks["mississippi_main_mouth"] & ~cairo
    if not np.any(cairo_to_vicksburg) or not np.any(cairo_to_mouth):
        raise ValueError("A configured Mississippi incremental mask is empty")
    if np.any(cairo_to_vicksburg & cairo) or np.any(cairo_to_mouth & cairo):
        raise ValueError("A Mississippi incremental mask overlaps Cairo upstream")

    selections = {
        "amazon_mainstem_mouth": endpoint_masks["amazon_main_mouth"],
        "missouri_hermann": endpoint_masks["missouri_hermann"],
        "missouri_before_confluence": endpoint_masks["missouri_before_confluence"],
        "cairo_below_confluence": cairo,
        "mississippi_vicksburg": endpoint_masks["mississippi_vicksburg"],
        "mississippi_main_mouth": endpoint_masks["mississippi_main_mouth"],
        "mississippi_cairo_to_vicksburg_added_catchments": cairo_to_vicksburg,
        "mississippi_cairo_to_mouth_added_catchments": cairo_to_mouth,
        "mekong_stungtreng": endpoint_masks["mekong_stungtreng"],
        "mekong_main_mouth": endpoint_masks["mekong_main_mouth"],
    }
    specs = {
        "amazon_mainstem_mouth": (
            [ids["amazon_main_mouth"]],
            "U(Amazon mainstem mouth)",
        ),
        "missouri_hermann": (
            [ids["missouri_hermann"]],
            "U(Missouri at Hermann)",
        ),
        "missouri_before_confluence": (
            [ids["missouri_before_confluence"]],
            "U(Missouri unit immediately before Mississippi confluence)",
        ),
        "cairo_below_confluence": (
            [ids["cairo_below_confluence"]],
            "U(just below the Mississippi-Ohio confluence near Cairo)",
        ),
        "mississippi_vicksburg": (
            [ids["mississippi_vicksburg"]],
            "U(Mississippi at Vicksburg)",
        ),
        "mississippi_main_mouth": (
            [ids["mississippi_main_mouth"]],
            "U(Mississippi main mouth)",
        ),
        "mississippi_cairo_to_vicksburg_added_catchments": (
            [ids["mississippi_vicksburg"], ids["cairo_below_confluence"]],
            "U(Vicksburg) minus U(just below Cairo confluence); not USGS HUC08",
        ),
        "mississippi_cairo_to_mouth_added_catchments": (
            [ids["mississippi_main_mouth"], ids["cairo_below_confluence"]],
            "U(Mississippi main mouth) minus U(just below Cairo confluence); "
            "not USGS HUC08",
        ),
        "mekong_stungtreng": (
            [ids["mekong_stungtreng"]],
            "U(Mekong at Stung Treng)",
        ),
        "mekong_main_mouth": (
            [ids["mekong_main_mouth"]],
            "U(Mekong main-network mouth)",
        ),
    }
    order = (
        "amazon_mainstem_mouth",
        "missouri_hermann",
        "missouri_before_confluence",
        "cairo_below_confluence",
        "mississippi_vicksburg",
        "mississippi_main_mouth",
        "mississippi_cairo_to_vicksburg_added_catchments",
        "mississippi_cairo_to_mouth_added_catchments",
        "mekong_stungtreng",
        "mekong_main_mouth",
    )
    return order, selections, specs


def _run_products(config, detailed_parts):
    if detailed_parts:
        required_ids = {
            "amazon_main_mouth",
            "missouri_hermann",
            "missouri_before_confluence",
            "cairo_below_confluence",
            "mississippi_vicksburg",
            "mississippi_main_mouth",
            "mekong_stungtreng",
            "mekong_main_mouth",
        }
        output_dir = config.parts_output_dir
        product_set = "detailed_basin_parts"
    else:
        required_ids = {
            "amazon_main_mouth",
            "missouri_before_confluence",
            "cairo_below_confluence",
            "mississippi_main_mouth",
            "mekong_main_mouth",
        }
        output_dir = config.output_dir
        product_set = "standard_four_basins"

    source = config.routing_file.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Routing file not found: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    before = (source.stat().st_size, source.stat().st_mtime_ns)

    label = "detailed basin parts" if detailed_parts else "basin masks"
    print(f"=== CoLM {label}: {config.label} ===")
    print(f"Routing file: {source}")
    with xr.open_dataset(source, decode_cf=False, mask_and_scale=False) as ds:
        sx = _read_vector(ds, "seq_x", np.int64)
        sy = _read_vector(ds, "seq_y", np.int64)
        nxt = _read_vector(ds, "seq_next", np.int64)
        unit_lon = _read_vector(ds, "seq_lon", np.float64)
        unit_lat = _read_vector(ds, "seq_lat", np.float64)
        topo_uparea = _read_vector(ds, "topo_uparea", np.float64)
        nunit = len(nxt)
        if any(len(values) != nunit for values in (sx, sy, unit_lon, unit_lat, topo_uparea)):
            raise ValueError("Routing vectors do not share the nseqmax dimension")
        if str(ds["topo_uparea"].attrs.get("units", "")).strip().lower() not in {
            "m2",
            "m^2",
            "m**2",
            "m²",
        }:
            raise ValueError("topo_uparea units are not confirmed as square meters")
        if str(ds["inpmat_area"].attrs.get("units", "")).strip() != "1":
            raise ValueError("inpmat_area units are not confirmed as dimensionless")
        _validate_config(config, ds, nunit, required_ids)
        counts, first_upstream, sibling = build_topology(nxt)
        ids = config.outlet_ids

        for key in ("amazon_main_mouth", "mississippi_main_mouth", "mekong_main_mouth"):
            if int(nxt[ids[key] - 1]) != -9:
                raise ValueError(f"{config.label}: {key} ID {ids[key]} is not seq_next=-9")
        if int(nxt[ids["missouri_before_confluence"] - 1]) <= 0:
            raise ValueError("Missouri pre-confluence outlet does not flow to another unit")

        endpoint_masks = {
            key: upstream_mask(unit_id, first_upstream, sibling)
            for key, unit_id in ids.items()
            if key in required_ids
        }
        for key in required_ids:
            unit_id = ids[key]
            actual_count = int(endpoint_masks[key].sum())
            expected_count = int(counts[unit_id - 1])
            if actual_count != expected_count:
                raise ValueError(
                    f"{key}: reverse trace selected {actual_count}, topology says {expected_count}"
                )

        if detailed_parts:
            order, selections, product_specs = _part_products(endpoint_masks, ids)
        else:
            order, selections, product_specs = _standard_products(
                endpoint_masks, ids
            )
        if any(not np.any(selected) for selected in selections.values()):
            raise ValueError("At least one configured basin is empty")

        total_weight, selected_weights = _map_selected_weights(
            ds, selections, config.nlon, config.nlat, config.chunk_units
        )
        summaries = []
        for name in order:
            print(f"\n--- {name} ---")
            outlet_ids, definition = product_specs[name]
            selected_weight = selected_weights[name]
            fraction = np.divide(
                selected_weight,
                total_weight,
                out=np.zeros_like(selected_weight),
                where=total_weight > 0,
            )
            summaries.append(
                _export_product(
                    name,
                    selections[name],
                    outlet_ids,
                    definition,
                    config,
                    output_dir,
                    sx,
                    sy,
                    nxt,
                    unit_lon,
                    unit_lat,
                    topo_uparea,
                    fraction,
                    selected_weight,
                )
            )

    after = (source.stat().st_size, source.stat().st_mtime_ns)
    if before != after:
        raise RuntimeError("Source routing file size or mtime changed during processing")
    manifest = {
        "resolution": config.label,
        "product_set": product_set,
        "routing_file": str(source),
        "products": summaries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nCompleted {len(summaries)} basin products -> {output_dir}")
    return summaries


def run_all_basins(config: BasinRunConfig) -> List[Dict[str, object]]:
    """Generate the four standard products for one routing resolution."""
    return _run_products(config, detailed_parts=False)


def run_all_basin_parts(config: BasinRunConfig) -> List[Dict[str, object]]:
    """Generate the ten approved control-point and incremental products."""
    return _run_products(config, detailed_parts=True)
