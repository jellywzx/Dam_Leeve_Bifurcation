#!/usr/bin/env python3
"""Report active-cell bounds for the current standard basin-mask products."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from basin_mask_config import CONFIG_06MIN, CONFIG_15MIN


PRODUCTS = (
    ("Amazon", "amazon"),
    ("Missouri", "missouri"),
    ("Lower Mississippi", "lower_mississippi"),
    ("Mekong", "mekong"),
)
CONFIGS = {
    "06min": CONFIG_06MIN,
    "15min": CONFIG_15MIN,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resolution",
        choices=("06min", "15min", "all"),
        default="all",
        help="Resolution to inspect (default: all).",
    )
    return parser.parse_args()


def basin_path(output_dir, product_name):
    return Path(output_dir) / product_name / (product_name + "_basin.nc")


def active_cell_bounds(path):
    if not path.is_file():
        raise FileNotFoundError("Basin-mask file not found: {}".format(path))

    with xr.open_dataset(path) as dataset:
        if "basin_mask" not in dataset:
            raise ValueError("Missing 'basin_mask' variable: {}".format(path))
        if "lon" not in dataset.coords or "lat" not in dataset.coords:
            raise ValueError("Missing lon/lat coordinates: {}".format(path))

        mask = np.asarray(dataset["basin_mask"].values) > 0
        longitude = np.asarray(dataset["lon"].values, dtype=float)
        latitude = np.asarray(dataset["lat"].values, dtype=float)

    if mask.ndim != 2 or mask.shape != (len(latitude), len(longitude)):
        raise ValueError("basin_mask shape does not match lat/lon coordinates: {}".format(path))
    if len(longitude) < 2 or len(latitude) < 2:
        raise ValueError("lon/lat coordinates must each contain at least two values: {}".format(path))

    rows, columns = np.where(mask)
    if rows.size == 0:
        raise ValueError("No active basin-mask cells: {}".format(path))

    delta_lon = abs(float(np.median(np.diff(longitude))))
    delta_lat = abs(float(np.median(np.diff(latitude))))
    return (
        float(longitude[columns].min()) - delta_lon / 2.0,
        float(longitude[columns].max()) + delta_lon / 2.0,
        float(latitude[rows].min()) - delta_lat / 2.0,
        float(latitude[rows].max()) + delta_lat / 2.0,
    )


def main():
    args = parse_args()
    resolutions = tuple(CONFIGS) if args.resolution == "all" else (args.resolution,)

    for resolution in resolutions:
        config = CONFIGS[resolution]
        print("\n{} standard basin masks".format(resolution))
        for display_name, product_name in PRODUCTS:
            path = basin_path(config.output_dir, product_name)
            west, east, south, north = active_cell_bounds(path)
            print("\n{}".format(display_name))
            print("Longitude: {:.3f} to {:.3f}".format(west, east))
            print("Latitude:  {:.3f} to {:.3f}".format(south, north))
            print("basin_mask: {}".format(path))


if __name__ == "__main__":
    main()
