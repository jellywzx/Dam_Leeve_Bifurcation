#!/usr/bin/env python3
"""Resolution-specific configuration for the four production basin masks."""

from pathlib import Path
from types import MappingProxyType
from typing import Mapping


class BasinRunConfig:
    """Small immutable-by-convention container, compatible with Python 3.6."""

    __slots__ = (
        "label",
        "routing_file",
        "output_dir",
        "parts_output_dir",
        "nlon",
        "nlat",
        "chunk_units",
        "outlet_ids",
    )

    def __init__(
        self,
        label,
        routing_file,
        output_dir,
        parts_output_dir,
        nlon,
        nlat,
        chunk_units,
        outlet_ids,
    ):
        self.label = label
        self.routing_file = routing_file
        self.output_dir = output_dir
        self.parts_output_dir = parts_output_dir
        self.nlon = nlon
        self.nlat = nlat
        self.chunk_units = chunk_units
        self.outlet_ids = outlet_ids


_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = Path("/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment")

IDS_06MIN = MappingProxyType({
    "amazon_main_mouth": 1549853,
    "missouri_hermann": 1491826,
    "missouri_before_confluence": 1492035,
    "cairo_below_confluence": 1492387,
    "mississippi_vicksburg": 1493097,
    "mississippi_main_mouth": 1537441,
    "mekong_stungtreng": 1491348,
    "mekong_main_mouth": 1547074,
})

IDS_15MIN = MappingProxyType({
    "amazon_main_mouth": 248085,
    "missouri_hermann": 228486,
    "missouri_before_confluence": 228548,
    "cairo_below_confluence": 228714,
    "mississippi_vicksburg": 228947,
    "mississippi_main_mouth": 243364,
    "mekong_stungtreng": 228320,
    "mekong_main_mouth": 247181,
})

CONFIG_06MIN = BasinRunConfig(
    label="06min",
    routing_file=_ROOT / "grid_routing_data_06min.nc",
    output_dir=_SCRIPT_DIR / "outputs/basin_masks_06min",
    parts_output_dir=_SCRIPT_DIR / "outputs/basin_parts_06min",
    nlon=3600,
    nlat=1800,
    chunk_units=18000,
    outlet_ids=IDS_06MIN,
)

CONFIG_15MIN = BasinRunConfig(
    label="15min",
    routing_file=_ROOT / "grid_routing_data_15min_sed.nc",
    output_dir=_SCRIPT_DIR / "outputs/basin_masks_15min",
    parts_output_dir=_SCRIPT_DIR / "outputs/basin_parts_15min",
    nlon=1440,
    nlat=720,
    chunk_units=18000,
    outlet_ids=IDS_15MIN,
)
