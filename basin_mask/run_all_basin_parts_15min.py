#!/usr/bin/env python3
"""Generate all approved control-point and incremental masks at 15 arc minutes."""

import sys

from basin_mask_config import CONFIG_15MIN
from basin_mask_core import run_all_basin_parts


if __name__ == "__main__":
    try:
        run_all_basin_parts(CONFIG_15MIN)
    except (ValueError, FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        sys.exit(f"ERROR: {exc}")
