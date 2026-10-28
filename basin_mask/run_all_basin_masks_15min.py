#!/usr/bin/env python3
"""Generate the four standard basin masks from the 15-arc-minute network."""

import sys

from basin_mask_config import CONFIG_15MIN
from basin_mask_core import run_all_basins


if __name__ == "__main__":
    try:
        run_all_basins(CONFIG_15MIN)
    except (ValueError, FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        sys.exit(f"ERROR: {exc}")
