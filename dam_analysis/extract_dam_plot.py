#!/usr/bin/env python3
"""
Extract dam/reservoir records from a CoLM grid_routing_data_*.nc file and make plots.

No command-line arguments are required. Edit only the USER CONFIGURATION section
below, then run:

    python extract_plot_dams.py

The script can:
  1. export all dam records;
  2. draw the global dam map and the four project focus-region maps;
  3. screen a single built-in region;
  4. screen a custom lon/lat box;
  5. select dams upstream of an outlet using seq -> seq_next routing topology.

Important:
* The built-in region boxes are quick geographic screening windows, not exact
  basin boundaries. For paper statistics, prefer topology mode or a basin polygon.
* CDO may interpret the unlimited dam-record dimension as a time dimension.
  This script reads the NetCDF/HDF5 structure directly and does not use CDO
  time semantics.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except Exception:
    HAVE_CARTOPY = False


# ============================================================================
# USER CONFIGURATION -- normally you only need to edit this section
# ============================================================================

# Input routing file. A relative path is interpreted from the directory where
# you run this script. If the script and NetCDF file are in the same directory,
# the default below is enough.
NCFILE = Path("/share/home/dq013/zhwei/colm/data/CoLMruntime/unitcatchment/grid_routing_data_06min.nc")

# Output directory.
OUTDIR = Path("/share/home/dq134/wzx/openbench/postprocess/dam_analysis/dam_data")

# Analysis mode:
#   "all"      : global + Missouri + Lower Mississippi + Amazon + Mekong
#   "region"   : one built-in region selected by REGION_NAME
#   "bbox"     : one custom lon/lat box defined by CUSTOM_BBOX
#   "topology" : exact upstream-of-outlet screening using seq -> seq_next
ANALYSIS_MODE = "all"

# Used only when ANALYSIS_MODE = "region".
# Choices: "global", "missouri", "lower_mississippi", "amazon", "mekong"
REGION_NAME = "missouri"

# Used only when ANALYSIS_MODE = "bbox".
# Format: (lon_min, lon_max, lat_min, lat_max)
CUSTOM_BBOX = (-116.0, -89.0, 36.0, 50.5)
CUSTOM_NAME = "Missouri_custom"

# Used only when ANALYSIS_MODE = "topology".
# Preferred: provide OUTLET_SEQ directly if you know it.
# Otherwise leave OUTLET_SEQ = None and provide OUTLET_LON / OUTLET_LAT.
# The script will find the nearest routing cell and then retain only dams whose
# downstream seq_next chain reaches that outlet.
OUTLET_SEQ = None
OUTLET_LON = None
OUTLET_LAT = None
TOPOLOGY_NAME = "Missouri_Hermann"

# Example for topology mode (replace with your validated Hermann coordinates):
# ANALYSIS_MODE = "topology"
# OUTLET_SEQ = None
# OUTLET_LON = -91.0
# OUTLET_LAT = 38.7
# TOPOLOGY_NAME = "Missouri_Hermann"

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


DAM_VARS = {
    "GRAND_ID": "dam_GRAND_ID",
    "DamName": "dam_DamName",
    "lat": "dam_DamLat",
    "lon": "dam_DamLon",
    "area_CaMa": "dam_area_CaMa",
    "DamIX": "dam_DamIX",
    "DamIY": "dam_DamIY",
    "FldVol_mcm": "dam_FldVol_mcm",
    "ConVol_mcm": "dam_ConVol_mcm",
    "TotalVol_mcm": "dam_TotalVol_mcm",
    "Qn_m3s": "dam_Qn",
    "Qf_m3s": "dam_Qf",
    "year": "dam_year",
    "seq": "dam_seq",
}

# Quick screening boxes for the four focus regions in the project design.
# Format: lon_min, lon_max, lat_min, lat_max
REGIONS = {
    "global": (-180.0, 180.0, -90.0, 90.0),
    "missouri": (-116.0, -89.0, 36.0, 50.5),
    "lower_mississippi": (-94.5, -88.0, 28.0, 37.0),
    "amazon": (-82.0, -44.0, -22.0, 7.0),
    "mekong": (93.0, 110.0, 8.0, 34.0),
}

MISSOURI_MAINSTEM = [
    "Fort Peck",
    "Garrison",
    "Oahe",
    "Big Bend",
    "Fort Randall",
    "Gavins Point",
]


def _clean_string(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")
    s = str(x).replace("\x00", "").strip()
    return s


def _apply_scale_and_fill(ds, arr):
    """Apply common NetCDF numeric metadata when reading through h5py."""
    a = np.asarray(arr)
    if a.dtype.kind not in "iuf":
        return a
    a = a.astype(float, copy=False)
    fill = ds.attrs.get("_FillValue", None)
    if fill is not None:
        try:
            fv = float(np.asarray(fill).reshape(-1)[0])
            a = np.where(a == fv, np.nan, a)
        except Exception:
            pass
    miss = ds.attrs.get("missing_value", None)
    if miss is not None:
        try:
            mv = float(np.asarray(miss).reshape(-1)[0])
            a = np.where(a == mv, np.nan, a)
        except Exception:
            pass
    if "scale_factor" in ds.attrs:
        a = a * float(np.asarray(ds.attrs["scale_factor"]).reshape(-1)[0])
    if "add_offset" in ds.attrs:
        a = a + float(np.asarray(ds.attrs["add_offset"]).reshape(-1)[0])
    return a


def read_1d(h5: h5py.File, varname: str):
    """Read one dam variable and return a 1-D numpy array."""
    if varname not in h5:
        return None
    ds = h5[varname]
    arr = ds[...]

    # NetCDF NC_STRING / HDF5 variable-length strings.
    if arr.dtype.kind in {"O", "U"}:
        return np.asarray([_clean_string(x) for x in arr.reshape(-1)], dtype=object)

    # Fixed-length byte string or char array. If the final axis is a char
    # length axis, join each row into one string.
    if arr.dtype.kind == "S":
        if arr.ndim >= 2 and arr.dtype.itemsize == 1:
            rows = arr.reshape((-1, arr.shape[-1]))
            vals = []
            for row in rows:
                vals.append(_clean_string(b"".join(row.tolist())))
            return np.asarray(vals, dtype=object)
        return np.asarray([_clean_string(x) for x in arr.reshape(-1)], dtype=object)

    arr = _apply_scale_and_fill(ds, arr)
    return np.asarray(arr).reshape(-1)


def extract_dams(nc_path: Path) -> pd.DataFrame:
    with h5py.File(nc_path, "r") as h5:
        print("\n[Top-level HDF5/NetCDF objects]")
        # NetCDF4 dimensions are represented internally as HDF5 dimension
        # scales. For this task the variable shapes are what matter most.
        for name in list(h5.keys())[:30]:
            obj = h5[name]
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:24s} shape={obj.shape} dtype={obj.dtype}")
        if len(h5.keys()) > 30:
            print("  ...")

        print("\n[Dam variable shapes]")
        lengths = []
        raw = {}
        for col, var in DAM_VARS.items():
            if var in h5:
                vv = h5[var]
                print(f"  {var:20s} shape={vv.shape} dtype={vv.dtype}")
                a = read_1d(h5, var)
                raw[col] = a
                if a is not None:
                    lengths.append(len(a))
            else:
                print(f"  {var:20s} MISSING")
                raw[col] = None

        if not lengths:
            raise RuntimeError("No dam_* variables were found.")

        # Most dam variables should share one record count (likely 5870 here).
        counts = pd.Series(lengths).value_counts()
        n = int(counts.index[0])
        print(f"\nDetected dam record count: {n}")

        data = {}
        for col, a in raw.items():
            if a is None:
                data[col] = np.full(n, np.nan if col != "DamName" else "", dtype=object)
                continue
            if len(a) == n:
                data[col] = a
            elif len(a) == 1:
                data[col] = np.repeat(a, n)
            else:
                print(f"WARNING: {DAM_VARS[col]} length={len(a)} != {n}; filling missing.")
                data[col] = np.full(n, np.nan if col != "DamName" else "", dtype=object)

    df = pd.DataFrame(data)

    numeric_cols = [c for c in df.columns if c != "DamName"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["GRAND_ID", "DamIX", "DamIY", "year", "seq"]:
        if c in df:
            df[c] = df[c].round().astype("Int64")

    df["DamName"] = df["DamName"].fillna("").map(_clean_string)

    valid_geo = (
        df["lon"].between(-180, 180, inclusive="both")
        & df["lat"].between(-90, 90, inclusive="both")
    )
    n_bad = int((~valid_geo).sum())
    if n_bad:
        print(f"WARNING: dropping {n_bad} records with invalid/missing lon/lat.")
    df = df.loc[valid_geo].copy()

    df["year_valid"] = df["year"].between(1800, 2030, inclusive="both").fillna(False)

    before = len(df)
    df = df.drop_duplicates(subset=["GRAND_ID", "seq"], keep="first")
    if len(df) != before:
        print(f"Dropped {before-len(df)} exact GRAND_ID+seq duplicates.")

    return df.reset_index(drop=True)

def subset_bbox(df: pd.DataFrame, bbox):
    w, e, s, n = bbox
    return df[df["lon"].between(w, e) & df["lat"].between(s, n)].copy()


def haversine_km(lon, lat, lon0, lat0):
    lon = np.deg2rad(np.asarray(lon, dtype=float))
    lat = np.deg2rad(np.asarray(lat, dtype=float))
    lon0 = math.radians(lon0)
    lat0 = math.radians(lat0)
    dlon = (lon - lon0 + np.pi) % (2 * np.pi) - np.pi
    dlat = lat - lat0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * math.cos(lat0) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _as_numeric_1d(h5: h5py.File, name: str):
    if name not in h5:
        raise RuntimeError(f"Missing variable {name}")
    return np.asarray(_apply_scale_and_fill(h5[name], h5[name][...])).reshape(-1)


def find_nearest_outlet_seq(nc_path: Path, lon0: float, lat0: float):
    with h5py.File(nc_path, "r") as nc:
        for name in ("seq", "seq_lon", "seq_lat"):
            if name not in nc:
                raise RuntimeError(f"Topology mode requires variable {name}")
        seq = _as_numeric_1d(nc, "seq")
        lon = _as_numeric_1d(nc, "seq_lon")
        lat = _as_numeric_1d(nc, "seq_lat")

    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(seq)
    idx_good = np.flatnonzero(good)
    d = haversine_km(lon[good], lat[good], lon0, lat0)
    j = int(np.argmin(d))
    i = int(idx_good[j])
    return int(seq[i]), float(lon[i]), float(lat[i]), float(d[j])


def dams_upstream_of_outlet(nc_path: Path, dams: pd.DataFrame, outlet_seq: int) -> pd.DataFrame:
    """
    Determine dam membership by following each dam's seq_next downstream until
    outlet_seq is reached. This avoids loading a full basin polygon.
    """
    with h5py.File(nc_path, "r") as nc:
        for name in ("seq", "seq_next"):
            if name not in nc:
                raise RuntimeError(f"Topology mode requires variable {name}")
        seq = _as_numeric_1d(nc, "seq").astype(np.int64)
        seq_next = _as_numeric_1d(nc, "seq_next").astype(np.int64)

    # Binary-search lookup from arbitrary seq IDs to array indices.
    order = np.argsort(seq)
    seq_sorted = seq[order]

    def next_of(s: int):
        j = int(np.searchsorted(seq_sorted, s))
        if j >= len(seq_sorted) or seq_sorted[j] != s:
            return None
        return int(seq_next[order[j]])

    memo = {int(outlet_seq): True}

    def reaches_outlet(start):
        if pd.isna(start):
            return False
        cur = int(start)
        path = []
        seen = set()
        for _ in range(200000):
            if cur in memo:
                ans = memo[cur]
                for p in path:
                    memo[p] = ans
                return ans
            if cur == outlet_seq:
                for p in path:
                    memo[p] = True
                return True
            if cur <= 0 or cur in seen:
                for p in path:
                    memo[p] = False
                return False
            seen.add(cur)
            path.append(cur)
            nxt = next_of(cur)
            if nxt is None:
                for p in path:
                    memo[p] = False
                return False
            cur = nxt
        raise RuntimeError(f"Exceeded downstream trace limit from seq={start}")

    mask = dams["seq"].map(reaches_outlet).to_numpy(dtype=bool)
    return dams.loc[mask].copy()


def name_match(s: str, target: str) -> bool:
    s0 = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    t0 = re.sub(r"[^a-z0-9]+", " ", target.lower()).strip()
    return t0 in s0


def print_summary(df: pd.DataFrame, name: str):
    valid_year = df[df["year_valid"]].copy()
    built_pre = int((valid_year["year"] <= 2000).sum())
    built_eval = int(valid_year["year"].between(2001, 2005).sum())
    built_post = int((valid_year["year"] > 2005).sum())
    total_vol = pd.to_numeric(df["TotalVol_mcm"], errors="coerce").clip(lower=0).sum(min_count=1)

    print(f"\n===== {name} =====")
    print(f"Dam records: {len(df)}")
    if len(valid_year):
        print(f"Construction year range: {int(valid_year['year'].min())} - {int(valid_year['year'].max())}")
        print(f"Built by 2000: {built_pre}")
        print(f"Built during 2001-2005: {built_eval}")
        print(f"Built after 2005: {built_post}")
    else:
        print("No valid construction years (1800-2030).")
    if pd.notna(total_vol):
        print(f"Sum of dam_TotalVol_mcm: {total_vol:,.1f} million m3")

    hits = []
    for target in MISSOURI_MAINSTEM:
        m = df["DamName"].map(lambda s: name_match(s, target))
        if m.any():
            hits.append(df.loc[m, ["DamName", "year", "TotalVol_mcm", "lon", "lat", "GRAND_ID", "seq"]])
    if hits:
        print("\nMissouri mainstem-name matches:")
        print(pd.concat(hits).drop_duplicates().to_string(index=False))


def marker_sizes(vol):
    v = pd.to_numeric(vol, errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    if len(v) == 0:
        return np.array([])
    z = np.log1p(v)
    zmax = np.nanmax(z) if np.isfinite(z).any() else 0.0
    if zmax <= 0:
        return np.full(len(v), 25.0)
    return 15.0 + 115.0 * z / zmax


def plot_map(df: pd.DataFrame, name: str, outpng: Path, bbox=None):
    if df.empty:
        print(f"Skip map for {name}: no dams.")
        return

    sizes = marker_sizes(df["TotalVol_mcm"])
    valid = df["year_valid"].to_numpy(bool)

    if HAVE_CARTOPY:
        fig = plt.figure(figsize=(10, 7))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.95")
        ax.add_feature(cfeature.OCEAN, facecolor="white")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="0.7", linewidth=0.4)
        ax.add_feature(cfeature.RIVERS, edgecolor="0.65", linewidth=0.35)
        if bbox is not None:
            ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
        else:
            pad_lon = max(1.0, 0.05 * (df["lon"].max() - df["lon"].min() + 1e-6))
            pad_lat = max(1.0, 0.05 * (df["lat"].max() - df["lat"].min() + 1e-6))
            ax.set_extent([
                df["lon"].min()-pad_lon, df["lon"].max()+pad_lon,
                df["lat"].min()-pad_lat, df["lat"].max()+pad_lat
            ], crs=ccrs.PlateCarree())
        gridlines = ax.gridlines(
            crs=ccrs.PlateCarree(), draw_labels=True,
            linewidth=0.5, color="gray", alpha=0.5, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        transform = None
        if bbox is not None:
            ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.25)

    # Unknown/invalid construction year first.
    if (~valid).any():
        kwargs = dict(s=sizes[~valid], marker="o", edgecolors="k", linewidths=0.25,
                      alpha=0.65, label="year missing/invalid")
        if transform is not None:
            kwargs["transform"] = transform
        ax.scatter(df.loc[~valid, "lon"], df.loc[~valid, "lat"], **kwargs)

    sc = None
    if valid.any():
        kwargs = dict(c=df.loc[valid, "year"].astype(float), s=sizes[valid], marker="o",
                      cmap="viridis", edgecolors="k", linewidths=0.25, alpha=0.82)
        if transform is not None:
            kwargs["transform"] = transform
        sc = ax.scatter(df.loc[valid, "lon"], df.loc[valid, "lat"], **kwargs)

    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.80, pad=0.03)
        cb.set_label("Construction year")

    # Label the six Missouri mainstem dams when present.
    for target in MISSOURI_MAINSTEM:
        m = df["DamName"].map(lambda s: name_match(s, target))
        for _, r in df.loc[m].iterrows():
            ann = dict(xy=(r["lon"], r["lat"]), xytext=(4, 4), textcoords="offset points",
                       fontsize=8)
            if transform is not None:
                ann["transform"] = transform
            ax.annotate(target, **ann)

    ax.set_title(f"Dams in {name}\nmarker size ~ log(total storage), color = construction year")
    fig.tight_layout()
    fig.savefig(outpng, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpng}")


def plot_construction_history(df: pd.DataFrame, name: str, outpng: Path):
    d = df[df["year_valid"]].copy()
    if d.empty:
        print(f"Skip construction history for {name}: no valid years.")
        return

    d["year_i"] = d["year"].astype(int)
    first = int(math.floor(d["year_i"].min() / 10) * 10)
    last = int(math.ceil((d["year_i"].max() + 1) / 10) * 10)
    bins = np.arange(first, last + 10, 10)
    labels = bins[:-1]
    decade = pd.cut(d["year_i"], bins=bins, right=False, labels=labels)
    count = decade.value_counts(sort=False)
    vol = d.assign(decade=decade).groupby("decade", observed=False)["TotalVol_mcm"].sum(min_count=1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(count))
    ax.bar(x, count.to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) if pd.notna(v) else "" for v in count.index], rotation=45)
    ax.set_xlabel("Construction decade")
    ax.set_ylabel("Number of dams")
    ax.set_title(f"Dam construction history: {name}")
    ax.grid(axis="y", alpha=0.25)

    # Add storage per decade as text above each bar; avoids a second axis.
    for xi, n, vv in zip(x, count.to_numpy(), vol.to_numpy()):
        if pd.notna(vv) and n > 0:
            ax.text(xi, n, f"{vv:.0f} MCM", ha="center", va="bottom", fontsize=7, rotation=90)

    fig.tight_layout()
    fig.savefig(outpng, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outpng}")


def safe_slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())
    return s.strip("_") or "region"


def save_region(df: pd.DataFrame, name: str, outdir: Path, bbox=None):
    slug = safe_slug(name)
    csv_path = outdir / f"dams_{slug}.csv"
    map_path = outdir / f"dams_{slug}_map.png"
    hist_path = outdir / f"dams_{slug}_construction.png"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {csv_path}")
    print_summary(df, name)
    plot_map(df, name, map_path, bbox=bbox)
    plot_construction_history(df, name, hist_path)


def main():
    ncfile = NCFILE.expanduser()
    outdir = OUTDIR.expanduser()
    mode = str(ANALYSIS_MODE).strip().lower()

    if not ncfile.exists():
        raise FileNotFoundError(
            f"Input file not found: {ncfile}\n"
            "Please edit NCFILE in the USER CONFIGURATION section at the top of the script."
        )

    if mode not in {"all", "region", "bbox", "topology"}:
        raise ValueError(
            f"Unsupported ANALYSIS_MODE={ANALYSIS_MODE!r}. "
            "Use one of: all, region, bbox, topology."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CoLM dam/reservoir extraction")
    print(f"Input file    : {ncfile}")
    print(f"Output folder : {outdir}")
    print(f"Analysis mode : {mode}")
    print("=" * 72)

    dams = extract_dams(ncfile)

    all_csv = outdir / "dams_all_records.csv"
    dams.to_csv(all_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {all_csv}")

    if mode == "topology":
        if OUTLET_SEQ is not None:
            outlet_seq = int(OUTLET_SEQ)
            print(f"\nUsing configured outlet seq: {outlet_seq}")
        else:
            if OUTLET_LON is None or OUTLET_LAT is None:
                raise ValueError(
                    "ANALYSIS_MODE='topology' requires either OUTLET_SEQ, "
                    "or both OUTLET_LON and OUTLET_LAT in the USER CONFIGURATION section."
                )
            outlet_seq, rlon, rlat, dist = find_nearest_outlet_seq(
                ncfile, float(OUTLET_LON), float(OUTLET_LAT)
            )
            print(
                f"\nNearest routing cell to outlet ({OUTLET_LON}, {OUTLET_LAT}): "
                f"seq={outlet_seq}, route_lon={rlon:.5f}, "
                f"route_lat={rlat:.5f}, distance={dist:.2f} km"
            )

        sub = dams_upstream_of_outlet(ncfile, dams, outlet_seq)
        name = TOPOLOGY_NAME.strip() or f"upstream_of_seq_{outlet_seq}"
        save_region(sub, name, outdir, bbox=None)
        return

    if mode == "bbox":
        if CUSTOM_BBOX is None or len(CUSTOM_BBOX) != 4:
            raise ValueError("CUSTOM_BBOX must contain four values: W, E, S, N.")
        bbox = tuple(float(x) for x in CUSTOM_BBOX)
        name = CUSTOM_NAME.strip() or "custom_bbox"
        save_region(subset_bbox(dams, bbox), name, outdir, bbox=bbox)
        return

    if mode == "region":
        region = str(REGION_NAME).strip().lower()
        if region not in REGIONS:
            raise ValueError(
                f"Unknown REGION_NAME={REGION_NAME!r}. Choices: {', '.join(REGIONS)}"
            )
        bbox = REGIONS[region]
        save_region(subset_bbox(dams, bbox), region, outdir, bbox=bbox)
        return

    # mode == "all": global summary plus the four project focus-region views.
    save_region(dams, "global", outdir, bbox=REGIONS["global"])
    for name in ("missouri", "lower_mississippi", "amazon", "mekong"):
        bbox = REGIONS[name]
        save_region(subset_bbox(dams, bbox), name, outdir, bbox=bbox)


if __name__ == "__main__":
    main()
