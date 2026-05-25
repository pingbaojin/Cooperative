"""Parse navigation pose text: KF-GINS .nav (NED) or AVP *avp.txt (NEU / 东北天)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple

import numpy as np

# Shared 11-column layout (0-based):
# 0 week/reserved, 1 time(s), 2 lat(deg), 3 lon(deg), 4 alt(m),
# 5-7 velocity, 8-10 roll,pitch,yaw(deg)
# KF-GINS: vn,ve,vd NED (m/s); AVP: col5 East, col6 North, col7 Up (m/s)
NAV_COLS = 11
EPOCH_TIME_THRESHOLD = 1e9
ALT_PLACEHOLDER_THRESHOLD = 1.0

NavFormat = Literal["auto", "avp", "kfgins"]
LocalFrame = Literal["neu", "ned"]


@dataclass
class NavSeries:
    """Interpolatable host/target navigation."""

    t_gps: np.ndarray
    week: np.ndarray
    sow: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    alt_m: np.ndarray
    vn: np.ndarray
    ve: np.ndarray
    vd: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    yaw_deg: np.ndarray
    local_frame: LocalFrame = "ned"


def _load_numeric_rows(path: Path) -> np.ndarray:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < NAV_COLS:
                continue
            try:
                vals = [float(x) for x in parts[:NAV_COLS]]
            except ValueError:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"No valid navigation rows (>= {NAV_COLS} columns) in {path}")
    return np.asarray(rows, dtype=np.float64)


def _resolve_nav_format(path: Path, nav_format: NavFormat) -> NavFormat:
    if nav_format != "auto":
        return nav_format
    name = path.name.lower()
    if "avp" in name and name.endswith(".txt"):
        return "avp"
    if name.endswith(".nav"):
        return "kfgins"
    return "avp" if "avp" in name else "kfgins"


def _time_axis(week: np.ndarray, sow: np.ndarray, nav_format: NavFormat, use_full_week_time: bool) -> np.ndarray:
    if nav_format == "avp":
        if np.max(np.abs(week)) > 1.0:
            warnings.warn(
                "AVP file: column 0 expected ~0 (reserved); using column 1 as Unix epoch time.",
                stacklevel=3,
            )
        return sow.copy()
    if use_full_week_time and float(np.median(sow)) > EPOCH_TIME_THRESHOLD:
        return sow.copy()
    if use_full_week_time:
        return week * 604800.0 + sow
    return sow.copy()


def _sort_dedupe_nav_arrays(
    t_gps: np.ndarray, arrays: Tuple[np.ndarray, ...]
) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
    order = np.argsort(t_gps)
    t_sorted = t_gps[order]
    sorted_arrays = tuple(a[order] for a in arrays)
    if len(t_sorted) <= 1:
        return t_sorted, sorted_arrays
    keep = np.concatenate([[True], np.diff(t_sorted) > 0])
    if not np.all(keep):
        warnings.warn("Duplicate navigation timestamps removed after sort.", stacklevel=3)
    return t_sorted[keep], tuple(a[keep] for a in sorted_arrays)


def load_nav(
    path: str | Path,
    nav_format: NavFormat = "auto",
    use_full_week_time: bool = True,
    nav_ref_alt_m: float | None = None,
) -> NavSeries:
    """
    Load KF-GINS .nav or AVP *avp.txt.

    AVP velocity columns: East, North, Up -> stored as ve, vn, vd (vd positive up).
    KF-GINS velocity: NED vn, ve, vd (vd positive down).
    """
    path = Path(path)
    fmt = _resolve_nav_format(path, nav_format)
    local_frame: LocalFrame = "neu" if fmt == "avp" else "ned"

    data = _load_numeric_rows(path)
    week = data[:, 0]
    sow = data[:, 1]
    t_gps = _time_axis(week, sow, fmt, use_full_week_time)

    lat_deg = data[:, 2]
    lon_deg = data[:, 3]
    alt_m = data[:, 4].copy()

    if fmt == "avp":
        ve = data[:, 5]
        vn = data[:, 6]
        vd = data[:, 7]
    else:
        vn = data[:, 5]
        ve = data[:, 6]
        vd = data[:, 7]

    roll_deg = data[:, 8]
    pitch_deg = data[:, 9]
    yaw_deg = data[:, 10]

    if fmt == "avp" and float(np.max(np.abs(alt_m))) < ALT_PLACEHOLDER_THRESHOLD:
        if nav_ref_alt_m is not None:
            alt_m = alt_m + float(nav_ref_alt_m)
        else:
            warnings.warn(
                f"AVP {path.name}: altitude column near zero; horizontal pose uses lat/lon only.",
                stacklevel=2,
            )

    bundle = (week, sow, lat_deg, lon_deg, alt_m, vn, ve, vd, roll_deg, pitch_deg, yaw_deg)
    t_gps, sorted_bundle = _sort_dedupe_nav_arrays(t_gps, bundle)
    (
        week,
        sow,
        lat_deg,
        lon_deg,
        alt_m,
        vn,
        ve,
        vd,
        roll_deg,
        pitch_deg,
        yaw_deg,
    ) = sorted_bundle

    return NavSeries(
        t_gps=t_gps,
        week=week,
        sow=sow,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_m=alt_m,
        vn=vn,
        ve=ve,
        vd=vd,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        local_frame=local_frame,
    )


def interp_series(nav: NavSeries, t_query: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Linear interpolate onto t_query; nav.t_gps must be strictly increasing."""
    t = nav.t_gps
    if np.any(np.diff(t) <= 0):
        raise ValueError("nav.t_gps must be strictly increasing; sort the file or fix duplicates.")
    tq = np.asarray(t_query, dtype=np.float64)
    out = []
    for channel in (
        nav.lat_deg,
        nav.lon_deg,
        nav.alt_m,
        nav.vn,
        nav.ve,
        nav.vd,
        nav.roll_deg,
        nav.pitch_deg,
        nav.yaw_deg,
    ):
        out.append(np.interp(tq, t, channel))
    return tuple(out)


def load_nav_pair_from_cfg(cfg: dict) -> Tuple[NavSeries, NavSeries]:
    """Load host/target nav from YAML keys host_nav_path, target_nav_path."""
    nav_format: NavFormat = cfg.get("nav_format", "auto")
    ref_alt = cfg.get("nav_ref_alt_m")
    nav_ref_alt_m = float(ref_alt) if ref_alt is not None else None
    nav_h = load_nav(
        cfg["host_nav_path"],
        nav_format=nav_format,
        nav_ref_alt_m=nav_ref_alt_m,
    )
    nav_t = load_nav(
        cfg["target_nav_path"],
        nav_format=nav_format,
        nav_ref_alt_m=nav_ref_alt_m,
    )
    if nav_h.local_frame != nav_t.local_frame:
        raise ValueError(
            f"host local_frame={nav_h.local_frame} != target {nav_t.local_frame}"
        )
    return nav_h, nav_t


def warn_if_bag_starts_before_nav(
    ros_stamps_sec: np.ndarray, time_offset_sec: float, t_nav_min: float
) -> None:
    """Warn when first lidar (after offset) is before navigation coverage."""
    if ros_stamps_sec.size == 0:
        return
    first_nav_t = float(ros_stamps_sec[0]) + float(time_offset_sec)
    if first_nav_t < float(t_nav_min) - 1e-3:
        gap = float(t_nav_min) - first_nav_t
        warnings.warn(
            f"First lidar nav_time {first_nav_t:.3f} is {gap:.1f}s before nav t_min {t_nav_min:.3f}; "
            "poses will be clamped to nav start (increase skip_start_sec or check time_offset_sec).",
            stacklevel=2,
        )


def unwrap_deg_interp(nav: NavSeries, t_query: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Interp lat/lon/alt/vel linearly; unwrap roll/pitch/yaw (deg) then interp."""
    t = nav.t_gps
    tq = np.asarray(t_query, dtype=np.float64)
    lat, lon, alt, vn, ve, vd, _, _, _ = interp_series(nav, t_query)

    def unwrap_channel(ch: np.ndarray) -> np.ndarray:
        rad = np.deg2rad(ch)
        rad_u = np.unwrap(rad)
        return np.rad2deg(np.interp(tq, t, rad_u))

    r = unwrap_channel(nav.roll_deg)
    p = unwrap_channel(nav.pitch_deg)
    y = unwrap_channel(nav.yaw_deg)
    return lat, lon, alt, vn, ve, vd, r, p, y


def unwrap_yaw_interp(
    nav: NavSeries, t_query: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interp lat/lon linearly; unwrap yaw (deg) then interp. For 2D pose pipeline."""
    t = nav.t_gps
    tq = np.asarray(t_query, dtype=np.float64)
    lat = np.interp(tq, t, nav.lat_deg)
    lon = np.interp(tq, t, nav.lon_deg)
    rad = np.deg2rad(nav.yaw_deg)
    yaw_deg = np.rad2deg(np.interp(tq, t, np.unwrap(rad)))
    return lat, lon, yaw_deg


def pose_dict_2d(nav: NavSeries, t_query: float | np.ndarray) -> Dict[str, float]:
    """Single-time 2D pose dict: lat_deg, lon_deg, yaw_deg (roll/pitch not used)."""
    tq = np.asarray([float(t_query)], dtype=np.float64)
    lat, lon, yaw = unwrap_yaw_interp(nav, tq)
    return {
        "lat_deg": float(lat[0]),
        "lon_deg": float(lon[0]),
        "yaw_deg": float(yaw[0]),
    }
