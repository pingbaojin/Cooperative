"""Local flat Earth ENU (East, North, Up) from WGS84 LLA."""

from __future__ import annotations

import numpy as np

WGS84_A = 6378137.0  # semi-major axis (m)


def lla_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> np.ndarray:
    """
    Flat-Earth ENU offsets from reference (lat0, lon0, alt0).

    Returns (N, 3) with columns [East, North, Up] in meters.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)
    dlat = lat - lat0
    dlon = lon - lon0
    rm = WGS84_A + alt0_m
    east = rm * np.cos(lat0) * dlon
    north = rm * dlat
    up = alt_m - alt0_m
    return np.stack([east, north, up], axis=-1)


def lla_to_en(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float = 0.0,
    alt_m: np.ndarray | None = None,
) -> np.ndarray:
    """
    Horizontal ENU slice [East, North] in meters (alt ignored for horizontal position).

    Returns (N, 2).
    """
    if alt_m is None:
        alt_m = np.zeros_like(lat_deg, dtype=np.float64)
    enu = lla_to_enu(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m)
    return enu[:, :2].astype(np.float64)


def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    """Map [E,N,U] -> [N,E,D] NED vector (same length, shape (...,3))."""
    e, n, u = v_enu[..., 0], v_enu[..., 1], v_enu[..., 2]
    return np.stack([n, e, -u], axis=-1)
