"""Host ego-motion compensation: warp previous lidar cloud into current frame."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from cooperative_link.dual_nav_relative import R_body_to_en
from cooperative_link.geodesy_local_enu import lla_to_en


def _mat2(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float64)[:2, :2]


def _host_en_xy(host: Dict[str, float], lat0_deg: float, lon0_deg: float) -> np.ndarray:
    p = lla_to_en(
        np.array([host["lat_deg"]]),
        np.array([host["lon_deg"]]),
        lat0_deg,
        lon0_deg,
    )[0]
    return p.astype(np.float64)


def lidar_xy_to_en(
    xy: np.ndarray,
    host: Dict[str, float],
    lat0_deg: float,
    lon0_deg: float,
    R_imu_to_vehicle: np.ndarray,
    R_lidar_vehicle: np.ndarray,
    t_lidar_in_vehicle: np.ndarray,
) -> np.ndarray:
    """(N,2) lidar xy -> (N,2) horizontal EN."""
    R_imu_2d = _mat2(R_imu_to_vehicle)
    R_lidar_2d = _mat2(R_lidar_vehicle)
    t_lidar = np.asarray(t_lidar_in_vehicle, dtype=np.float64).reshape(3)
    R_host = R_body_to_en(host["yaw_deg"])
    p_host = _host_en_xy(host, lat0_deg, lon0_deg)

    p_vehicle = (xy - t_lidar[:2]) @ R_lidar_2d.T
    p_body = p_vehicle @ R_imu_2d.T
    return p_host + p_body @ R_host.T


def en_xy_to_lidar(
    en_xy: np.ndarray,
    host: Dict[str, float],
    lat0_deg: float,
    lon0_deg: float,
    R_imu_to_vehicle: np.ndarray,
    R_lidar_vehicle: np.ndarray,
    t_lidar_in_vehicle: np.ndarray,
) -> np.ndarray:
    """(N,2) EN -> (N,2) lidar xy at host pose."""
    R_imu_2d = _mat2(R_imu_to_vehicle)
    R_lidar_2d = _mat2(R_lidar_vehicle)
    t_lidar = np.asarray(t_lidar_in_vehicle, dtype=np.float64).reshape(3)
    R_host = R_body_to_en(host["yaw_deg"])
    p_host = _host_en_xy(host, lat0_deg, lon0_deg)

    p_body = (en_xy - p_host) @ R_host
    p_vehicle = p_body @ R_imu_2d
    return p_vehicle @ R_lidar_2d.T + t_lidar[:2]


def compensate_points_to_current(
    prev_pts: np.ndarray,
    host_prev: Dict[str, float],
    host_curr: Dict[str, float],
    lat0_deg: float,
    lon0_deg: float,
    R_imu_to_vehicle: np.ndarray,
    R_lidar_vehicle: np.ndarray,
    t_lidar_in_vehicle: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Static-scene warp: previous lidar (N,3+) -> current lidar frame.

    Returns (comp_pts, motion_residual_xy) where motion_residual_xy is per-point
    NN distance to compensated prev cloud in current frame (zeros if no prev).
    """
    n = prev_pts.shape[0]
    if n == 0:
        return prev_pts.copy(), np.zeros(0, dtype=np.float64)

    xy_prev = prev_pts[:, :2].astype(np.float64)
    en = lidar_xy_to_en(
        xy_prev, host_prev, lat0_deg, lon0_deg,
        R_imu_to_vehicle, R_lidar_vehicle, t_lidar_in_vehicle,
    )
    xy_comp = en_xy_to_lidar(
        en, host_curr, lat0_deg, lon0_deg,
        R_imu_to_vehicle, R_lidar_vehicle, t_lidar_in_vehicle,
    )
    comp = prev_pts.copy()
    comp[:, 0] = xy_comp[:, 0]
    comp[:, 1] = xy_comp[:, 1]

    return comp, np.zeros(n, dtype=np.float64)


def motion_residual_nn(
    curr_pts: np.ndarray,
    prev_comp: np.ndarray,
    max_dist_m: float,
) -> np.ndarray:
    """Per current point: distance to nearest compensated previous point (xy)."""
    n = curr_pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if prev_comp.shape[0] == 0:
        return np.full(n, max_dist_m, dtype=np.float64)

    xy_c = curr_pts[:, :2].astype(np.float64)
    xy_p = prev_comp[:, :2].astype(np.float64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy_p)
        dist, _ = tree.query(xy_c, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float64)
    except Exception:
        # O(N*M) fallback for small clouds
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            d = np.linalg.norm(xy_p - xy_c[i], axis=1)
            out[i] = float(d.min()) if d.size else max_dist_m
        return out
