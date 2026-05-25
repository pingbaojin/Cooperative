"""Dual NAV -> 2D relative geometry in host vehicle / lidar frames (yaw-only, planar)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from cooperative_link.geodesy_local_enu import lla_to_en


def R_body_to_en(yaw_deg: float) -> np.ndarray:
    """
    Body FLU (x forward, y left) -> local horizontal EN [East, North].

    Yaw: 0 deg = North, positive toward East (北偏东为正).
    """
    y = np.deg2rad(float(yaw_deg))
    sy, cy = np.sin(y), np.cos(y)
    return np.array([[sy, -cy], [cy, sy]], dtype=np.float64)


def _mat2(R: np.ndarray) -> np.ndarray:
    return np.asarray(R, dtype=np.float64)[:2, :2]


def compute_frame(
    host: Dict[str, float],
    target: Dict[str, float],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
    R_imu_to_vehicle: np.ndarray,
    R_lidar_vehicle: np.ndarray,
    t_lidar_in_vehicle: np.ndarray,
    target_imu_to_center_offset: np.ndarray,
    lidar_yaw_bias: float,
    gt_center_z_m: float = 0.0,
) -> Dict[str, Any]:
    """
    Planar relative pose: host/target use lat_deg, lon_deg, yaw_deg only (roll/pitch ignored).

    Returns center_lidar (3,) with z=gt_center_z_m, yaw_lidar (rad), delta_enu (3,) for diagnostics.
    """
    del alt0_m  # horizontal origin uses lat0/lon0 only

    p_host_en = lla_to_en(
        np.array([host["lat_deg"]]),
        np.array([host["lon_deg"]]),
        lat0_deg,
        lon0_deg,
    )[0]
    p_tgt_en = lla_to_en(
        np.array([target["lat_deg"]]),
        np.array([target["lon_deg"]]),
        lat0_deg,
        lon0_deg,
    )[0]

    offset = np.asarray(target_imu_to_center_offset, dtype=np.float64).reshape(3)
    R_tgt = R_body_to_en(target["yaw_deg"])
    p_center_tgt_en = p_tgt_en + R_tgt @ offset[:2]

    delta_en = p_center_tgt_en - p_host_en

    R_host = R_body_to_en(host["yaw_deg"])
    delta_body = R_host.T @ delta_en

    R_imu_2d = _mat2(R_imu_to_vehicle)
    R_lidar_2d = _mat2(R_lidar_vehicle)
    t_lidar = np.asarray(t_lidar_in_vehicle, dtype=np.float64).reshape(3)

    center_vehicle_xy = R_imu_2d @ delta_body
    center_lidar_xy = R_lidar_2d @ center_vehicle_xy + t_lidar[:2]
    center_lidar = np.array(
        [center_lidar_xy[0], center_lidar_xy[1], float(gt_center_z_m)],
        dtype=np.float64,
    )

    fwd_body = np.array([1.0, 0.0], dtype=np.float64)
    fwd_en = R_tgt @ fwd_body
    fwd_host_body = R_host.T @ fwd_en
    fwd_vehicle = R_imu_2d @ fwd_host_body
    fwd_lidar = R_lidar_2d @ fwd_vehicle
    yaw_lidar = float(np.arctan2(fwd_lidar[1], fwd_lidar[0]) + lidar_yaw_bias)

    delta_enu = np.array([delta_en[0], delta_en[1], 0.0], dtype=np.float64)

    return {
        "center_lidar": center_lidar,
        "yaw_lidar": yaw_lidar,
        "center_vehicle": np.array(
            [center_vehicle_xy[0], center_vehicle_xy[1], 0.0], dtype=np.float64
        ),
        "delta_enu": delta_enu,
    }
