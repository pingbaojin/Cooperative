"""Per-frame instantaneous range/bearing prior (no target track filtering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from cooperative_link.dual_nav_relative import compute_frame


@dataclass
class InstantPrior:
    r_prior_m: float
    bearing_prior_rad: float
    center_lidar: np.ndarray
    yaw_lidar: float
    center_vehicle: np.ndarray
    delta_enu: np.ndarray


def compute_instant_prior(
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
) -> InstantPrior:
    """
    Current-frame only: r and bearing from dual NAV via compute_frame.

    No temporal smoothing or prediction on target state.
    """
    fr = compute_frame(
        host,
        target,
        lat0_deg,
        lon0_deg,
        alt0_m,
        R_imu_to_vehicle,
        R_lidar_vehicle,
        t_lidar_in_vehicle,
        target_imu_to_center_offset,
        lidar_yaw_bias,
        gt_center_z_m=gt_center_z_m,
    )
    center = np.asarray(fr["center_lidar"], dtype=np.float64)
    x, y = float(center[0]), float(center[1])
    r_prior = float(np.hypot(x, y))
    bearing = float(np.arctan2(y, x))
    return InstantPrior(
        r_prior_m=r_prior,
        bearing_prior_rad=bearing,
        center_lidar=center,
        yaw_lidar=float(fr["yaw_lidar"]),
        center_vehicle=np.asarray(fr["center_vehicle"], dtype=np.float64),
        delta_enu=np.asarray(fr["delta_enu"], dtype=np.float64),
    )
