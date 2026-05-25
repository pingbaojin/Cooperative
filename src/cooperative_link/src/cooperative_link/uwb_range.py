"""UWB-style range from dual-NAV ENU geometry (nav_sim)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cooperative_link.partner_config import PartnerSpec
from cooperative_link.prior import InstantPrior, compute_instant_prior
from cooperative_link.uwb_config import UwbConfig


@dataclass
class UwbRangeReading:
    from_id: int
    to_id: int
    range_m: float
    range_std: float
    valid: bool
    source: str


def enu_range_from_prior(prior: InstantPrior) -> float:
    """Horizontal ENU distance host nav origin -> target body center."""
    d = prior.delta_enu[:2]
    return float(np.hypot(d[0], d[1]))


def enu_range_host_to_partner(
    host: Dict[str, float],
    partner: Dict[str, float],
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
    R_imu_to_vehicle: np.ndarray,
    R_lidar_vehicle: np.ndarray,
    t_lidar_in_vehicle: np.ndarray,
    target_imu_to_center_offset: np.ndarray,
    lidar_yaw_bias: float = 0.0,
    gt_center_z_m: float = 0.0,
) -> float:
    prior = compute_instant_prior(
        host,
        partner,
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
    return enu_range_from_prior(prior)


def apply_range_noise(
    range_m: float,
    std: float,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> float:
    if std <= 0.0:
        return range_m
    gen = rng if rng is not None else np.random.default_rng(seed)
    return float(range_m + gen.normal(0.0, std))


def build_uwb_range_readings(
    host_id: int,
    partners: Sequence[PartnerSpec],
    host_pose: Optional[Dict[str, float]],
    partner_poses: Dict[int, Optional[Dict[str, float]]],
    lat0_deg: float,
    lon0_deg: float,
    cfg: UwbConfig,
    geometry: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> List[UwbRangeReading]:
    """Build host -> each partner UWB readings."""
    if host_pose is None:
        return []

    readings: List[UwbRangeReading] = []
    alt0_m = float(geometry.get("alt0_m", 0.0))
    R_imu = np.asarray(geometry["R_imu_to_vehicle"], dtype=np.float64)
    R_lidar = np.asarray(geometry["R_lidar_vehicle"], dtype=np.float64)
    t_lidar = np.asarray(geometry["t_lidar_in_vehicle"], dtype=np.float64)
    target_offset = np.asarray(geometry["target_imu_to_center_offset"], dtype=np.float64)
    lidar_yaw_bias = float(geometry.get("lidar_yaw_bias", 0.0))
    gt_z = float(geometry.get("gt_center_z_m", 0.0))

    for spec in partners:
        p_pose = partner_poses.get(spec.partner_id)
        if p_pose is None:
            if cfg.skip_invalid:
                continue
            readings.append(
                UwbRangeReading(
                    from_id=host_id,
                    to_id=spec.partner_id,
                    range_m=0.0,
                    range_std=cfg.range_noise_std,
                    valid=False,
                    source=cfg.source_label,
                )
            )
            continue

        raw = enu_range_host_to_partner(
            host_pose,
            p_pose,
            lat0_deg,
            lon0_deg,
            alt0_m,
            R_imu,
            R_lidar,
            t_lidar,
            target_offset,
            lidar_yaw_bias,
            gt_center_z_m=gt_z,
        )
        noisy = apply_range_noise(
            raw,
            cfg.range_noise_std,
            rng=rng,
            seed=spec.partner_id if rng is None else None,
        )
        readings.append(
            UwbRangeReading(
                from_id=host_id,
                to_id=spec.partner_id,
                range_m=noisy,
                range_std=cfg.range_noise_std,
                valid=True,
                source=cfg.source_label,
            )
        )
    return readings
