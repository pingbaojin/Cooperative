"""Smoke tests for cooperative_link (no bag required)."""

from __future__ import annotations

import numpy as np

from cooperative_link.config import load_link_config
from cooperative_link.dynamic_detect import detect_dynamic_clusters, prior_gate_mask
from cooperative_link.prior import compute_instant_prior


def test_instant_prior_r_bearing():
    host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
    target = {"lat_deg": 30.0001, "lon_deg": 120.0001, "yaw_deg": 45.0}
    R = np.eye(3)
    t = np.array([0.2, 0.0, -0.3])
    prior = compute_instant_prior(
        host, target, 30.0, 120.0, 0.0, R, R, t, np.zeros(3), 0.0, gt_center_z_m=0.0
    )
    x, y = prior.center_lidar[0], prior.center_lidar[1]
    assert abs(prior.r_prior_m - np.hypot(x, y)) < 1e-6
    assert abs(prior.bearing_prior_rad - np.arctan2(y, x)) < 1e-6


def test_gate_and_cluster():
    cfg = load_link_config({"cooperative_link": {}, "target_size_wlh": [0.5, 0.7, 0.65]})
    host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
    target = {"lat_deg": 30.0002, "lon_deg": 120.0002, "yaw_deg": 0.0}
    R = np.eye(3)
    t = np.zeros(3)
    prior = compute_instant_prior(
        host, target, 30.0, 120.0, 0.0, R, R, t, np.zeros(3), 0.0
    )
    cx, cy = prior.center_lidar[0], prior.center_lidar[1]
    pts = np.random.randn(200, 5).astype(np.float32) * 0.1
    pts[:, 0] += cx
    pts[:, 1] += cy
    motion = np.ones(pts.shape[0]) * 0.5
    gate, clusters = detect_dynamic_clusters(pts, prior, cfg, motion, 0.0)
    assert gate.sum() > 10
    assert len(clusters) >= 1


if __name__ == "__main__":
    test_instant_prior_r_bearing()
    test_gate_and_cluster()
    print("cooperative_link tests OK")
