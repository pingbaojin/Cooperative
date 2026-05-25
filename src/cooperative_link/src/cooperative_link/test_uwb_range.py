"""Unit tests for NAV-sim UWB range from InstantPrior."""

from __future__ import annotations

import unittest

import numpy as np

from cooperative_link.prior import InstantPrior, compute_instant_prior
from cooperative_link.uwb_range import apply_range_noise, enu_range_from_prior


class TestUwbRange(unittest.TestCase):
    def test_enu_range_from_delta_enu(self) -> None:
        prior = InstantPrior(
            r_prior_m=5.0,
            bearing_prior_rad=0.0,
            center_lidar=np.array([3.0, 4.0, 0.0]),
            yaw_lidar=0.0,
            center_vehicle=np.zeros(3),
            delta_enu=np.array([3.0, 4.0, 0.0]),
        )
        self.assertAlmostEqual(enu_range_from_prior(prior), 5.0, places=6)

    def test_matches_compute_instant_prior(self) -> None:
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        target = {"lat_deg": 30.0001, "lon_deg": 120.0001, "yaw_deg": 90.0}
        prior = compute_instant_prior(
            host,
            target,
            lat0_deg=30.0,
            lon0_deg=120.0,
            alt0_m=0.0,
            R_imu_to_vehicle=np.eye(3),
            R_lidar_vehicle=np.eye(3),
            t_lidar_in_vehicle=np.zeros(3),
            target_imu_to_center_offset=np.zeros(3),
            lidar_yaw_bias=0.0,
        )
        d = enu_range_from_prior(prior)
        expected = float(np.hypot(prior.delta_enu[0], prior.delta_enu[1]))
        self.assertAlmostEqual(d, expected, places=6)
        self.assertGreater(d, 0.0)

    def test_noise_zero_unchanged(self) -> None:
        self.assertEqual(apply_range_noise(10.0, 0.0), 10.0)

    def test_noise_nonzero(self) -> None:
        rng = np.random.default_rng(0)
        out = apply_range_noise(10.0, 0.5, rng=rng)
        self.assertNotAlmostEqual(out, 10.0, places=3)


if __name__ == "__main__":
    unittest.main()
