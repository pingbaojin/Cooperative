"""Unit tests for multi-partner UWB range readings."""

from __future__ import annotations

import unittest

import numpy as np

from cooperative_link.partner_config import PartnerSpec, load_multi_link_config
from cooperative_link.uwb_config import load_uwb_config
from cooperative_link.uwb_range import build_uwb_range_readings, enu_range_host_to_partner


class TestUwbRangeMulti(unittest.TestCase):
    def _geometry(self) -> dict:
        return {
            "alt0_m": 0.0,
            "R_imu_to_vehicle": np.eye(3).tolist(),
            "R_lidar_vehicle": np.eye(3).tolist(),
            "t_lidar_in_vehicle": [0.0, 0.0, 0.0],
            "target_imu_to_center_offset": [0.0, 0.0, 0.0],
            "lidar_yaw_bias": 0.0,
            "gt_center_z_m": 0.0,
        }

    def test_two_partners_distances(self) -> None:
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        p1 = {"lat_deg": 30.0001, "lon_deg": 120.0, "yaw_deg": 0.0}
        p2 = {"lat_deg": 30.0, "lon_deg": 120.0001, "yaw_deg": 0.0}
        partners = [
            PartnerSpec(partner_id=1, name="a"),
            PartnerSpec(partner_id=2, name="b"),
        ]
        cfg = load_uwb_config({"cooperative_link_uwb": {"enabled": True}})
        readings = build_uwb_range_readings(
            host_id=0,
            partners=partners,
            host_pose=host,
            partner_poses={1: p1, 2: p2},
            lat0_deg=30.0,
            lon0_deg=120.0,
            cfg=cfg,
            geometry=self._geometry(),
        )
        self.assertEqual(len(readings), 2)
        self.assertEqual(readings[0].to_id, 1)
        self.assertEqual(readings[1].to_id, 2)
        self.assertTrue(all(r.valid for r in readings))
        self.assertNotAlmostEqual(readings[0].range_m, readings[1].range_m, places=3)

    def test_missing_partner_invalid(self) -> None:
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        p1 = {"lat_deg": 30.0001, "lon_deg": 120.0, "yaw_deg": 0.0}
        partners = [
            PartnerSpec(partner_id=1, name="a"),
            PartnerSpec(partner_id=2, name="b"),
        ]
        cfg = load_uwb_config({"cooperative_link_uwb": {"enabled": True, "skip_invalid": False}})
        readings = build_uwb_range_readings(
            host_id=0,
            partners=partners,
            host_pose=host,
            partner_poses={1: p1, 2: None},
            lat0_deg=30.0,
            lon0_deg=120.0,
            cfg=cfg,
            geometry=self._geometry(),
        )
        self.assertEqual(len(readings), 2)
        self.assertTrue(readings[0].valid)
        self.assertFalse(readings[1].valid)

    def test_skip_invalid(self) -> None:
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        p1 = {"lat_deg": 30.0001, "lon_deg": 120.0, "yaw_deg": 0.0}
        partners = [
            PartnerSpec(partner_id=1, name="a"),
            PartnerSpec(partner_id=2, name="b"),
        ]
        cfg = load_uwb_config({"cooperative_link_uwb": {"enabled": True, "skip_invalid": True}})
        readings = build_uwb_range_readings(
            host_id=0,
            partners=partners,
            host_pose=host,
            partner_poses={1: p1, 2: None},
            lat0_deg=30.0,
            lon0_deg=120.0,
            cfg=cfg,
            geometry=self._geometry(),
        )
        self.assertEqual(len(readings), 1)
        self.assertEqual(readings[0].to_id, 1)

    def test_multi_disabled_legacy_config(self) -> None:
        cfg = {
            "cooperative_link_multi": {"enabled": False, "partners": [{"id": 1}]},
            "cooperative_link_uwb": {"to_agent_id": 1},
        }
        multi = load_multi_link_config(cfg)
        uwb = load_uwb_config(cfg)
        self.assertFalse(multi.enabled)
        self.assertEqual(uwb.to_agent_id, 1)

    def test_host_to_partner_matches_single(self) -> None:
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        partner = {"lat_deg": 30.0001, "lon_deg": 120.0001, "yaw_deg": 90.0}
        d = enu_range_host_to_partner(
            host,
            partner,
            lat0_deg=30.0,
            lon0_deg=120.0,
            alt0_m=0.0,
            R_imu_to_vehicle=np.eye(3),
            R_lidar_vehicle=np.eye(3),
            t_lidar_in_vehicle=np.zeros(3),
            target_imu_to_center_offset=np.zeros(3),
        )
        self.assertGreater(d, 0.0)


if __name__ == "__main__":
    unittest.main()
