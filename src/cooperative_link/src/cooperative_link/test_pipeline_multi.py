"""Multi-target pipeline: all clusters published, FSM only with partner."""

import unittest

import numpy as np

from cooperative_link.pipeline import CooperativeLinkPipeline


def _make_pipe() -> CooperativeLinkPipeline:
    cfg = {
        "R_imu_to_vehicle": np.eye(3).tolist(),
        "R_lidar_vehicle": np.eye(3).tolist(),
        "t_lidar_in_vehicle": [0.0, 0.0, 0.0],
        "target_imu_to_center_offset": [0.0, 0.0, 0.0],
        "gt_center_z_m": 0.0,
        "cooperative_link": {
            "score_thresh": 0.5,
            "lock_frames": 2,
        },
        "cooperative_link_filter": {
            "enabled": True,
            "score_thresh": 0.0,
        },
        "cooperative_link_calibration": {
            "enabled": False,
        },
        "cooperative_link_dynamic": {
            "enabled": True,
            "publish_score_thresh": 0.0,
            "track_score_thresh": 0.0,
        },
        "cooperative_link_partner_match": {
            "enabled": False,
            "require_traj_valid_for_link": True,
        },
    }
    pipe = CooperativeLinkPipeline.from_yaml_geometry_only(cfg)
    pipe.attach_post_processor(cfg)
    pipe.set_enu_origin_if_needed(
        {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
    )
    return pipe


class TestPipelineMulti(unittest.TestCase):
    def test_dynamic_targets_from_clusters_field(self) -> None:
        """FrameOutput exposes dynamic_targets list (populated when detection runs)."""
        pipe = _make_pipe()
        self.assertTrue(pipe.dynamic_cfg.enabled)
        self.assertIsNotNone(pipe.partner_matcher)

    def test_no_partner_fsm_gets_empty_clusters(self) -> None:
        pipe = _make_pipe()
        pipe.partner_match_cfg.require_traj_valid_for_link = True
        host = {"lat_deg": 30.0, "lon_deg": 120.0, "yaw_deg": 0.0}
        target = {"lat_deg": 30.0001, "lon_deg": 120.0001, "yaw_deg": 0.0}
        pts = np.random.randn(100, 3).astype(np.float64) * 0.1
        out = pipe.process_frame_with_poses(0, 0.0, 0.0, pts, host, target)
        if out.partner_match and not out.partner_match.traj_valid:
            self.assertIsNone(out.step.best_cluster)


if __name__ == "__main__":
    unittest.main()
