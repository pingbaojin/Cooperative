"""Tests for CV predict-on-miss (velocity coast)."""

import unittest

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import FilterConfig
from cooperative_link.track_filter import TrackFilterManager, init_track, kalman_predict


def _cluster(x: float, y: float = 0.0) -> ClusterDetection:
    xy = np.array([x, y], dtype=np.float64)
    return ClusterDetection(
        centroid_xy=xy,
        r_det_m=float(np.hypot(x, y)),
        bearing_det_rad=float(np.arctan2(y, x)),
        n_points=10,
        motion_score=0.8,
        match_score=0.9,
        label=0,
    )


class TestTrackPredict(unittest.TestCase):
    def test_coast_advances_position(self) -> None:
        cfg = FilterConfig(
            enabled=True,
            score_thresh=0.0,
            predict_on_miss=True,
            coast_publish=True,
            max_missed=5,
            max_speed_mps=50.0,
        )
        mgr = TrackFilterManager(cfg, z_default=0.0)
        dt = 0.1
        outs = mgr.update([_cluster(0.0)], dt=dt)
        self.assertEqual(len(outs), 1)
        x0 = float(outs[0].ekf_xy[0])

        outs = mgr.update([_cluster(0.5)], dt=dt)
        x1 = float(outs[0].ekf_xy[0])
        self.assertGreater(x1, x0)

        for _ in range(3):
            outs = mgr.update([], dt=dt)
        self.assertEqual(len(outs), 1)
        self.assertTrue(outs[0].coasted)
        self.assertGreater(outs[0].missed, 0)
        x_coast = float(outs[0].ekf_xy[0])
        self.assertGreater(x_coast, x1)

    def test_track_removed_after_max_missed(self) -> None:
        cfg = FilterConfig(
            enabled=True,
            score_thresh=0.0,
            predict_on_miss=True,
            coast_publish=True,
            max_missed=2,
        )
        mgr = TrackFilterManager(cfg, z_default=0.0)
        mgr.update([_cluster(0.0)], dt=0.1)
        mgr.update([_cluster(0.1)], dt=0.1)
        for _ in range(4):
            outs = mgr.update([], dt=0.1)
        self.assertEqual(len(outs), 0)

    def test_kalman_predict_increments_position_with_velocity(self) -> None:
        trk = init_track(np.array([0.0, 0.0, 0.0]), 1)
        trk.x[3, 0] = 2.0
        kalman_predict(trk, 0.5, 0.1)
        self.assertAlmostEqual(float(trk.x[0, 0]), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
