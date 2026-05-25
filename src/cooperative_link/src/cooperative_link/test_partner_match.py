"""Tests for partner trajectory matcher."""

import unittest

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import CalibrationConfig, PartnerMatchConfig
from cooperative_link.partner_match import PartnerTrajectoryMatcher
from cooperative_link.track_filter import TrackOutput


def _cluster(xy: np.ndarray, label: int = 0, score: float = 0.9) -> ClusterDetection:
    return ClusterDetection(
        centroid_xy=xy,
        r_det_m=float(np.hypot(xy[0], xy[1])),
        bearing_det_rad=float(np.arctan2(xy[1], xy[0])),
        n_points=10,
        motion_score=0.8,
        match_score=score,
        label=label,
    )


def _track(tid: int, xy: np.ndarray) -> TrackOutput:
    return TrackOutput(track_id=tid, ekf_xy=xy, poly_xy=xy, valid=True)


class TestPartnerMatch(unittest.TestCase):
    def test_picks_nav_aligned_track(self) -> None:
        match_cfg = PartnerMatchConfig(
            enabled=True,
            window_frames=20,
            min_frames=5,
            max_align_cost=10.0,
            partner_lock_frames=1,
            partner_unlock_miss=3,
            require_match_score=False,
        )
        cal = CalibrationConfig(
            enabled=True,
            window_frames=20,
            min_frames=5,
            lambda_t=2.0,
            lambda_v=3.5,
        )
        matcher = PartnerTrajectoryMatcher(match_cfg, cal, link_score_thresh=0.0)

        c_good = _cluster(np.array([1.0, 0.0]), label=0)
        c_bad = _cluster(np.array([10.0, 10.0]), label=1)
        clusters = [c_good, c_bad]

        result = None
        for i in range(12):
            nav = np.array([float(i) * 0.2, 0.0])
            trk_good = _track(1, nav + np.array([0.05, 0.02]))
            trk_bad = _track(2, np.array([10.0 + i * 0.01, 10.0]))
            result = matcher.step(nav, clusters, [trk_good, trk_bad])

        assert result is not None
        self.assertTrue(result.traj_valid)
        self.assertEqual(result.partner_track_id, 1)
        self.assertIsNotNone(result.partner_cluster)
        self.assertEqual(result.partner_cluster.label, 0)

    def test_hysteresis_keeps_partner(self) -> None:
        match_cfg = PartnerMatchConfig(
            enabled=True,
            min_frames=3,
            window_frames=10,
            partner_lock_frames=5,
            partner_unlock_miss=5,
            require_match_score=False,
        )
        cal = CalibrationConfig(min_frames=3, window_frames=10)
        matcher = PartnerTrajectoryMatcher(match_cfg, cal, link_score_thresh=0.0)
        c = _cluster(np.array([1.0, 0.0]))
        for i in range(8):
            nav = np.array([float(i) * 0.1, 0.0])
            trk = _track(1, nav)
            matcher.step(nav, [c], [trk])
        r1 = matcher.step(np.array([0.8, 0.0]), [c], [_track(1, np.array([0.8, 0.0]))])
        self.assertEqual(r1.partner_track_id, 1)


if __name__ == "__main__":
    unittest.main()
