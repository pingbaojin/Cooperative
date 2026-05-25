"""Tests for KF track filter."""

import unittest

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import FilterConfig
from cooperative_link.track_filter import (
    TrackFilterManager,
    associate_tracks,
    init_track,
    kalman_update,
    poly_fit_position,
)


class TestTrackFilter(unittest.TestCase):
    def test_kalman_moves_toward_measurement(self) -> None:
        trk = init_track(np.array([0.0, 0.0, 0.0]), 1)
        trk = kalman_update(trk, np.array([1.0, 0.0, 0.0]), 0.1, 0.1, 0.2)
        self.assertGreater(float(trk.x[0]), 0.0)

    def test_associate_gate(self) -> None:
        trk = init_track(np.array([0.0, 0.0, 0.0]), 1)
        dets = np.array([[0.5, 0.0, 0.0, 0.9]], dtype=np.float64)
        assign, un_t, un_d = associate_tracks([trk], dets, gate_m=2.5)
        self.assertEqual(assign.shape[0], 1)
        self.assertEqual(len(un_d), 0)

    def test_poly_fit(self) -> None:
        hist = np.array([[0, 0, 0], [1, 0.1, 0], [2, 0.4, 0]], dtype=np.float64)
        p = poly_fit_position(hist, order=2, fit_win=10)
        self.assertAlmostEqual(p[0], 2.0, places=3)

    def test_manager_creates_track(self) -> None:
        cfg = FilterConfig(enabled=True, score_thresh=0.0)
        mgr = TrackFilterManager(cfg, z_default=0.0)
        c = ClusterDetection(
            centroid_xy=np.array([1.0, 2.0]),
            r_det_m=2.2,
            bearing_det_rad=1.0,
            n_points=10,
            motion_score=0.8,
            match_score=0.9,
            label=0,
        )
        outs = mgr.update([c], dt=0.1)
        self.assertEqual(len(outs), 1)
        self.assertAlmostEqual(outs[0].ekf_xy[0], 1.0, places=1)


if __name__ == "__main__":
    unittest.main()
