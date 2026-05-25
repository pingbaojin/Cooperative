"""End-to-end post-processor buffer + align."""

import unittest

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import CalibrationConfig, FilterConfig
from cooperative_link.post_process import CooperativePostProcessor
from cooperative_link.track_filter import TrackFilterManager


def _cluster(xy: np.ndarray, score: float = 0.9) -> ClusterDetection:
    return ClusterDetection(
        centroid_xy=xy,
        r_det_m=float(np.hypot(xy[0], xy[1])),
        bearing_det_rad=float(np.arctan2(xy[1], xy[0])),
        n_points=10,
        motion_score=0.5,
        match_score=score,
        label=0,
    )


class TestFilterPipeline(unittest.TestCase):
    def test_align_becomes_valid_with_window(self) -> None:
        filt = FilterConfig(enabled=True, score_thresh=0.0)
        cal = CalibrationConfig(enabled=True, window_frames=10, min_frames=5)
        pp = CooperativePostProcessor(filt, cal, gt_z=0.0)
        tracker = TrackFilterManager(filt, z_default=0.0)
        c = _cluster(np.array([1.0, 0.0]))
        last = None
        for i in range(12):
            nav = np.array([float(i) * 0.1, 0.0])
            tracks = tracker.update([c], dt=0.1, score_thresh=0.0)
            last = pp.process_frame(
                c,
                nav,
                t_nav=float(i),
                track_outputs=tracks,
                tracker=tracker,
            )
        assert last is not None
        self.assertTrue(last.align_valid)
        self.assertIsNotNone(last.filtered_xy)

    def test_filtered_not_nav_when_detection_offset(self) -> None:
        filt = FilterConfig(enabled=True, score_thresh=0.0)
        cal = CalibrationConfig(enabled=False)
        pp = CooperativePostProcessor(filt, cal, gt_z=0.0)
        tracker = TrackFilterManager(filt, z_default=0.0)
        nav = np.array([0.0, 0.0])
        det = _cluster(np.array([3.0, 4.0]))
        tracker.update([det], dt=0.1, score_thresh=0.0)
        out = pp.process_frame(
            det,
            nav,
            t_nav=0.0,
            track_outputs=tracker.update([det], dt=0.1, score_thresh=0.0),
            tracker=tracker,
        )
        self.assertIsNotNone(out.filtered_xy)
        self.assertGreater(float(np.linalg.norm(out.filtered_xy - nav)), 0.5)

    def test_no_cluster_no_filtered(self) -> None:
        filt = FilterConfig(enabled=True, score_thresh=0.0)
        cal = CalibrationConfig(enabled=False)
        pp = CooperativePostProcessor(filt, cal, gt_z=0.0)
        out = pp.process_frame(None, np.array([1.0, 2.0]), t_nav=0.0)
        self.assertIsNone(out.filtered_xy)

    def test_pick_track_none_without_cluster(self) -> None:
        mgr = TrackFilterManager(FilterConfig(enabled=True, score_thresh=0.0))
        c = _cluster(np.array([1.0, 0.0]))
        mgr.update([c], dt=0.1)
        self.assertIsNone(mgr.pick_track_for_cluster(None))


if __name__ == "__main__":
    unittest.main()
