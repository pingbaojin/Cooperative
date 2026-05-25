"""Tests for RViz dynamic marker selection and stale id cleanup."""

import unittest

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.dynamic_markers import (
    dynamic_marker_id,
    marker_ids_from_tracks,
    select_tracks_for_rviz,
    stale_marker_ids,
    RvizDynamicTrack,
)
from cooperative_link.dynamic_targets import DynamicTargetInfo
from cooperative_link.filter_config import DynamicConfig
from cooperative_link.track_filter import TrackOutput


def _dyn_target(tid: int, score: float = 0.8, partner: bool = False) -> DynamicTargetInfo:
    return DynamicTargetInfo(
        track_id=tid,
        cluster_label=0,
        x=float(tid),
        y=0.0,
        z=0.0,
        range_m=1.0,
        bearing_rad=0.0,
        match_score=score,
        motion_score=0.5,
        is_link_partner=partner,
        traj_align_cost=0.0,
        traj_match_valid=partner,
    )


def _trk(tid: int, x: float, coasted: bool = False) -> TrackOutput:
    return TrackOutput(
        track_id=tid,
        ekf_xy=__import__("numpy").array([x, 0.0]),
        poly_xy=__import__("numpy").array([x, 0.0]),
        valid=True,
        coasted=coasted,
        missed=1 if coasted else 0,
    )


class TestDynamicMarkers(unittest.TestCase):
    def test_stale_marker_ids(self) -> None:
        prev = {101, 102, 103}
        curr = {102, 104}
        self.assertEqual(stale_marker_ids(prev, curr), {101, 103})

    def test_dynamic_marker_id_stable(self) -> None:
        self.assertEqual(dynamic_marker_id(1), 101)
        self.assertEqual(dynamic_marker_id(2), 102)

    def test_select_caps_non_partner(self) -> None:
        cfg = DynamicConfig(
            rviz_tracks_only=True,
            max_dynamic_markers=2,
            min_motion_score=0.0,
        )
        tracks = [_trk(1, 1.0), _trk(2, 2.0), _trk(3, 3.0)]
        dts = [
            _dyn_target(1, 0.9),
            _dyn_target(2, 0.8),
            _dyn_target(3, 0.7),
        ]
        picked = select_tracks_for_rviz(tracks, dts, -1, cfg)
        self.assertEqual(len(picked), 2)
        self.assertEqual(picked[0].track_id, 1)

    def test_partner_always_included(self) -> None:
        cfg = DynamicConfig(
            rviz_tracks_only=True,
            max_dynamic_markers=2,
            min_motion_score=0.0,
        )
        tracks = [_trk(1, 1.0), _trk(5, 5.0)]
        dts = [_dyn_target(1, 0.3), _dyn_target(5, 0.9)]
        picked = select_tracks_for_rviz(tracks, dts, 1, cfg)
        ids = {p.track_id for p in picked}
        self.assertIn(1, ids)
        self.assertTrue(any(p.is_link_partner for p in picked if p.track_id == 1))

    def test_marker_ids_from_tracks(self) -> None:
        rows = [
            RvizDynamicTrack(1, 0.0, 0.0, 0.0, False, 0.5),
            RvizDynamicTrack(3, 0.0, 0.0, 0.0, False, 0.5),
        ]
        self.assertEqual(marker_ids_from_tracks(rows), {101, 103})


if __name__ == "__main__":
    unittest.main()
