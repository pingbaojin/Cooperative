"""Post-processing: trajectory alignment after link FSM (KF runs in pipeline)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import CalibrationConfig, FilterConfig, load_calibration_config, load_filter_config
from cooperative_link.track_filter import TrackFilterManager, TrackOutput
from cooperative_link.trajectory_align import AlignResult, align_trajectories


@dataclass
class PostProcessResult:
    filtered_xy: Optional[np.ndarray] = None
    track_id: int = -1
    yaw_error_rad: float = 0.0
    tx: float = 0.0
    ty: float = 0.0
    align_cost: float = float("inf")
    align_valid: bool = False


@dataclass
class _FrameSample:
    nav_xy: np.ndarray
    det_xy: np.ndarray
    filtered_xy: np.ndarray
    t_nav: float


class CooperativePostProcessor:
    def __init__(
        self,
        filter_cfg: FilterConfig,
        cal_cfg: CalibrationConfig,
        gt_z: float = 0.0,
    ) -> None:
        self.filter_cfg = filter_cfg
        self.cal_cfg = cal_cfg
        self.gt_z = gt_z
        self._nav_buf: Deque[_FrameSample] = deque(maxlen=max(cal_cfg.window_frames, 1))

    @classmethod
    def from_yaml(cls, cfg: dict, gt_z: float = 0.0) -> "CooperativePostProcessor":
        return cls(load_filter_config(cfg), load_calibration_config(cfg), gt_z=gt_z)

    def process_frame(
        self,
        best_cluster: Optional[ClusterDetection],
        nav_xy: np.ndarray,
        t_nav: float,
        track_outputs: Optional[List[TrackOutput]] = None,
        tracker: Optional[TrackFilterManager] = None,
    ) -> PostProcessResult:
        """NAV is GT for alignment only; filtered positions come from pipeline KF."""
        out = PostProcessResult()
        if not self.filter_cfg.enabled and not self.cal_cfg.enabled:
            return out

        det_xy: Optional[np.ndarray] = None
        if best_cluster is not None:
            det_xy = best_cluster.centroid_xy.copy()

        track_out: Optional[TrackOutput] = None
        if self.filter_cfg.enabled and tracker is not None and best_cluster is not None:
            track_out = tracker.pick_track_for_cluster(best_cluster)

        if track_out is not None:
            out.track_id = track_out.track_id
            if self.filter_cfg.filter_output == "poly":
                out.filtered_xy = track_out.poly_xy.copy()
            else:
                out.filtered_xy = track_out.ekf_xy.copy()
        elif track_outputs and best_cluster is not None:
            for trk in track_outputs:
                d = float(np.linalg.norm(trk.ekf_xy - best_cluster.centroid_xy))
                if d < 3.0:
                    out.track_id = trk.track_id
                    if self.filter_cfg.filter_output == "poly":
                        out.filtered_xy = trk.poly_xy.copy()
                    else:
                        out.filtered_xy = trk.ekf_xy.copy()
                    break

        should_buffer = False
        if self.cal_cfg.enabled:
            if self.filter_cfg.enabled:
                should_buffer = out.filtered_xy is not None
            else:
                should_buffer = det_xy is not None

        if should_buffer:
            det_store = (
                det_xy.copy()
                if det_xy is not None
                else np.array([np.nan, np.nan], dtype=np.float64)
            )
            filtered_store = (
                out.filtered_xy.copy()
                if out.filtered_xy is not None
                else det_store.copy()
            )
            self._nav_buf.append(
                _FrameSample(
                    nav_xy=nav_xy.copy(),
                    det_xy=det_store,
                    filtered_xy=filtered_store,
                    t_nav=float(t_nav),
                )
            )

        if self.cal_cfg.enabled and len(self._nav_buf) >= self.cal_cfg.min_frames:
            TS = np.array([s.nav_xy for s in self._nav_buf], dtype=np.float64).T
            if self.filter_cfg.enabled:
                TD = np.array([s.filtered_xy for s in self._nav_buf], dtype=np.float64).T
            else:
                TD = np.array([s.det_xy for s in self._nav_buf], dtype=np.float64).T
            align: AlignResult = align_trajectories(TS, TD, self.cal_cfg)
            if align.valid:
                out.yaw_error_rad = align.theta_rad
                out.tx = float(align.t[0])
                out.ty = float(align.t[1])
                out.align_cost = align.cost
                out.align_valid = True

        return out
