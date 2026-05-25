"""Select link partner by trajectory alignment with NAV reference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import CalibrationConfig, PartnerMatchConfig
from cooperative_link.track_filter import TrackOutput
from cooperative_link.trajectory_align import align_trajectories


@dataclass
class PartnerCandidate:
    track_id: int
    align_cost: float
    align_valid: bool
    cluster: Optional[ClusterDetection]


@dataclass
class PartnerMatchResult:
    partner_track_id: int = -1
    partner_cluster: Optional[ClusterDetection] = None
    traj_valid: bool = False
    align_cost: float = float("inf")
    candidates: List[PartnerCandidate] = field(default_factory=list)


class PartnerTrajectoryMatcher:
    def __init__(
        self,
        match_cfg: PartnerMatchConfig,
        cal_cfg: CalibrationConfig,
        link_score_thresh: float,
    ) -> None:
        self.match_cfg = match_cfg
        self.cal_cfg = cal_cfg
        self.link_score_thresh = float(link_score_thresh)
        self._nav_buf: Deque[np.ndarray] = deque(maxlen=max(match_cfg.window_frames, 1))
        self._track_bufs: Dict[int, Deque[np.ndarray]] = {}
        self._locked_track_id: int = -1
        self._lock_age: int = 0
        self._miss_count: int = 0

    def reset(self) -> None:
        self._nav_buf.clear()
        self._track_bufs.clear()
        self._locked_track_id = -1
        self._lock_age = 0
        self._miss_count = 0

    def _push_track_sample(self, track_id: int, xy: np.ndarray) -> None:
        if track_id < 0:
            return
        buf = self._track_bufs.setdefault(
            track_id, deque(maxlen=max(self.match_cfg.window_frames, 1))
        )
        buf.append(xy.copy())

    def _eval_track(self, track_id: int) -> Tuple[float, bool]:
        nav_samples = list(self._nav_buf)
        trk_samples = list(self._track_bufs.get(track_id, []))
        if len(nav_samples) < self.match_cfg.min_frames or len(trk_samples) < self.match_cfg.min_frames:
            return float("inf"), False
        n = min(len(nav_samples), len(trk_samples))
        TS = np.array(nav_samples[-n:], dtype=np.float64).T
        TD = np.array(trk_samples[-n:], dtype=np.float64).T
        result = align_trajectories(TS, TD, self.cal_cfg)
        if not result.valid:
            return float("inf"), False
        ok = result.cost <= self.match_cfg.max_align_cost
        return float(result.cost), ok

    def step(
        self,
        nav_xy: np.ndarray,
        clusters: List[ClusterDetection],
        track_outputs: List[TrackOutput],
    ) -> PartnerMatchResult:
        out = PartnerMatchResult()
        if not self.match_cfg.enabled:
            if clusters:
                best = clusters[0]
                if (
                    not self.match_cfg.require_match_score
                    or best.match_score >= self.link_score_thresh
                ):
                    out.partner_cluster = best
                    out.traj_valid = True
                    if track_outputs:
                        out.partner_track_id = track_outputs[0].track_id
            return out

        self._nav_buf.append(nav_xy[:2].copy())
        for trk in track_outputs:
            xy = trk.poly_xy if hasattr(trk, "poly_xy") else trk.ekf_xy
            self._push_track_sample(trk.track_id, xy)

        candidates: List[PartnerCandidate] = []
        for trk in track_outputs:
            cost, valid = self._eval_track(trk.track_id)
            cluster = cluster_for_track(clusters, trk)
            candidates.append(
                PartnerCandidate(
                    track_id=trk.track_id,
                    align_cost=cost,
                    align_valid=valid,
                    cluster=cluster,
                )
            )
        candidates.sort(key=lambda c: c.align_cost)
        out.candidates = candidates

        best_valid: Optional[PartnerCandidate] = None
        for cand in candidates:
            if not cand.align_valid or cand.cluster is None:
                continue
            if (
                self.match_cfg.require_match_score
                and cand.cluster.match_score < self.link_score_thresh
            ):
                continue
            best_valid = cand
            break

        selected: Optional[PartnerCandidate] = None
        if self._locked_track_id >= 0:
            locked = next(
                (c for c in candidates if c.track_id == self._locked_track_id),
                None,
            )
            if locked is not None and locked.align_valid and locked.cluster is not None:
                if (
                    not self.match_cfg.require_match_score
                    or locked.cluster.match_score >= self.link_score_thresh
                ):
                    selected = locked
                    self._miss_count = 0
                    self._lock_age += 1
                else:
                    self._miss_count += 1
            else:
                self._miss_count += 1

            if selected is None and self._lock_age < self.match_cfg.partner_lock_frames:
                for cand in candidates:
                    if cand.track_id == self._locked_track_id and cand.cluster is not None:
                        selected = cand
                        break

            if self._miss_count >= self.match_cfg.partner_unlock_miss:
                self._locked_track_id = -1
                self._lock_age = 0
                self._miss_count = 0

        if selected is None and best_valid is not None:
            selected = best_valid
            self._locked_track_id = best_valid.track_id
            self._lock_age = 0
            self._miss_count = 0

        if selected is not None and selected.cluster is not None:
            out.partner_track_id = selected.track_id
            out.partner_cluster = selected.cluster
            out.align_cost = selected.align_cost
            out.traj_valid = selected.align_valid

        return out


def cluster_for_track(
    clusters: List[ClusterDetection],
    track_out: TrackOutput,
) -> Optional[ClusterDetection]:
    """Nearest cluster centroid to track EKF position."""
    if not clusters:
        return None
    c_xy = track_out.ekf_xy
    best = min(
        clusters,
        key=lambda c: float(np.linalg.norm(c.centroid_xy - c_xy)),
    )
    return best
