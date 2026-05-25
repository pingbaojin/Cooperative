"""Association state machine: Unlocked -> Locking -> Locked (+ Reacquire)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from cooperative_link.config import LinkConfig
from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.prior import InstantPrior


class LinkState(str, Enum):
    UNLOCKED = "unlocked"
    LOCKING = "locking"
    LOCKED = "locked"
    REACQUIRE = "reacquire"


def finite_diff_vel_xy(
    centers: np.ndarray, times: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Same logic as build_infos._finite_diff_vel_xy."""
    n = centers.shape[0]
    vx = np.zeros(n, dtype=np.float64)
    vy = np.zeros(n, dtype=np.float64)
    if n == 1:
        return vx, vy
    for i in range(n):
        if i == 0:
            dt = times[1] - times[0]
            if abs(dt) > 1e-9:
                vx[i] = (centers[1, 0] - centers[0, 0]) / dt
                vy[i] = (centers[1, 1] - centers[0, 1]) / dt
        elif i == n - 1:
            dt = times[i] - times[i - 1]
            if abs(dt) > 1e-9:
                vx[i] = (centers[i, 0] - centers[i - 1, 0]) / dt
                vy[i] = (centers[i, 1] - centers[i - 1, 1]) / dt
        else:
            dt = times[i + 1] - times[i - 1]
            if abs(dt) > 1e-9:
                vx[i] = (centers[i + 1, 0] - centers[i - 1, 0]) / dt
                vy[i] = (centers[i + 1, 1] - centers[i - 1, 1]) / dt
    return vx, vy


@dataclass
class StepResult:
    state: LinkState
    best_cluster: Optional[ClusterDetection]
    center_fused_xy: np.ndarray
    center_nav_xy: np.ndarray
    match_score: float
    consecutive_hits: int
    miss_frames: int
    used_coast: bool


@dataclass
class CooperativeAssociator:
    link_cfg: LinkConfig
    gt_center_z_m: float = 0.0
    state: LinkState = LinkState.UNLOCKED
    consecutive_hits: int = 0
    miss_frames: int = 0
    last_centroid_xy: Optional[np.ndarray] = None
    _nav_centers: List[np.ndarray] = field(default_factory=list)
    _nav_times: List[float] = field(default_factory=list)
    _vx: float = 0.0
    _vy: float = 0.0
    _coast_count: int = 0
    _last_fused_xy: Optional[np.ndarray] = None

    def _pick_best(
        self, clusters: List[ClusterDetection]
    ) -> Optional[ClusterDetection]:
        if not clusters:
            return None
        best = clusters[0]
        if best.match_score < self.link_cfg.score_thresh:
            return None
        if self.last_centroid_xy is not None and self.state in (
            LinkState.LOCKING,
            LinkState.LOCKED,
            LinkState.REACQUIRE,
        ):
            jump = float(np.linalg.norm(best.centroid_xy - self.last_centroid_xy))
            if jump > self.link_cfg.assoc_max_jump_m:
                return None
        return best

    def _update_nav_velocity(self, prior: InstantPrior, t_nav: float) -> None:
        c = prior.center_lidar[:2].copy()
        self._nav_centers.append(c)
        self._nav_times.append(float(t_nav))
        if len(self._nav_centers) > 500:
            self._nav_centers = self._nav_centers[-500:]
            self._nav_times = self._nav_times[-500:]
        arr = np.stack(self._nav_centers, axis=0)
        ts = np.asarray(self._nav_times, dtype=np.float64)
        vx, vy = finite_diff_vel_xy(arr, ts)
        self._vx = float(vx[-1])
        self._vy = float(vy[-1])

    def _coast_xy(self, dt: float) -> np.ndarray:
        base = self._last_fused_xy
        if base is None:
            base = np.zeros(2, dtype=np.float64)
        return base + np.array([self._vx * dt, self._vy * dt], dtype=np.float64)

    def step(
        self,
        prior: InstantPrior,
        clusters: List[ClusterDetection],
        t_nav: float,
        dt: float,
    ) -> StepResult:
        nav_xy = prior.center_lidar[:2].copy()
        self._update_nav_velocity(prior, t_nav)
        best = self._pick_best(clusters)
        used_coast = False

        if self.state == LinkState.UNLOCKED:
            if best is not None:
                self.state = LinkState.LOCKING
                self.consecutive_hits = 1
                self.last_centroid_xy = best.centroid_xy.copy()
                fused = best.centroid_xy.copy()
            else:
                self.consecutive_hits = 0
                fused = nav_xy

        elif self.state == LinkState.LOCKING:
            if best is not None:
                self.consecutive_hits += 1
                self.last_centroid_xy = best.centroid_xy.copy()
                fused = best.centroid_xy.copy()
                if self.consecutive_hits >= self.link_cfg.lock_frames:
                    self.state = LinkState.LOCKED
                    self.miss_frames = 0
                    self._coast_count = 0
            else:
                self.state = LinkState.UNLOCKED
                self.consecutive_hits = 0
                self.last_centroid_xy = None
                fused = nav_xy

        elif self.state == LinkState.LOCKED:
            if best is not None:
                self.miss_frames = 0
                self._coast_count = 0
                self.last_centroid_xy = best.centroid_xy.copy()
                fused = nav_xy
            else:
                self.miss_frames += 1
                if self.miss_frames <= self.link_cfg.coast_max_frames:
                    fused = self._coast_xy(max(dt, 0.0))
                    used_coast = True
                    self._coast_count += 1
                elif self.miss_frames >= self.link_cfg.reacquire_miss_frames:
                    self.state = LinkState.REACQUIRE
                    fused = nav_xy
                else:
                    fused = nav_xy

        else:  # REACQUIRE
            if best is not None:
                self.state = LinkState.LOCKED
                self.miss_frames = 0
                self.consecutive_hits = self.link_cfg.lock_frames
                self.last_centroid_xy = best.centroid_xy.copy()
                fused = nav_xy
            else:
                self.miss_frames += 1
                fused = nav_xy
                if self.miss_frames >= self.link_cfg.reacquire_miss_frames:
                    self.state = LinkState.UNLOCKED
                    self.consecutive_hits = 0
                    self.last_centroid_xy = None

        self._last_fused_xy = fused.copy()
        score = float(best.match_score) if best is not None else 0.0
        return StepResult(
            state=self.state,
            best_cluster=best,
            center_fused_xy=fused,
            center_nav_xy=nav_xy,
            match_score=score,
            consecutive_hits=self.consecutive_hits,
            miss_frames=self.miss_frames,
            used_coast=used_coast,
        )
