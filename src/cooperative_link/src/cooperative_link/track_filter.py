"""Multi-target CV Kalman tracking + polynomial smoothing (MATLAB Dynamic_Object port)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import FilterConfig


@dataclass
class TrackState:
    track_id: int
    x: np.ndarray  # (6,1) pos+vel
    P: np.ndarray  # (6,6)
    missed: int = 0
    det_hist: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class TrackOutput:
    track_id: int
    ekf_xy: np.ndarray
    poly_xy: np.ndarray
    valid: bool
    coasted: bool = False
    missed: int = 0


def _cv_matrices(
    dt: float, q_scale: float, r_scale: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    F = np.eye(6, dtype=np.float64)
    F[0:3, 3:6] = np.eye(3, dtype=np.float64) * dt
    H = np.zeros((3, 6), dtype=np.float64)
    H[0:3, 0:3] = np.eye(3)
    Q = q_scale * np.eye(6, dtype=np.float64)
    R = r_scale * np.eye(3, dtype=np.float64)
    return F, H, Q, R


def clamp_velocity(track: TrackState, max_speed_mps: float) -> None:
    """Limit horizontal speed magnitude after predict."""
    if max_speed_mps <= 0.0:
        return
    v = track.x[3:6, 0]
    v_xy = v[:2]
    spd = float(np.linalg.norm(v_xy))
    if spd <= max_speed_mps:
        return
    scale = max_speed_mps / max(spd, 1e-9)
    track.x[3, 0] *= scale
    track.x[4, 0] *= scale


def kalman_predict(track: TrackState, dt: float, q_scale: float) -> TrackState:
    F, _, Q, _ = _cv_matrices(dt, q_scale, 0.2)
    track.x = F @ track.x
    track.P = F @ track.P @ F.T + Q
    return track


def kalman_correct(track: TrackState, z: np.ndarray, r_scale: float) -> TrackState:
    """Measurement update on already-predicted state."""
    _, H, _, R = _cv_matrices(0.1, 0.1, r_scale)
    S = H @ track.P @ H.T + R
    K = track.P @ H.T @ np.linalg.inv(S)
    innov = z.reshape(3, 1) - H @ track.x
    track.x = track.x + K @ innov
    track.P = (np.eye(6) - K @ H) @ track.P
    return track


def kalman_update(
    track: TrackState,
    z: np.ndarray,
    dt: float,
    q_scale: float,
    r_scale: float,
) -> TrackState:
    """Full predict + correct (single-step convenience)."""
    kalman_predict(track, dt, q_scale)
    kalman_correct(track, z, r_scale)
    return track


def init_track(pos_xyz: np.ndarray, track_id: int) -> TrackState:
    x = np.zeros((6, 1), dtype=np.float64)
    x[0:3, 0] = np.asarray(pos_xyz, dtype=np.float64).reshape(3)
    return TrackState(
        track_id=track_id,
        x=x,
        P=np.eye(6, dtype=np.float64),
        missed=0,
        det_hist=np.asarray(pos_xyz, dtype=np.float64).reshape(1, 3),
    )


def associate_tracks(
    tracks: List[TrackState],
    detections: np.ndarray,
    gate_m: float,
    coast_gate_extra_m: float = 0.0,
) -> Tuple[np.ndarray, List[int], List[int]]:
    n_t = len(tracks)
    n_d = detections.shape[0] if detections.size else 0
    if n_t == 0 or n_d == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            list(range(n_t)),
            list(range(n_d)),
        )
    cost = np.zeros((n_t, n_d), dtype=np.float64)
    for i, trk in enumerate(tracks):
        pred = trk.x[0:3, 0]
        gate = gate_m + (coast_gate_extra_m if trk.missed > 0 else 0.0)
        for j in range(n_d):
            cost[i, j] = np.linalg.norm(pred - detections[j, :3])
    assignments: List[List[int]] = []
    used_dets: List[int] = []
    for i in range(n_t):
        gate = gate_m + (coast_gate_extra_m if tracks[i].missed > 0 else 0.0)
        j_min = int(np.argmin(cost[i, :]))
        if cost[i, j_min] < gate and j_min not in used_dets:
            assignments.append([i, j_min])
            used_dets.append(j_min)
    assign_arr = (
        np.asarray(assignments, dtype=np.int32)
        if assignments
        else np.zeros((0, 2), dtype=np.int32)
    )
    assigned_t = {a[0] for a in assignments}
    assigned_d = set(used_dets)
    unassigned_tracks = [i for i in range(n_t) if i not in assigned_t]
    unassigned_dets = [j for j in range(n_d) if j not in assigned_d]
    return assign_arr, unassigned_tracks, unassigned_dets


def poly_fit_position(
    hist: np.ndarray,
    order: int,
    fit_win: int,
) -> np.ndarray:
    """hist: (K,3) rows are observations; returns (3,) at end of window."""
    if hist.shape[0] == 0:
        return np.zeros(3, dtype=np.float64)
    pts = hist[-fit_win:] if hist.shape[0] > fit_win else hist
    fit_pos = np.zeros(3, dtype=np.float64)
    t_seq = np.arange(pts.shape[0], dtype=np.float64) + 1.0
    ord_use = min(order, max(pts.shape[0] - 1, 0))
    if ord_use < 1:
        return pts[-1].copy()
    for dim in range(3):
        coeff = np.polyfit(t_seq, pts[:, dim], ord_use)
        fit_pos[dim] = float(np.polyval(coeff, t_seq[-1]))
    return fit_pos


def clusters_to_detections(
    clusters: List[ClusterDetection],
    score_thresh: float,
    z_default: float,
    min_motion_score: float = 0.0,
) -> np.ndarray:
    rows = []
    for c in clusters:
        if c.match_score < score_thresh:
            continue
        if c.motion_score < min_motion_score:
            continue
        rows.append([c.centroid_xy[0], c.centroid_xy[1], z_default, c.match_score])
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


class TrackFilterManager:
    def __init__(self, cfg: FilterConfig, z_default: float = 0.0) -> None:
        self.cfg = cfg
        self.z_default = z_default
        self.tracks: List[TrackState] = []
        self._next_id = 1

    def update(
        self,
        clusters: List[ClusterDetection],
        dt: float,
        score_thresh: Optional[float] = None,
        min_motion_score: float = 0.0,
    ) -> List[TrackOutput]:
        if not self.cfg.enabled:
            return []
        thresh = self.cfg.score_thresh if score_thresh is None else score_thresh
        dets = clusters_to_detections(
            clusters, thresh, self.z_default, min_motion_score=min_motion_score
        )
        dt = max(float(dt), 1e-3) if dt <= 0 else float(dt)

        if self.cfg.predict_on_miss:
            for trk in self.tracks:
                kalman_predict(trk, dt, self.cfg.q_scale)
                clamp_velocity(trk, self.cfg.max_speed_mps)

        assign, un_trk, un_det = associate_tracks(
            self.tracks,
            dets,
            self.cfg.assoc_gate_m,
            coast_gate_extra_m=self.cfg.coast_gate_extra_m,
        )

        assigned_track_idx = set()
        for i in range(assign.shape[0]):
            ti, di = int(assign[i, 0]), int(assign[i, 1])
            z = dets[di, :3]
            if self.cfg.predict_on_miss:
                kalman_correct(self.tracks[ti], z, self.cfg.r_scale)
            else:
                self.tracks[ti] = kalman_update(
                    self.tracks[ti], z, dt, self.cfg.q_scale, self.cfg.r_scale
                )
            self.tracks[ti].missed = 0
            hist = self.tracks[ti].det_hist
            self.tracks[ti].det_hist = np.vstack([hist, z.reshape(1, 3)])
            if self.tracks[ti].det_hist.shape[0] > self.cfg.fit_win:
                self.tracks[ti].det_hist = self.tracks[ti].det_hist[-self.cfg.fit_win :]
            assigned_track_idx.add(ti)

        for ti in un_trk:
            self.tracks[ti].missed += 1

        self.tracks = [t for t in self.tracks if t.missed <= self.cfg.max_missed]

        for di in un_det:
            pos = dets[di, :3]
            self.tracks.append(init_track(pos, self._next_id))
            self._next_id += 1

        outputs: List[TrackOutput] = []
        for trk in self.tracks:
            if not self.cfg.coast_publish and trk.missed > 0:
                continue
            ekf = trk.x[0:3, 0].copy()
            coasted = trk.missed > 0
            if coasted or trk.det_hist.shape[0] < 2:
                poly = ekf.copy()
            else:
                poly = poly_fit_position(
                    trk.det_hist, self.cfg.poly_order, self.cfg.fit_win
                )
            outputs.append(
                TrackOutput(
                    track_id=trk.track_id,
                    ekf_xy=ekf[:2],
                    poly_xy=poly[:2],
                    valid=True,
                    coasted=coasted,
                    missed=int(trk.missed),
                )
            )
        return outputs

    def pick_track_for_cluster(
        self,
        best_cluster: Optional[ClusterDetection],
    ) -> Optional[TrackOutput]:
        """Pick track closest to detection centroid; no NAV/prior fallback."""
        if best_cluster is None or not self.tracks:
            return None
        outputs = []
        for t in self.tracks:
            ekf = t.x[0:3, 0].copy()
            coasted = t.missed > 0
            if coasted or t.det_hist.shape[0] < 2:
                poly = ekf.copy()
            else:
                poly = poly_fit_position(t.det_hist, self.cfg.poly_order, self.cfg.fit_win)
            outputs.append(
                TrackOutput(
                    t.track_id,
                    ekf[:2].copy(),
                    poly[:2],
                    True,
                    coasted=coasted,
                    missed=int(t.missed),
                )
            )
        c_xy = best_cluster.centroid_xy
        return min(outputs, key=lambda o: float(np.linalg.norm(o.ekf_xy - c_xy)))
