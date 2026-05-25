"""YAML cooperative_link section parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LinkConfig:
    range_tol_m: float = 3.0
    bearing_tol_deg: float = 15.0
    z_tol_m: float = 1.5
    motion_thresh_m: float = 0.15
    min_cluster_points: int = 8
    lock_frames: int = 5
    score_thresh: float = 0.6
    assoc_max_jump_m: float = 2.0
    coast_max_frames: int = 10
    reacquire_miss_frames: int = 15
    cluster_eps_m: float = 0.35
    w_r: float = 0.4
    w_b: float = 0.3
    w_m: float = 0.3
    sigma_r_m: float = 2.0
    sigma_bearing_deg: float = 8.0
    nn_max_dist_m: float = 0.25


def load_link_config(cfg: Dict[str, Any]) -> LinkConfig:
    raw = cfg.get("cooperative_link") or {}
    weights = raw.get("weights") or {}
    sigmas = raw.get("sigmas") or {}
    wlh = cfg.get("target_size_wlh") or [0.5, 0.7, 0.65]
    eps_default = max(float(wlh[0]), float(wlh[1])) * 0.5
    return LinkConfig(
        range_tol_m=float(raw.get("range_tol_m", 3.0)),
        bearing_tol_deg=float(raw.get("bearing_tol_deg", 15.0)),
        z_tol_m=float(raw.get("z_tol_m", 1.5)),
        motion_thresh_m=float(raw.get("motion_thresh_m", 0.15)),
        min_cluster_points=int(raw.get("min_cluster_points", 8)),
        lock_frames=int(raw.get("lock_frames", 5)),
        score_thresh=float(raw.get("score_thresh", 0.6)),
        assoc_max_jump_m=float(raw.get("assoc_max_jump_m", 2.0)),
        coast_max_frames=int(raw.get("coast_max_frames", 10)),
        reacquire_miss_frames=int(raw.get("reacquire_miss_frames", 15)),
        cluster_eps_m=float(raw.get("cluster_eps_m", eps_default)),
        w_r=float(weights.get("w_r", 0.4)),
        w_b=float(weights.get("w_b", 0.3)),
        w_m=float(weights.get("w_m", 0.3)),
        sigma_r_m=float(sigmas.get("sigma_r_m", 2.0)),
        sigma_bearing_deg=float(sigmas.get("sigma_bearing_deg", 8.0)),
        nn_max_dist_m=float(raw.get("nn_max_dist_m", 0.25)),
    )
