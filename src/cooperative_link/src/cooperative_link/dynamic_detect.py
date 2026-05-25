"""Prior-gated clustering + motion residual fusion scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from cooperative_link.config import LinkConfig
from cooperative_link.prior import InstantPrior


def wrap_rad(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _bearing_xy(xy: np.ndarray) -> np.ndarray:
    return np.arctan2(xy[:, 1], xy[:, 0])


def prior_gate_mask(
    pts: np.ndarray,
    prior: InstantPrior,
    link_cfg: LinkConfig,
    gt_center_z_m: float,
) -> np.ndarray:
    """Boolean mask: points inside range annulus + bearing sector (+ optional z)."""
    xyz = pts[:, :3].astype(np.float64)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    bearing = _bearing_xy(xyz)
    dr = np.abs(r - prior.r_prior_m)
    db = np.abs(wrap_rad(bearing - prior.bearing_prior_rad))
    tol_b = np.deg2rad(link_cfg.bearing_tol_deg)
    m = (dr <= link_cfg.range_tol_m) & (db <= tol_b)
    if link_cfg.z_tol_m > 0:
        m &= np.abs(xyz[:, 2] - gt_center_z_m) <= link_cfg.z_tol_m
    return m


def _cluster_dbscan(xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    try:
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(xy).labels_
        return labels.astype(np.int32)
    except Exception:
        return _cluster_grid(xy, eps, min_samples)


def _cluster_grid(xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Grid connected-components fallback when sklearn is unavailable."""
    n = xy.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    cell = max(eps, 1e-3)
    keys = np.floor(xy / cell).astype(np.int64)
    buckets: dict = {}
    for i, k in enumerate(map(tuple, keys)):
        buckets.setdefault(k, []).append(i)
    label_id = 0
    for indices in buckets.values():
        if len(indices) < min_samples:
            continue
        for idx in indices:
            labels[idx] = label_id
        label_id += 1
    return labels


@dataclass
class ClusterDetection:
    centroid_xy: np.ndarray
    r_det_m: float
    bearing_det_rad: float
    n_points: int
    motion_score: float
    match_score: float
    label: int


def _match_score(
    r_det: float,
    bearing_det: float,
    motion_score: float,
    prior: InstantPrior,
    link_cfg: LinkConfig,
) -> float:
    dr = abs(r_det - prior.r_prior_m)
    db_deg = abs(np.rad2deg(wrap_rad(bearing_det - prior.bearing_prior_rad)))
    s_r = np.exp(-dr / max(link_cfg.sigma_r_m, 1e-6))
    s_b = np.exp(-db_deg / max(link_cfg.sigma_bearing_deg, 1e-6))
    w_sum = link_cfg.w_r + link_cfg.w_b + link_cfg.w_m
    if w_sum < 1e-9:
        return float(s_r * s_b * motion_score)
    return float(
        (link_cfg.w_r * s_r + link_cfg.w_b * s_b + link_cfg.w_m * motion_score) / w_sum
    )


def detect_dynamic_clusters(
    pts: np.ndarray,
    prior: InstantPrior,
    link_cfg: LinkConfig,
    motion_residual: np.ndarray,
    gt_center_z_m: float = 0.0,
) -> Tuple[np.ndarray, List[ClusterDetection]]:
    """
    Gate -> cluster -> motion score -> fused match score per cluster.

    Returns (gate_mask, sorted clusters best-first).
    """
    gate = prior_gate_mask(pts, prior, link_cfg, gt_center_z_m)
    idx = np.where(gate)[0]
    if idx.size == 0:
        return gate, []

    gated = pts[idx]
    xy = gated[:, :2].astype(np.float64)
    labels = _cluster_dbscan(xy, link_cfg.cluster_eps_m, link_cfg.min_cluster_points)

    clusters: List[ClusterDetection] = []
    if motion_residual.shape[0] != pts.shape[0]:
        motion_residual = np.zeros(pts.shape[0], dtype=np.float64)
    res_gated = motion_residual[gate]

    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        cm = labels == lab
        if int(cm.sum()) < link_cfg.min_cluster_points:
            continue
        c_xy = xy[cm].mean(axis=0)
        r_det = float(np.hypot(c_xy[0], c_xy[1]))
        b_det = float(np.arctan2(c_xy[1], c_xy[0]))
        res = res_gated[cm]
        motion_stat = float(np.median(res)) if res.size else 0.0
        motion_score = float(np.clip(motion_stat / max(link_cfg.motion_thresh_m, 1e-6), 0.0, 1.0))
        ms = _match_score(r_det, b_det, motion_score, prior, link_cfg)
        clusters.append(
            ClusterDetection(
                centroid_xy=c_xy,
                r_det_m=r_det,
                bearing_det_rad=b_det,
                n_points=int(cm.sum()),
                motion_score=motion_score,
                match_score=ms,
                label=int(lab),
            )
        )

    clusters.sort(key=lambda c: c.match_score, reverse=True)
    return gate, clusters
