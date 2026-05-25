"""Build per-frame dynamic target records for publishing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from cooperative_link.dynamic_detect import ClusterDetection
from cooperative_link.filter_config import DynamicConfig
from cooperative_link.partner_match import PartnerMatchResult, cluster_for_track
from cooperative_link.track_filter import TrackOutput


@dataclass
class DynamicTargetInfo:
    track_id: int
    cluster_label: int
    x: float
    y: float
    z: float
    range_m: float
    bearing_rad: float
    match_score: float
    motion_score: float
    is_link_partner: bool
    traj_align_cost: float
    traj_match_valid: bool


def _track_id_for_cluster(
    cluster: ClusterDetection,
    track_outputs: List[TrackOutput],
) -> int:
    if not track_outputs:
        return -1
    best_id = -1
    best_d = float("inf")
    for trk in track_outputs:
        d = float(np.linalg.norm(trk.ekf_xy - cluster.centroid_xy))
        if d < best_d:
            best_d = d
            best_id = trk.track_id
    return best_id


def build_dynamic_targets(
    clusters: List[ClusterDetection],
    track_outputs: List[TrackOutput],
    dynamic_cfg: DynamicConfig,
    partner_match: PartnerMatchResult,
    gt_z: float,
) -> List[DynamicTargetInfo]:
    if not dynamic_cfg.enabled:
        return []

    targets: List[DynamicTargetInfo] = []
    partner_label = (
        partner_match.partner_cluster.label
        if partner_match.partner_cluster is not None
        else -1
    )

    for c in clusters:
        if c.match_score < dynamic_cfg.publish_score_thresh:
            continue
        if c.motion_score < dynamic_cfg.min_motion_score:
            continue

        is_partner = (
            partner_match.traj_valid
            and c.label == partner_label
            and partner_match.partner_cluster is not None
        )
        align_cost = partner_match.align_cost if is_partner else float("inf")

        targets.append(
            DynamicTargetInfo(
                track_id=_track_id_for_cluster(c, track_outputs),
                cluster_label=int(c.label),
                x=float(c.centroid_xy[0]),
                y=float(c.centroid_xy[1]),
                z=float(gt_z),
                range_m=float(c.r_det_m),
                bearing_rad=float(c.bearing_det_rad),
                match_score=float(c.match_score),
                motion_score=float(c.motion_score),
                is_link_partner=is_partner,
                traj_align_cost=align_cost,
                traj_match_valid=is_partner,
            )
        )

    covered_ids = {t.track_id for t in targets if t.track_id >= 0}
    partner_tid = int(partner_match.partner_track_id)
    for trk in track_outputs:
        if trk.track_id < 0 or trk.track_id in covered_ids:
            continue
        if not trk.coasted:
            continue
        x, y = float(trk.ekf_xy[0]), float(trk.ekf_xy[1])
        r = float(np.hypot(x, y))
        b = float(np.arctan2(y, x))
        is_partner = partner_tid >= 0 and trk.track_id == partner_tid
        targets.append(
            DynamicTargetInfo(
                track_id=int(trk.track_id),
                cluster_label=-1,
                x=x,
                y=y,
                z=float(gt_z),
                range_m=r,
                bearing_rad=b,
                match_score=0.0,
                motion_score=0.0,
                is_link_partner=is_partner,
                traj_align_cost=partner_match.align_cost if is_partner else float("inf"),
                traj_match_valid=is_partner and partner_match.traj_valid,
            )
        )

    return targets
