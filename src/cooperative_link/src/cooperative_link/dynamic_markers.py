"""RViz dynamic marker selection: stable KF track ids, capped count, partner priority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from cooperative_link.dynamic_targets import DynamicTargetInfo
from cooperative_link.filter_config import DynamicConfig
from cooperative_link.track_filter import TrackOutput

MARKER_ID_DYNAMIC_BASE = 100
MARKER_ID_DYNAMIC_MAX = 199


@dataclass
class RvizDynamicTrack:
    track_id: int
    x: float
    y: float
    z: float
    is_link_partner: bool
    match_score: float
    coasted: bool = False


def dynamic_marker_id(track_id: int) -> int:
    """Stable RViz marker id for a KF track."""
    mid = MARKER_ID_DYNAMIC_BASE + int(track_id)
    return min(max(mid, MARKER_ID_DYNAMIC_BASE), MARKER_ID_DYNAMIC_MAX)


def marker_ids_from_tracks(tracks: List[RvizDynamicTrack]) -> Set[int]:
    return {dynamic_marker_id(t.track_id) for t in tracks}


def _track_meta_from_dynamic_targets(
    dynamic_targets: List[DynamicTargetInfo],
) -> Dict[int, DynamicTargetInfo]:
    by_id: Dict[int, DynamicTargetInfo] = {}
    for dt in dynamic_targets:
        if dt.track_id < 0:
            continue
        prev = by_id.get(dt.track_id)
        if prev is None or dt.match_score > prev.match_score:
            by_id[dt.track_id] = dt
    return by_id


def select_tracks_for_rviz(
    track_outputs: List[TrackOutput],
    dynamic_targets: List[DynamicTargetInfo],
    partner_track_id: int,
    dynamic_cfg: DynamicConfig,
) -> List[RvizDynamicTrack]:
    """Pick up to max_dynamic_markers KF tracks for RViz; partner always included."""
    if not dynamic_cfg.rviz_tracks_only:
        rows: List[RvizDynamicTrack] = []
        for dt in dynamic_targets:
            if dt.track_id < 0 and dynamic_cfg.rviz_tracks_only:
                continue
            rows.append(
                RvizDynamicTrack(
                    track_id=max(dt.track_id, 0),
                    x=dt.x,
                    y=dt.y,
                    z=dt.z,
                    is_link_partner=dt.is_link_partner,
                    match_score=dt.match_score,
                )
            )
        rows.sort(key=lambda r: r.match_score, reverse=True)
        return rows[: max(dynamic_cfg.max_dynamic_markers, 0)]

    meta = _track_meta_from_dynamic_targets(dynamic_targets)
    candidates: List[RvizDynamicTrack] = []
    partner_row: Optional[RvizDynamicTrack] = None

    for trk in track_outputs:
        tid = int(trk.track_id)
        if tid < 0:
            continue
        dt = meta.get(tid)
        match_score = float(dt.match_score) if dt is not None else 0.0
        motion_score = float(dt.motion_score) if dt is not None else 1.0
        if motion_score < dynamic_cfg.min_motion_score:
            continue
        is_partner = partner_track_id >= 0 and tid == partner_track_id
        row = RvizDynamicTrack(
            track_id=tid,
            x=float(trk.ekf_xy[0]),
            y=float(trk.ekf_xy[1]),
            z=float(dt.z) if dt is not None else 0.0,
            is_link_partner=is_partner,
            match_score=match_score,
            coasted=bool(trk.coasted),
        )
        if is_partner:
            partner_row = row
        else:
            candidates.append(row)

    candidates.sort(key=lambda r: r.match_score, reverse=True)
    max_n = max(int(dynamic_cfg.max_dynamic_markers), 0)
    out: List[RvizDynamicTrack] = []
    if partner_row is not None:
        out.append(partner_row)
        max_n = max(max_n - 1, 0)
    out.extend(candidates[:max_n])
    return out


def stale_marker_ids(previous: Set[int], current: Set[int]) -> Set[int]:
    return set(previous) - set(current)
