"""Build cooperative target registry fields for /cooperative_targets."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple


def build_cooperative_targets(
    host_id: int,
    link_target_id: int,
    partner_ids: Sequence[int],
    nav_valid_by_id: Dict[int, bool],
) -> Tuple[int, int, List[int], List[bool], int]:
    """Return host_id, count, sorted partner_ids, aligned nav_valid, link_target_id."""
    sorted_ids = sorted(int(pid) for pid in partner_ids)
    nav_valid = [bool(nav_valid_by_id.get(pid, False)) for pid in sorted_ids]
    count = len(sorted_ids)
    return int(host_id), count, sorted_ids, nav_valid, int(link_target_id)


def partner_ids_from_multi(partners: Sequence) -> List[int]:
    return [int(p.partner_id) for p in partners]


def nav_valid_from_poses(
    partner_ids: Sequence[int],
    poses: Dict[int, Optional[dict]],
) -> Dict[int, bool]:
    return {int(pid): poses.get(int(pid)) is not None for pid in partner_ids}
