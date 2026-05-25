"""Lidar message decimation helpers (from cooperative_link.build_infos)."""

from __future__ import annotations

import warnings
from typing import Any, List, Tuple


def filter_decimated_lidar_messages(
    messages: List[Tuple[float, Any]],
    skip_start_sec: float,
    lidar_output_hz: float,
) -> List[Tuple[float, Any]]:
    """
    After lidar_decimate collection: drop an initial time window, then optional Hz limit.

    Order: skip_start_sec relative to **first decimated** ros_t, then greedy keep with
    spacing >= 1/lidar_output_hz when lidar_output_hz > 0.
    """
    if not messages:
        return messages
    skip = float(skip_start_sec)
    if skip < 0:
        warnings.warn("skip_start_sec < 0, treating as 0", stacklevel=2)
        skip = 0.0
    ros_t0 = float(messages[0][0])
    out = [(ts, m) for ts, m in messages if float(ts) >= ros_t0 + skip]

    hz = float(lidar_output_hz)
    if hz < 0:
        warnings.warn("lidar_output_hz < 0, treating as 0 (no rate filter)", stacklevel=2)
        hz = 0.0
    if hz <= 0 or not out:
        return out

    period = 1.0 / hz
    eps = 1e-6
    kept: List[Tuple[float, Any]] = []
    last_kept = -1e30
    for ts, m in out:
        tsf = float(ts)
        if tsf >= last_kept + period - eps:
            kept.append((ts, m))
            last_kept = tsf
    return kept


