"""Align ROS lidar timestamps to navigation time axis."""

from __future__ import annotations

import numpy as np


def ros_to_nav_time(ros_stamps_sec: np.ndarray, time_offset_sec: float) -> np.ndarray:
    """nav_time = ros_stamp_seconds + time_offset_sec (typically Unix epoch for AVP)."""
    return np.asarray(ros_stamps_sec, dtype=np.float64) + float(time_offset_sec)


def clamp_to_nav_range(t_query: np.ndarray, t_nav_min: float, t_nav_max: float) -> np.ndarray:
    """Clamp query times into [t_nav_min, t_nav_max] for stable interp (edges constant)."""
    return np.clip(t_query, t_nav_min, t_nav_max)
