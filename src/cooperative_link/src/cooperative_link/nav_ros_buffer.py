"""Ring buffers for host/target NavSatFix + yaw topics; interpolate at lidar time."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


def stamp_to_sec(stamp: object) -> float:
    """ROS rospy.Time or genpy Time -> seconds."""
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def _interp_yaw_deg(t_query: float, times: np.ndarray, yaws: np.ndarray) -> float:
    if times.size == 0:
        raise ValueError("empty yaw series")
    if times.size == 1:
        return float(yaws[0])
    rad = np.deg2rad(yaws)
    rad_u = np.unwrap(rad)
    return float(np.rad2deg(np.interp(t_query, times, rad_u)))


@dataclass
class _RoleBuffer:
    nav_buffer_sec: float
    time_offset_sec: float
    max_query_slop_sec: float = 0.15
    _ll: Deque[Tuple[float, float, float]] = field(default_factory=deque)
    _yaw: Deque[Tuple[float, float]] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _nav_t(self, stamp_sec: float) -> float:
        return float(stamp_sec) + self.time_offset_sec

    def push_navsat(self, stamp_sec: float, lat_deg: float, lon_deg: float) -> None:
        t = self._nav_t(stamp_sec)
        with self._lock:
            self._ll.append((t, float(lat_deg), float(lon_deg)))
            self._trim(self._ll)

    def push_yaw(self, stamp_sec: float, yaw_deg: float) -> None:
        t = self._nav_t(stamp_sec)
        with self._lock:
            self._yaw.append((t, float(yaw_deg)))
            self._trim(self._yaw)

    def push_pose(
        self, stamp_sec: float, lat_deg: float, lon_deg: float, yaw_deg: float
    ) -> None:
        """Atomically append lat/lon and yaw at the same nav_time."""
        t = self._nav_t(stamp_sec)
        with self._lock:
            self._ll.append((t, float(lat_deg), float(lon_deg)))
            self._yaw.append((t, float(yaw_deg)))
            self._trim(self._ll)
            self._trim(self._yaw)

    def _trim(self, dq: Deque) -> None:
        if not dq:
            return
        t_max = dq[-1][0]
        t_min = t_max - self.nav_buffer_sec
        while dq and dq[0][0] < t_min:
            dq.popleft()

    def _snapshots(self) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
        with self._lock:
            return list(self._ll), list(self._yaw)

    def buffer_span(self) -> Dict[str, Any]:
        ll_snap, yaw_snap = self._snapshots()
        out: Dict[str, Any] = {
            "ll_n": len(ll_snap),
            "yaw_n": len(yaw_snap),
            "ll_range": None,
            "yaw_range": None,
        }
        if ll_snap:
            out["ll_range"] = (ll_snap[0][0], ll_snap[-1][0])
        if yaw_snap:
            out["yaw_range"] = (yaw_snap[0][0], yaw_snap[-1][0])
        return out

    def query_miss_reason(self, t_nav: float) -> str:
        ll_snap, yaw_snap = self._snapshots()
        if not ll_snap or not yaw_snap:
            return "empty"
        t_ll = np.array([x[0] for x in ll_snap], dtype=np.float64)
        t_y = np.array([x[0] for x in yaw_snap], dtype=np.float64)
        t_min = float(max(t_ll.min(), t_y.min()))
        t_max = float(min(t_ll.max(), t_y.max()))
        if t_y.max() + 1e-6 < t_ll.max() - 0.05:
            return "desync"
        tq = float(t_nav)
        slop = float(self.max_query_slop_sec)
        if tq < t_min - slop or tq > t_max + slop:
            return "out_of_range"
        if tq < t_min or tq > t_max:
            return "clamped"
        return "ok"

    def query_pose(self, t_nav: float) -> Optional[Dict[str, float]]:
        ll_snap, yaw_snap = self._snapshots()
        if len(ll_snap) < 1 or len(yaw_snap) < 1:
            return None
        t_ll = np.array([x[0] for x in ll_snap], dtype=np.float64)
        lat = np.array([x[1] for x in ll_snap], dtype=np.float64)
        lon = np.array([x[2] for x in ll_snap], dtype=np.float64)
        t_y = np.array([x[0] for x in yaw_snap], dtype=np.float64)
        yaws = np.array([x[1] for x in yaw_snap], dtype=np.float64)
        tq_raw = float(t_nav)
        t_min = float(max(t_ll.min(), t_y.min()))
        t_max = float(min(t_ll.max(), t_y.max()))
        slop = float(self.max_query_slop_sec)
        if tq_raw < t_min - slop or tq_raw > t_max + slop:
            return None
        tq = float(np.clip(tq_raw, t_min, t_max))
        lat_i = float(np.interp(tq, t_ll, lat))
        lon_i = float(np.interp(tq, t_ll, lon))
        yaw_i = _interp_yaw_deg(tq, t_y, yaws)
        return {"lat_deg": lat_i, "lon_deg": lon_i, "yaw_deg": yaw_i}


@dataclass
class NavRosBuffer:
    """Host and target NAV ring buffers keyed by nav_time (stamp + offset)."""

    nav_buffer_sec: float = 60.0
    time_offset_sec: float = 0.0
    max_query_slop_sec: float = 0.15
    host: _RoleBuffer = field(init=False)
    target: _RoleBuffer = field(init=False)

    def __post_init__(self) -> None:
        kw = {
            "nav_buffer_sec": self.nav_buffer_sec,
            "time_offset_sec": self.time_offset_sec,
            "max_query_slop_sec": self.max_query_slop_sec,
        }
        self.host = _RoleBuffer(**kw)
        self.target = _RoleBuffer(**kw)

    def query_both(self, t_nav: float) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        return self.host.query_pose(t_nav), self.target.query_pose(t_nav)


@dataclass
class MultiPartnerNavBuffer:
    """Host + per-partner NAV ring buffers."""

    nav_buffer_sec: float = 60.0
    time_offset_sec: float = 0.0
    max_query_slop_sec: float = 0.15
    partner_ids: List[int] = field(default_factory=list)
    host: _RoleBuffer = field(init=False)
    partners: Dict[int, _RoleBuffer] = field(init=False)

    def __post_init__(self) -> None:
        kw = {
            "nav_buffer_sec": self.nav_buffer_sec,
            "time_offset_sec": self.time_offset_sec,
            "max_query_slop_sec": self.max_query_slop_sec,
        }
        self.host = _RoleBuffer(**kw)
        self.partners = {pid: _RoleBuffer(**kw) for pid in self.partner_ids}

    def query_host(self, t_nav: float) -> Optional[Dict[str, float]]:
        return self.host.query_pose(t_nav)

    def query_partner(self, partner_id: int, t_nav: float) -> Optional[Dict[str, float]]:
        buf = self.partners.get(partner_id)
        if buf is None:
            return None
        return buf.query_pose(t_nav)

    def query_all_partners(
        self, t_nav: float
    ) -> Dict[int, Optional[Dict[str, float]]]:
        return {pid: buf.query_pose(t_nav) for pid, buf in self.partners.items()}
