"""Unit tests for nav_ros_buffer (no ROS required)."""

from __future__ import annotations

import threading

from cooperative_link.nav_ros_buffer import MultiPartnerNavBuffer, NavRosBuffer


def test_interp_pose():
    buf = NavRosBuffer(nav_buffer_sec=10.0, time_offset_sec=0.0)
    for t in [0.0, 1.0, 2.0]:
        buf.host.push_navsat(t, 30.0 + t * 1e-5, 120.0 + t * 1e-5)
        buf.host.push_yaw(t, 10.0 * t)
    p = buf.host.query_pose(1.0)
    assert p is not None
    assert abs(p["lat_deg"] - (30.0 + 1e-5)) < 1e-9
    assert abs(p["yaw_deg"] - 10.0) < 1e-6


def test_query_both():
    buf = NavRosBuffer(nav_buffer_sec=10.0, time_offset_sec=1.0)
    buf.host.push_navsat(0.0, 30.0, 120.0)
    buf.host.push_yaw(0.0, 0.0)
    buf.target.push_navsat(0.0, 30.0001, 120.0001)
    buf.target.push_yaw(0.0, 45.0)
    h, t = buf.query_both(1.0)
    assert h is not None and t is not None


def test_multi_partner_interp():
    buf = MultiPartnerNavBuffer(
        nav_buffer_sec=10.0, time_offset_sec=0.0, partner_ids=[1, 2]
    )
    for t in [0.0, 1.0, 2.0]:
        buf.host.push_navsat(t, 30.0 + t * 1e-5, 120.0)
        buf.host.push_yaw(t, 0.0)
        buf.partners[1].push_navsat(t, 30.0001, 120.0 + t * 1e-5)
        buf.partners[1].push_yaw(t, 10.0 * t)
        buf.partners[2].push_navsat(t, 30.0002, 120.0002)
        buf.partners[2].push_yaw(t, 20.0 * t)
    host = buf.query_host(1.0)
    p1 = buf.query_partner(1, 1.0)
    p2 = buf.query_partner(2, 1.0)
    assert host is not None and p1 is not None and p2 is not None
    assert abs(p1["yaw_deg"] - 10.0) < 1e-6
    assert abs(p2["yaw_deg"] - 20.0) < 1e-6
    all_p = buf.query_all_partners(1.0)
    assert set(all_p.keys()) == {1, 2}


def test_concurrent_push_query():
    """Regression: lidar query vs NAV push must not mutate deque during iteration."""
    buf = NavRosBuffer(nav_buffer_sec=30.0, time_offset_sec=0.0)
    role = buf.host
    for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
        role.push_navsat(t, 30.0, 120.0)
        role.push_yaw(t, 10.0 * t)
    stop = threading.Event()
    errors: list = []

    def pusher() -> None:
        try:
            for i in range(400):
                if stop.is_set():
                    break
                t = 2.0 + (i % 15) * 0.1
                role.push_navsat(t, 30.0 + t * 1e-6, 120.0 + t * 1e-6)
                role.push_yaw(t, 5.0 + t * 0.1)
        except Exception as exc:
            errors.append(exc)

    def querier() -> None:
        try:
            for _ in range(500):
                role.query_pose(1.0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=pusher), threading.Thread(target=querier)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=5.0)
    stop.set()
    assert not errors, errors
    p = role.query_pose(1.0)
    assert p is not None
    assert abs(p["yaw_deg"] - 10.0) < 1e-5


def test_push_pose_atomic():
    buf = NavRosBuffer(nav_buffer_sec=10.0, time_offset_sec=0.0)
    role = buf.host
    for t in [0.0, 1.0, 2.0]:
        role.push_pose(t, 30.0 + t * 1e-5, 120.0 + t * 1e-5, 10.0 * t)
    span = role.buffer_span()
    assert span["ll_n"] == span["yaw_n"] == 3
    assert span["ll_range"] == span["yaw_range"]


def test_query_clamp_inside_slop():
    buf = NavRosBuffer(nav_buffer_sec=10.0, max_query_slop_sec=0.1)
    role = buf.host
    role.push_pose(0.0, 30.0, 120.0, 0.0)
    role.push_pose(2.0, 30.0001, 120.0001, 20.0)
    p = role.query_pose(2.02)
    assert p is not None
    assert role.query_miss_reason(2.02) == "clamped"


def test_query_reject_beyond_slop():
    buf = NavRosBuffer(nav_buffer_sec=10.0, max_query_slop_sec=0.1)
    role = buf.host
    role.push_pose(0.0, 30.0, 120.0, 0.0)
    role.push_pose(2.0, 30.0001, 120.0001, 20.0)
    assert role.query_pose(2.25) is None
    assert role.query_miss_reason(2.25) == "out_of_range"


def test_default_slop_covers_104ms_ahead():
    """Default 0.15s slop covers lidar slightly ahead of last NAV sample."""
    buf = NavRosBuffer(nav_buffer_sec=10.0)
    role = buf.host
    role.push_pose(0.0, 30.0, 120.0, 0.0)
    role.push_pose(2.0, 30.0001, 120.0001, 20.0)
    assert role.query_pose(2.104) is not None
    assert role.query_miss_reason(2.104) == "clamped"


if __name__ == "__main__":
    test_interp_pose()
    test_query_both()
    test_multi_partner_interp()
    test_concurrent_push_query()
    test_push_pose_atomic()
    test_query_clamp_inside_slop()
    test_query_reject_beyond_slop()
    test_default_slop_covers_104ms_ahead()
    print("nav_ros_buffer tests OK")
