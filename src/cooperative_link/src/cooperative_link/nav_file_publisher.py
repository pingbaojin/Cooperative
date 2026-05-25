#!/usr/bin/env python3
"""
Publish host/target NAV from AVP/KF-GINS files as NavSatFix + Float64 yaw topics.

Use with rosbag play --clock when the bag has lidar but no NAV topics.

Usage:
  source /opt/ros/noetic/setup.bash
  rosparam set use_sim_time true
  rosbag play your.bag --clock &
  python3 -m cooperative_link.nav_file_publisher --config config/default.yaml
  roslaunch cooperative_link play_bag_with_nav.launch
"""

from __future__ import annotations


def _default_config_path() -> Path:
    try:
        import rospkg
        return Path(rospkg.RosPack().get_path("cooperative_link")) / "config" / "default.yaml"
    except Exception:
        return Path(__file__).resolve().parents[2] / "config" / "default.yaml"

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64, Header

from cooperative_link.nav_pose import NavSeries, load_nav, load_nav_pair_from_cfg, unwrap_yaw_interp
from cooperative_link.partner_config import load_multi_link_config
from cooperative_link.ros_config import load_ros_link_config


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_nav_from_cfg(cfg: Dict[str, Any], path: str) -> NavSeries:
    nav_format = cfg.get("nav_format", "auto")
    ref_alt = cfg.get("nav_ref_alt_m")
    nav_ref_alt_m = float(ref_alt) if ref_alt is not None else None
    return load_nav(path, nav_format=nav_format, nav_ref_alt_m=nav_ref_alt_m)


class NavFilePublisher:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.ros_cfg = load_ros_link_config(cfg)
        self.multi_cfg = load_multi_link_config(cfg)
        self.time_offset = float(cfg.get("time_offset_sec", 0.0))

        if self.multi_cfg.enabled:
            self._init_multi()
        else:
            self._init_legacy()

        self._skip_warned = False
        rospy.loginfo(
            "nav_file_publisher AVP nav_time range [%.3f, %.3f] (multi=%s)",
            self.t_min,
            self.t_max,
            self.multi_cfg.enabled,
        )

        rate_hz = float(rospy.get_param("~rate", 50.0))
        self._timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self._on_timer)

    def _init_legacy(self) -> None:
        self.nav_h, self.nav_t = load_nav_pair_from_cfg(self.cfg)
        self.t_min = max(self.nav_h.t_gps.min(), self.nav_t.t_gps.min())
        self.t_max = min(self.nav_h.t_gps.max(), self.nav_t.t_gps.max())
        self._roles: List[Tuple[NavSeries, Any, Any]] = [
            (
                self.nav_h,
                rospy.Publisher(self.ros_cfg.host_navsat_topic, NavSatFix, queue_size=10),
                rospy.Publisher(self.ros_cfg.host_yaw_topic, Float64, queue_size=10),
            ),
            (
                self.nav_t,
                rospy.Publisher(self.ros_cfg.target_navsat_topic, NavSatFix, queue_size=10),
                rospy.Publisher(self.ros_cfg.target_yaw_topic, Float64, queue_size=10),
            ),
        ]

    def _init_multi(self) -> None:
        self.nav_h = _load_nav_from_cfg(self.cfg, self.cfg["host_nav_path"])
        partner_navs: Dict[int, NavSeries] = {}
        t_mins = [self.nav_h.t_gps.min()]
        t_maxs = [self.nav_h.t_gps.max()]
        for spec in self.multi_cfg.partners:
            if not spec.nav_path:
                rospy.logwarn("partner %d: empty nav_path, skipped", spec.partner_id)
                continue
            nav = _load_nav_from_cfg(self.cfg, spec.nav_path)
            partner_navs[spec.partner_id] = nav
            t_mins.append(nav.t_gps.min())
            t_maxs.append(nav.t_gps.max())
        self.t_min = max(t_mins)
        self.t_max = min(t_maxs)
        self._roles = [
            (
                self.nav_h,
                rospy.Publisher(self.ros_cfg.host_navsat_topic, NavSatFix, queue_size=10),
                rospy.Publisher(self.ros_cfg.host_yaw_topic, Float64, queue_size=10),
            )
        ]
        for spec in self.multi_cfg.partners:
            nav = partner_navs.get(spec.partner_id)
            if nav is None:
                continue
            self._roles.append(
                (
                    nav,
                    rospy.Publisher(spec.navsat_topic, NavSatFix, queue_size=10),
                    rospy.Publisher(spec.yaw_topic, Float64, queue_size=10),
                )
            )

    def _make_header(self, stamp: rospy.Time) -> Header:
        h = Header()
        h.stamp = stamp
        h.frame_id = "nav"
        return h

    def _publish_role(
        self,
        nav: NavSeries,
        stamp: rospy.Time,
        pub_nav,
        pub_yaw,
    ) -> None:
        t_ros = stamp.to_sec()
        t_nav = t_ros + self.time_offset
        if t_nav < self.t_min or t_nav > self.t_max:
            if not self._skip_warned:
                self._skip_warned = True
                rospy.logwarn(
                    "sim nav_time %.3f outside AVP [%.3f, %.3f]; "
                    "NAV not published until /clock enters range "
                    "(cooperative_link may log missing host for lidar)",
                    t_nav,
                    self.t_min,
                    self.t_max,
                )
            return
        lat, lon, yaw = unwrap_yaw_interp(nav, np.array([t_nav]))
        hdr = self._make_header(stamp)

        fix = NavSatFix()
        fix.header = hdr
        fix.latitude = float(lat[0])
        fix.longitude = float(lon[0])
        fix.altitude = 0.0
        fix.status.status = 0
        pub_nav.publish(fix)

        ymsg = Float64()
        ymsg.data = float(yaw[0])
        pub_yaw.publish(ymsg)

    def _on_timer(self, _event) -> None:
        stamp = rospy.Time.now()
        for nav, pub_nav, pub_yaw in self._roles:
            self._publish_role(nav, stamp, pub_nav, pub_yaw)


def main() -> None:
    rospy.init_node("nav_file_to_ros_publisher")
    default_cfg = str(_default_config_path())
    config_path = rospy.get_param("~config", "")
    if not config_path:
        parser = argparse.ArgumentParser(description="Publish NAV files to ROS topics")
        parser.add_argument("--config", type=str, default="")
        args = parser.parse_args(rospy.myargv(sys.argv)[1:])
        config_path = args.config or default_cfg
    cfg = load_config(config_path)
    NavFilePublisher(cfg)
    rospy.loginfo(
        "nav_file_to_ros_publisher running (multi=%s, use_sim_time=%s)",
        load_multi_link_config(cfg).enabled,
        rospy.get_param("/use_sim_time", False),
    )
    rospy.spin()


if __name__ == "__main__":
    main()
