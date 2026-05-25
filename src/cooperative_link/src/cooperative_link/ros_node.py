#!/usr/bin/env python3
"""
ROS1 node: cooperative link on live lidar + NavSatFix/yaw topics.

Publishes:
  - relative_pose_nav: dual-NAV reference in lidar frame (always when NAV ok)
  - relative_pose_det: radar cluster centroid (when dynamic detection matches)
  - relative_pose: fused output from link FSM (legacy / control)
  - cooperative_uwb/range: UWB-style ENU range from dual NAV (nav_sim)
  - markers: MarkerArray for RViz (green=NAV, magenta=det, orange=fused)

Usage:
  source /opt/ros/noetic/setup.bash
  cd /docker_ws/ubuntu20_04/CenterPoint
  python3 data_generate/cooperative_link_ros_node.py _config:=data_generate/example_config.yaml
"""

from __future__ import annotations


def _default_config_path() -> Path:
    try:
        import rospkg
        return Path(rospkg.RosPack().get_path("cooperative_link")) / "config" / "default.yaml"
    except Exception:
        return Path(__file__).resolve().parents[2] / "config" / "default.yaml"

import math
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import yaml

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import NavSatFix, PointCloud2
from std_msgs.msg import Float32MultiArray, Float64, String
from visualization_msgs.msg import Marker, MarkerArray

from cooperative_link.associate import LinkState
from cooperative_link.nav_ros_buffer import (
    MultiPartnerNavBuffer,
    NavRosBuffer,
    stamp_to_sec,
)
from cooperative_link.partner_config import load_multi_link_config
from cooperative_link.targets_registry import (
    build_cooperative_targets,
    nav_valid_from_poses,
    partner_ids_from_multi,
)
from cooperative_link.dynamic_markers import (
    dynamic_marker_id,
    marker_ids_from_tracks,
    select_tracks_for_rviz,
    stale_marker_ids,
)
from cooperative_link.pipeline import CooperativeLinkPipeline
from cooperative_link.prior import compute_instant_prior
from cooperative_link.ros_config import load_ros_link_config
from cooperative_link.topic_names import partner_topic
from cooperative_link.export_pointcloud_bin import read_cloud_numpy
from cooperative_link.uwb_config import UwbConfig, load_uwb_config
from cooperative_link.uwb_range import (
    apply_range_noise,
    build_uwb_range_readings,
    enu_range_from_prior,
)

MARKER_ID_NAV = 0
MARKER_ID_DET = 1
MARKER_ID_FUSED = 2
MARKER_ID_LINE_NAV = 3
MARKER_ID_DYNAMIC_BASE = 100


def _yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw_rad * 0.5)
    q.w = math.cos(yaw_rad * 0.5)
    return q


def _output_valid(state: LinkState, has_cluster: bool) -> bool:
    if state == LinkState.LOCKED:
        return True
    if state == LinkState.LOCKING and has_cluster:
        return True
    return False


def _make_pose(
    stamp: Any,
    frame_id: str,
    x: float,
    y: float,
    z: float,
    yaw_rad: float,
) -> PoseStamped:
    pose = PoseStamped()
    pose.header.stamp = stamp
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    pose.pose.orientation = _yaw_to_quaternion(yaw_rad)
    return pose


def _make_sphere_marker(
    stamp: Any,
    frame_id: str,
    marker_id: int,
    ns: str,
    x: float,
    y: float,
    z: float,
    diameter: float,
    rgba: Tuple[float, float, float, float],
    action: int = Marker.ADD,
    lifetime_sec: float = 0.0,
) -> Marker:
    m = Marker()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = action
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = z
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = diameter
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    if lifetime_sec > 0.0:
        m.lifetime = rospy.Duration(lifetime_sec)
    return m


def _make_line_to_origin(
    stamp: Any,
    frame_id: str,
    marker_id: int,
    ns: str,
    x: float,
    y: float,
    z: float,
    rgba: Tuple[float, float, float, float],
) -> Marker:
    line = Marker()
    line.header.stamp = stamp
    line.header.frame_id = frame_id
    line.ns = ns
    line.id = marker_id
    line.type = Marker.LINE_STRIP
    line.action = Marker.ADD
    line.scale.x = 0.05
    line.points = []
    from geometry_msgs.msg import Point

    line.points.append(Point(x=0.0, y=0.0, z=z))
    line.points.append(Point(x=x, y=y, z=z))
    line.color.r, line.color.g, line.color.b, line.color.a = rgba
    return line


class CooperativeLinkRosNode:
    def __init__(self) -> None:
        config_path = rospy.get_param("~config", "")
        if not config_path:
            config_path = str(_default_config_path())
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.ros_cfg = load_ros_link_config(self.cfg)
        self.multi_cfg = load_multi_link_config(self.cfg)
        self.uwb_cfg = load_uwb_config(self.cfg)
        self.time_offset = float(self.cfg.get("time_offset_sec", 0.0))
        self.gt_z = float(self.cfg.get("gt_center_z_m", 0.0))

        self.pipe = CooperativeLinkPipeline.from_yaml_geometry_only(self.cfg)
        self.pipe.attach_post_processor(self.cfg)
        self.nav_buf: Optional[NavRosBuffer] = None
        self.nav_multi: Optional[MultiPartnerNavBuffer] = None
        self._link_target_id = int(self.uwb_cfg.to_agent_id)
        if self.multi_cfg.enabled:
            partner_ids = [p.partner_id for p in self.multi_cfg.partners]
            self._link_target_id = partner_ids[0]
            self._configured_partner_ids = partner_ids_from_multi(self.multi_cfg.partners)
            self.nav_multi = MultiPartnerNavBuffer(
                nav_buffer_sec=self.ros_cfg.nav_buffer_sec,
                time_offset_sec=self.time_offset,
                max_query_slop_sec=self.ros_cfg.max_query_slop_sec,
                partner_ids=partner_ids,
            )
        else:
            self._configured_partner_ids = [int(self.uwb_cfg.to_agent_id)]
            self.nav_buf = NavRosBuffer(
                nav_buffer_sec=self.ros_cfg.nav_buffer_sec,
                time_offset_sec=self.time_offset,
                max_query_slop_sec=self.ros_cfg.max_query_slop_sec,
            )
        self._frame_idx = 0
        self._enu_set = False
        self._pending_host_navsat: Optional[Tuple[float, float, float]] = None
        self._pending_host_yaw: Optional[float] = None
        self._pending_target_navsat: Optional[Tuple[float, float, float]] = None
        self._pending_target_yaw: Optional[float] = None
        self._pending_partner_navsat: Dict[int, Tuple[float, float, float]] = {}
        self._pending_partner_yaw: Dict[int, float] = {}

        self.pub_pose_nav_by_id: Dict[int, Any] = {}
        self.pub_pose_det_by_id: Dict[int, Any] = {}
        self.pub_pose_fused_by_id: Dict[int, Any] = {}
        self.pub_polar_by_id: Dict[int, Any] = {}
        self.pub_state_by_id: Dict[int, Any] = {}
        self.pub_markers_by_id: Dict[int, Any] = {}
        self.pub_pose_filtered_by_id: Dict[int, Any] = {}
        self.pub_coord_by_id: Dict[int, Any] = {}
        self.pub_pose_nav: Optional[Any] = None
        self.pub_pose_det: Optional[Any] = None
        self.pub_pose_fused: Optional[Any] = None
        self.pub_polar: Optional[Any] = None
        self.pub_state: Optional[Any] = None
        self.pub_pose_filtered: Optional[Any] = None
        self.pub_coord_frame: Optional[Any] = None
        self.pub_dynamic_targets: Optional[Any] = None
        self.pub_dynamic_markers: Optional[Any] = None
        self._dynamic_marker_ids_active: Set[int] = set()
        self._CoordFrameEstimate = None
        self._init_output_publishers()
        self.pub_cooperative_targets = None
        try:
            from cooperative_link.msg import CooperativeTargets
            self._CooperativeTargets = CooperativeTargets
            self.pub_cooperative_targets = rospy.Publisher(
                self.ros_cfg.pub_cooperative_targets,
                CooperativeTargets,
                queue_size=1,
            )
        except ImportError:
            rospy.logerr(
                "cooperative_link CooperativeTargets msg not built; "
                "run catkin_make and source devel/setup.bash"
            )
        self.pub_marker = None
        if self.ros_cfg.publish_marker:
            self.pub_marker = rospy.Publisher(
                self.ros_cfg.pub_marker, Marker, queue_size=1
            )
        self.pub_markers: Optional[Any] = None
        self.pub_uwb = None
        self.pub_uwb_ranges = None
        self.pub_uwb_per_partner: Dict[int, Any] = {}
        if self.uwb_cfg.enabled:
            try:
                from cooperative_link.msg import UwbRange, UwbRangeArray
                self._UwbRange = UwbRange
                self._UwbRangeArray = UwbRangeArray
                if self.multi_cfg.enabled:
                    self.pub_uwb_ranges = rospy.Publisher(
                        self.uwb_cfg.pub_uwb_ranges, UwbRangeArray, queue_size=1
                    )
                    if self.uwb_cfg.publish_per_partner_topics:
                        for spec in self.multi_cfg.partners:
                            topic = self.uwb_cfg.pub_uwb_range_per_partner.format(
                                id=spec.partner_id
                            )
                            self.pub_uwb_per_partner[spec.partner_id] = rospy.Publisher(
                                topic, UwbRange, queue_size=1
                            )
                if self.uwb_cfg.publish_legacy_single and not self.multi_cfg.enabled:
                    self.pub_uwb = rospy.Publisher(
                        self.uwb_cfg.pub_uwb_range, UwbRange, queue_size=1
                    )
            except ImportError:
                rospy.logerr(
                    "cooperative_link UwbRange msg not built; run catkin_make and source devel/setup.bash"
                )

        rospy.Subscriber(
            self.ros_cfg.host_navsat_topic,
            NavSatFix,
            self._host_navsat_cb,
            queue_size=50,
        )
        rospy.Subscriber(
            self.ros_cfg.host_yaw_topic,
            Float64,
            self._host_yaw_cb,
            queue_size=50,
        )
        if self.multi_cfg.enabled:
            assert self.nav_multi is not None
            for spec in self.multi_cfg.partners:
                rospy.Subscriber(
                    spec.navsat_topic,
                    NavSatFix,
                    self._partner_navsat_cb,
                    callback_args=spec.partner_id,
                    queue_size=50,
                )
                rospy.Subscriber(
                    spec.yaw_topic,
                    Float64,
                    self._partner_yaw_cb,
                    callback_args=spec.partner_id,
                    queue_size=50,
                )
        else:
            assert self.nav_buf is not None
            rospy.Subscriber(
                self.ros_cfg.target_navsat_topic,
                NavSatFix,
                self._target_navsat_cb,
                queue_size=50,
            )
            rospy.Subscriber(
                self.ros_cfg.target_yaw_topic,
                Float64,
                self._target_yaw_cb,
                queue_size=50,
            )
        rospy.Subscriber(
            self.ros_cfg.lidar_topic,
            PointCloud2,
            self._lidar_cb,
            queue_size=1,
            buff_size=2**24,
        )

        uwb_topics = self.uwb_cfg.pub_uwb_ranges if self.multi_cfg.enabled else self.uwb_cfg.pub_uwb_range
        nav_topic = (
            partner_topic(self.ros_cfg.pub_relative_pose_nav, self._link_target_id)
            if self.multi_cfg.enabled
            else self.ros_cfg.pub_relative_pose_nav
        )
        rospy.loginfo(
            "cooperative_link_node: nav_pose=%s det_pose=%s fused=%s uwb=%s "
            "targets=%s multi=%s link_id=%d",
            nav_topic,
            self.ros_cfg.pub_relative_pose_det,
            self.ros_cfg.pub_relative_pose,
            uwb_topics if self.uwb_cfg.enabled else "disabled",
            self.ros_cfg.pub_cooperative_targets,
            self.multi_cfg.enabled,
            self._link_target_id,
        )

    def _init_output_publishers(self) -> None:
        if self.multi_cfg.enabled:
            for pid in self._configured_partner_ids:
                self.pub_pose_nav_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_relative_pose_nav, pid),
                    PoseStamped,
                    queue_size=1,
                )
                self.pub_pose_det_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_relative_pose_det, pid),
                    PoseStamped,
                    queue_size=1,
                )
                self.pub_pose_fused_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_relative_pose, pid),
                    PoseStamped,
                    queue_size=1,
                )
                self.pub_polar_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_relative_polar, pid),
                    Float32MultiArray,
                    queue_size=1,
                )
                self.pub_state_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_link_state, pid),
                    String,
                    queue_size=1,
                )
                if self.ros_cfg.publish_markers:
                    self.pub_markers_by_id[pid] = rospy.Publisher(
                        partner_topic(self.ros_cfg.pub_markers, pid),
                        MarkerArray,
                        queue_size=1,
                    )
                self.pub_pose_filtered_by_id[pid] = rospy.Publisher(
                    partner_topic(self.ros_cfg.pub_relative_pose_filtered, pid),
                    PoseStamped,
                    queue_size=1,
                )
            self._init_coord_publishers()
            self._init_dynamic_publishers()
            return

        self.pub_pose_nav = rospy.Publisher(
            self.ros_cfg.pub_relative_pose_nav, PoseStamped, queue_size=1
        )
        self.pub_pose_det = rospy.Publisher(
            self.ros_cfg.pub_relative_pose_det, PoseStamped, queue_size=1
        )
        self.pub_pose_fused = rospy.Publisher(
            self.ros_cfg.pub_relative_pose, PoseStamped, queue_size=1
        )
        self.pub_polar = rospy.Publisher(
            self.ros_cfg.pub_relative_polar, Float32MultiArray, queue_size=1
        )
        self.pub_state = rospy.Publisher(
            self.ros_cfg.pub_link_state, String, queue_size=1
        )
        if self.ros_cfg.publish_markers:
            self.pub_markers = rospy.Publisher(
                self.ros_cfg.pub_markers, MarkerArray, queue_size=1
            )
        self.pub_pose_filtered = rospy.Publisher(
            self.ros_cfg.pub_relative_pose_filtered, PoseStamped, queue_size=1
        )
        self._init_coord_publishers()
        self._init_dynamic_publishers()

    def _link_allowed(self, out: Any) -> bool:
        pm = out.partner_match
        if pm is None or pm.partner_cluster is None:
            return False
        if (
            self.pipe.partner_match_cfg.require_traj_valid_for_link
            and not pm.traj_valid
        ):
            return False
        return True

    def _init_dynamic_publishers(self) -> None:
        self.pub_dynamic_targets = None
        self.pub_dynamic_markers = None
        if not self.ros_cfg.publish_dynamic:
            return
        try:
            from cooperative_link.msg import DynamicTarget, DynamicTargetArray
            self._DynamicTarget = DynamicTarget
            self._DynamicTargetArray = DynamicTargetArray
            self.pub_dynamic_targets = rospy.Publisher(
                self.ros_cfg.pub_dynamic_targets,
                DynamicTargetArray,
                queue_size=1,
            )
            self.pub_dynamic_markers = rospy.Publisher(
                self.ros_cfg.pub_dynamic_markers,
                MarkerArray,
                queue_size=1,
            )
        except ImportError:
            rospy.logwarn(
                "DynamicTarget msg not built; run catkin_make. Dynamic publish disabled."
            )

    def _publish_dynamic_outputs(
        self,
        stamp: Any,
        frame_id: str,
        out: Any,
    ) -> None:
        if self.pub_dynamic_targets is None and self.pub_dynamic_markers is None:
            return

        dynamic_targets = out.dynamic_targets or []
        if self.pub_dynamic_targets is not None and dynamic_targets:
            arr = self._DynamicTargetArray()
            arr.header.stamp = stamp
            arr.header.frame_id = frame_id
            for dt in dynamic_targets:
                m = self._DynamicTarget()
                m.track_id = int(dt.track_id)
                m.cluster_label = int(dt.cluster_label)
                m.x = float(dt.x)
                m.y = float(dt.y)
                m.z = float(dt.z)
                m.range_m = float(dt.range_m)
                m.bearing_rad = float(dt.bearing_rad)
                m.match_score = float(dt.match_score)
                m.motion_score = float(dt.motion_score)
                m.is_link_partner = bool(dt.is_link_partner)
                m.traj_align_cost = float(dt.traj_align_cost)
                m.traj_match_valid = bool(dt.traj_match_valid)
                arr.targets.append(m)
            self.pub_dynamic_targets.publish(arr)

        if self.pub_dynamic_markers is None:
            return

        dyn_cfg = self.pipe.dynamic_cfg
        partner_tid = -1
        if out.partner_match is not None:
            partner_tid = int(out.partner_match.partner_track_id)

        track_outputs = getattr(out, "track_outputs", None) or []
        rviz_tracks = select_tracks_for_rviz(
            track_outputs,
            dynamic_targets,
            partner_tid,
            dyn_cfg,
        )
        current_ids = marker_ids_from_tracks(rviz_tracks)
        lifetime = float(dyn_cfg.marker_lifetime_sec)
        ns = "coop_dynamic"

        markers = MarkerArray()
        z_default = self.gt_z
        for row in rviz_tracks:
            mid = dynamic_marker_id(row.track_id)
            z = float(row.z) if row.z != 0.0 else z_default
            if row.is_link_partner:
                rgba = (1.0, 0.85, 0.1, 0.95)
                diam = 0.5
            elif row.coasted:
                rgba = (0.45, 0.45, 0.55, 0.55)
                diam = 0.32
            else:
                rgba = (0.55, 0.55, 0.55, 0.85)
                diam = 0.35
            markers.markers.append(
                _make_sphere_marker(
                    stamp,
                    frame_id,
                    mid,
                    ns,
                    float(row.x),
                    float(row.y),
                    z,
                    diam,
                    rgba,
                    lifetime_sec=lifetime,
                )
            )

        if dyn_cfg.marker_delete_stale:
            for stale_id in stale_marker_ids(
                self._dynamic_marker_ids_active, current_ids
            ):
                markers.markers.append(
                    _make_sphere_marker(
                        stamp,
                        frame_id,
                        stale_id,
                        ns,
                        0.0,
                        0.0,
                        z_default,
                        0.01,
                        (0.0, 0.0, 0.0, 0.0),
                        action=Marker.DELETE,
                    )
                )

        self._dynamic_marker_ids_active = current_ids
        if markers.markers:
            self.pub_dynamic_markers.publish(markers)

    def _init_coord_publishers(self) -> None:
        try:
            from cooperative_link.msg import CoordFrameEstimate
            self._CoordFrameEstimate = CoordFrameEstimate
            if self.multi_cfg.enabled:
                for pid in self._configured_partner_ids:
                    self.pub_coord_by_id[pid] = rospy.Publisher(
                        partner_topic(self.ros_cfg.pub_coord_frame_estimate, pid),
                        CoordFrameEstimate,
                        queue_size=1,
                    )
            else:
                self.pub_coord_frame = rospy.Publisher(
                    self.ros_cfg.pub_coord_frame_estimate,
                    CoordFrameEstimate,
                    queue_size=1,
                )
        except ImportError:
            rospy.logerr(
                "cooperative_link CoordFrameEstimate msg not built; "
                "run catkin_make and source devel/setup.bash"
            )

    def _publish_post_outputs(
        self,
        stamp: Any,
        frame_id: str,
        out: Any,
        partner_id: int,
        yaw_nav: float,
    ) -> None:
        if out.post is None:
            return
        post = out.post
        if post.filtered_xy is not None:
            pub_filt = (
                self.pub_pose_filtered_by_id.get(partner_id)
                if self.multi_cfg.enabled
                else self.pub_pose_filtered
            )
            if pub_filt is not None:
                pub_filt.publish(
                    _make_pose(
                        stamp,
                        frame_id,
                        float(post.filtered_xy[0]),
                        float(post.filtered_xy[1]),
                        self.gt_z,
                        yaw_nav,
                    )
                )
        if self._CoordFrameEstimate is None:
            return
        pub_coord = (
            self.pub_coord_by_id.get(partner_id)
            if self.multi_cfg.enabled
            else self.pub_coord_frame
        )
        if pub_coord is None:
            return
        msg = self._CoordFrameEstimate()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.partner_id = int(partner_id)
        msg.yaw_rad = float(post.yaw_error_rad)
        msg.tx = float(post.tx)
        msg.ty = float(post.ty)
        msg.alignment_cost = float(post.align_cost)
        msg.valid = bool(post.align_valid)
        msg.reference_frame = frame_id
        pub_coord.publish(msg)

    def _host_role_buffer(self):
        if self.nav_multi is not None:
            return self.nav_multi.host
        if self.nav_buf is not None:
            return self.nav_buf.host
        return None

    def _link_target_role_buffer(self):
        if self.nav_multi is not None:
            return self.nav_multi.partners.get(self._link_target_id)
        if self.nav_buf is not None:
            return self.nav_buf.target
        return None

    def _format_buffer_span(self, span: Dict[str, Any]) -> str:
        ll = span.get("ll_range")
        yaw = span.get("yaw_range")
        ll_s = f"[{ll[0]:.3f},{ll[1]:.3f}]" if ll else "none"
        yaw_s = f"[{yaw[0]:.3f},{yaw[1]:.3f}]" if yaw else "none"
        return f"ll={ll_s} n={span['ll_n']} yaw={yaw_s} n={span['yaw_n']}"

    def _log_missing_host(self, t_nav: float) -> None:
        buf = self._host_role_buffer()
        if buf is None:
            rospy.logwarn_throttle(
                5.0,
                "NAV buffer missing host at t_nav=%.3f; no buffer configured",
                t_nav,
            )
            return
        reason = buf.query_miss_reason(t_nav)
        span = buf.buffer_span()
        rospy.logwarn_throttle(
            5.0,
            "NAV buffer missing host at t_nav=%.3f reason=%s %s; "
            "check nav topics / time_offset_sec / rosbag --clock",
            t_nav,
            reason,
            self._format_buffer_span(span),
        )

    def _log_missing_link_target(self, t_nav: float) -> None:
        buf = self._link_target_role_buffer()
        if buf is None:
            rospy.logwarn_throttle(
                5.0,
                "NAV buffer missing link target id=%d at t_nav=%.3f; no buffer",
                self._link_target_id,
                t_nav,
            )
            return
        reason = buf.query_miss_reason(t_nav)
        span = buf.buffer_span()
        rospy.logwarn_throttle(
            5.0,
            "NAV buffer missing link target id=%d at t_nav=%.3f reason=%s %s",
            self._link_target_id,
            t_nav,
            reason,
            self._format_buffer_span(span),
        )

    def _push_host_pose(
        self, stamp_sec: float, lat_deg: float, lon_deg: float, yaw_deg: float
    ) -> None:
        if self.nav_multi is not None:
            self.nav_multi.host.push_pose(stamp_sec, lat_deg, lon_deg, yaw_deg)
        elif self.nav_buf is not None:
            self.nav_buf.host.push_pose(stamp_sec, lat_deg, lon_deg, yaw_deg)

    def _push_target_pose(
        self, stamp_sec: float, lat_deg: float, lon_deg: float, yaw_deg: float
    ) -> None:
        if self.nav_buf is not None:
            self.nav_buf.target.push_pose(stamp_sec, lat_deg, lon_deg, yaw_deg)

    def _push_partner_pose(
        self,
        partner_id: int,
        stamp_sec: float,
        lat_deg: float,
        lon_deg: float,
        yaw_deg: float,
    ) -> None:
        if self.nav_multi is None:
            return
        buf = self.nav_multi.partners.get(partner_id)
        if buf is not None:
            buf.push_pose(stamp_sec, lat_deg, lon_deg, yaw_deg)

    def _host_navsat_cb(self, msg: NavSatFix) -> None:
        t = stamp_to_sec(msg.header.stamp)
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if self._pending_host_yaw is not None:
            yaw = self._pending_host_yaw
            self._pending_host_yaw = None
            self._pending_host_navsat = None
            self._push_host_pose(t, lat, lon, yaw)
            return
        if self._pending_host_navsat is not None:
            rospy.logwarn_throttle(
                5.0, "host navsat without matching yaw; replacing pending sample"
            )
        self._pending_host_navsat = (t, lat, lon)

    def _host_yaw_cb(self, msg: Float64) -> None:
        yaw = float(msg.data)
        pending = self._pending_host_navsat
        if pending is not None:
            t, lat, lon = pending
            self._pending_host_navsat = None
            self._push_host_pose(t, lat, lon, yaw)
            return
        self._pending_host_yaw = yaw

    def _target_navsat_cb(self, msg: NavSatFix) -> None:
        t = stamp_to_sec(msg.header.stamp)
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        if self._pending_target_yaw is not None:
            yaw = self._pending_target_yaw
            self._pending_target_yaw = None
            self._pending_target_navsat = None
            self._push_target_pose(t, lat, lon, yaw)
            return
        if self._pending_target_navsat is not None:
            rospy.logwarn_throttle(
                5.0, "target navsat without matching yaw; replacing pending sample"
            )
        self._pending_target_navsat = (t, lat, lon)

    def _target_yaw_cb(self, msg: Float64) -> None:
        yaw = float(msg.data)
        pending = self._pending_target_navsat
        if pending is not None:
            t, lat, lon = pending
            self._pending_target_navsat = None
            self._push_target_pose(t, lat, lon, yaw)
            return
        self._pending_target_yaw = yaw

    def _partner_navsat_cb(self, msg: NavSatFix, partner_id: int) -> None:
        t = stamp_to_sec(msg.header.stamp)
        lat = float(msg.latitude)
        lon = float(msg.longitude)
        pending_yaw = self._pending_partner_yaw.pop(partner_id, None)
        if pending_yaw is not None:
            self._pending_partner_navsat.pop(partner_id, None)
            self._push_partner_pose(partner_id, t, lat, lon, pending_yaw)
            return
        if partner_id in self._pending_partner_navsat:
            rospy.logwarn_throttle(
                5.0,
                "partner %d navsat without matching yaw; replacing pending sample",
                partner_id,
            )
        self._pending_partner_navsat[partner_id] = (t, lat, lon)

    def _partner_yaw_cb(self, msg: Float64, partner_id: int) -> None:
        yaw = float(msg.data)
        pending = self._pending_partner_navsat.pop(partner_id, None)
        if pending is not None:
            t, lat, lon = pending
            self._push_partner_pose(partner_id, t, lat, lon, yaw)
            return
        self._pending_partner_yaw[partner_id] = yaw

    def _query_host(self, t_nav: float) -> Optional[Dict[str, float]]:
        if self.nav_multi is not None:
            return self.nav_multi.query_host(t_nav)
        if self.nav_buf is not None:
            host, _ = self.nav_buf.query_both(t_nav)
            return host
        return None

    def _query_link_target(self, t_nav: float) -> Optional[Dict[str, float]]:
        if self.nav_multi is not None:
            return self.nav_multi.query_partner(self._link_target_id, t_nav)
        if self.nav_buf is not None:
            _, target = self.nav_buf.query_both(t_nav)
            return target
        return None

    def _query_link_poses(
        self, t_nav: float
    ) -> Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        return self._query_host(t_nav), self._query_link_target(t_nav)

    def _publish_cooperative_targets(self, stamp: Any, t_nav: float) -> None:
        if self.pub_cooperative_targets is None:
            return
        if self.nav_multi is not None:
            poses = self.nav_multi.query_all_partners(t_nav)
            nav_valid_by_id = nav_valid_from_poses(self._configured_partner_ids, poses)
        elif self.nav_buf is not None:
            _, target = self.nav_buf.query_both(t_nav)
            nav_valid_by_id = {
                self._configured_partner_ids[0]: target is not None,
            }
        else:
            nav_valid_by_id = {}

        host_id, count, partner_ids, nav_valid, link_target_id = build_cooperative_targets(
            host_id=self.uwb_cfg.host_agent_id,
            link_target_id=self._link_target_id,
            partner_ids=self._configured_partner_ids,
            nav_valid_by_id=nav_valid_by_id,
        )
        msg = self._CooperativeTargets()
        msg.header.stamp = stamp
        msg.header.frame_id = "cooperative_link"
        msg.host_id = host_id
        msg.count = count
        msg.partner_ids = partner_ids
        msg.nav_valid = nav_valid
        msg.link_target_id = link_target_id
        self.pub_cooperative_targets.publish(msg)

    def _uwb_geometry(self) -> Dict[str, Any]:
        return {
            "alt0_m": float(getattr(self.pipe, "alt0_m", 0.0)),
            "R_imu_to_vehicle": self.cfg.get("R_imu_to_vehicle", np.eye(3).tolist()),
            "R_lidar_vehicle": self.cfg.get("R_lidar_vehicle", np.eye(3).tolist()),
            "t_lidar_in_vehicle": self.cfg.get("t_lidar_in_vehicle", [0.0, 0.0, 0.0]),
            "target_imu_to_center_offset": self.cfg.get(
                "target_imu_to_center_offset", [0.0, 0.0, 0.0]
            ),
            "lidar_yaw_bias": float(self.cfg.get("lidar_yaw_bias", 0.0)),
            "gt_center_z_m": self.gt_z,
        }

    def _downsample(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        max_n = self.ros_cfg.max_points
        if n <= max_n:
            return pts
        idx = np.random.choice(n, size=max_n, replace=False)
        return pts[idx]

    def _compute_prior_for_poses(
        self,
        host: Dict[str, float],
        target: Dict[str, float],
    ) -> Any:
        lat0, lon0 = self.pipe._require_enu_origin()
        return compute_instant_prior(
            host,
            target,
            lat0,
            lon0,
            self.pipe.alt0_m,
            self.pipe.R_imu_to_vehicle,
            self.pipe.R_lidar_vehicle,
            self.pipe.t_lidar,
            self.pipe.target_offset,
            self.pipe.lidar_yaw_bias,
            gt_center_z_m=self.gt_z,
        )

    def _publish_all_partner_nav(
        self,
        stamp: Any,
        t_nav: float,
        host: Dict[str, float],
        frame_id: str,
    ) -> None:
        assert self.nav_multi is not None
        for pid in self._configured_partner_ids:
            partner_pose = self.nav_multi.query_partner(pid, t_nav)
            pub = self.pub_pose_nav_by_id.get(pid)
            if partner_pose is None or pub is None:
                continue
            prior = self._compute_prior_for_poses(host, partner_pose)
            nav_xy = prior.center_lidar
            pub.publish(
                _make_pose(
                    stamp,
                    frame_id,
                    float(nav_xy[0]),
                    float(nav_xy[1]),
                    self.gt_z,
                    float(prior.yaw_lidar),
                )
            )

    def _publish_link_pipeline_outputs(
        self,
        stamp: Any,
        frame_id: str,
        out: Any,
        partner_id: int,
    ) -> None:
        self._publish_dynamic_outputs(stamp, frame_id, out)
        link_ok = self._link_allowed(out)
        yaw_nav = float(out.prior.yaw_lidar)
        nav_xy = out.step.center_nav_xy
        has_cluster = link_ok and out.step.best_cluster is not None
        det_xy = (
            out.step.best_cluster.centroid_xy.copy()
            if has_cluster
            else None
        )

        pub_state = self.pub_state_by_id.get(partner_id)
        pub_nav = self.pub_pose_nav_by_id.get(partner_id)
        pub_det = self.pub_pose_det_by_id.get(partner_id)
        pub_fused = self.pub_pose_fused_by_id.get(partner_id)
        pub_polar = self.pub_polar_by_id.get(partner_id)

        if pub_state is not None:
            pub_state.publish(String(data=out.step.state.value))
        if pub_nav is not None:
            pub_nav.publish(
                _make_pose(
                    stamp,
                    frame_id,
                    float(nav_xy[0]),
                    float(nav_xy[1]),
                    self.gt_z,
                    yaw_nav,
                )
            )
        if has_cluster and det_xy is not None and pub_det is not None:
            pub_det.publish(
                _make_pose(
                    stamp,
                    frame_id,
                    float(det_xy[0]),
                    float(det_xy[1]),
                    self.gt_z,
                    0.0,
                )
            )

        fx, fy = float(out.step.center_fused_xy[0]), float(out.step.center_fused_xy[1])
        valid = link_ok and _output_valid(out.step.state, has_cluster)
        if pub_polar is not None:
            polar = Float32MultiArray()
            polar.data = [
                float(np.hypot(fx, fy)),
                float(np.arctan2(fy, fx)),
                float(out.step.match_score),
                1.0 if valid else 0.0,
            ]
            pub_polar.publish(polar)

        if valid and pub_fused is not None:
            yaw_fused = yaw_nav if out.step.state == LinkState.LOCKED else 0.0
            pub_fused.publish(
                _make_pose(stamp, frame_id, fx, fy, self.gt_z, yaw_fused)
            )
            if self.pub_marker is not None:
                self.pub_marker.publish(
                    _make_sphere_marker(
                        stamp,
                        frame_id,
                        0,
                        f"cooperative_target_legacy_{partner_id}",
                        fx,
                        fy,
                        self.gt_z,
                        0.5,
                        (1.0, 0.85, 0.0, 0.9)
                        if out.step.state == LinkState.LOCKED
                        else (1.0, 0.3, 0.1, 0.9),
                    )
                )

        if link_ok:
            self._publish_post_outputs(stamp, frame_id, out, partner_id, yaw_nav)
        self._publish_markers(
            stamp,
            frame_id,
            nav_xy,
            det_xy,
            out.step.center_fused_xy,
            out.step.state,
            has_cluster,
            partner_id=partner_id,
            filtered_xy=out.post.filtered_xy if out.post else None,
        )

    def _publish_markers(
        self,
        stamp: Any,
        frame_id: str,
        nav_xy: np.ndarray,
        det_xy: Optional[np.ndarray],
        fused_xy: np.ndarray,
        state: LinkState,
        has_det: bool,
        partner_id: Optional[int] = None,
        filtered_xy: Optional[np.ndarray] = None,
    ) -> None:
        pub_markers = self.pub_markers
        if partner_id is not None:
            pub_markers = self.pub_markers_by_id.get(partner_id)
        if pub_markers is None:
            return
        arr = MarkerArray()
        z = self.gt_z
        nx, ny = float(nav_xy[0]), float(nav_xy[1])
        arr.markers.append(
            _make_sphere_marker(
                stamp, frame_id, MARKER_ID_NAV, "coop_nav",
                nx, ny, z, 0.45, (0.1, 0.95, 0.25, 0.95),
            )
        )
        arr.markers.append(
            _make_line_to_origin(
                stamp, frame_id, MARKER_ID_LINE_NAV, "coop_nav",
                nx, ny, z, (0.1, 0.85, 0.2, 0.6),
            )
        )
        if has_det and det_xy is not None:
            dx, dy = float(det_xy[0]), float(det_xy[1])
            arr.markers.append(
                _make_sphere_marker(
                    stamp, frame_id, MARKER_ID_DET, "coop_det",
                    dx, dy, z, 0.35, (0.95, 0.15, 0.85, 0.95),
                )
            )
        else:
            arr.markers.append(
                _make_sphere_marker(
                    stamp, frame_id, MARKER_ID_DET, "coop_det",
                    0.0, 0.0, z, 0.01, (0.0, 0.0, 0.0, 0.0),
                    action=Marker.DELETE,
                )
            )
        fx, fy = float(fused_xy[0]), float(fused_xy[1])
        if state in (LinkState.LOCKING, LinkState.LOCKED, LinkState.REACQUIRE):
            fused_color = (1.0, 0.85, 0.1, 0.85) if state == LinkState.LOCKED else (1.0, 0.45, 0.1, 0.85)
            arr.markers.append(
                _make_sphere_marker(
                    stamp, frame_id, MARKER_ID_FUSED, "coop_fused",
                    fx, fy, z, 0.5, fused_color,
                )
            )
        if filtered_xy is not None:
            fx2, fy2 = float(filtered_xy[0]), float(filtered_xy[1])
            arr.markers.append(
                _make_sphere_marker(
                    stamp, frame_id, 99, "coop_filtered",
                    fx2, fy2, z, 0.4, (0.85, 0.2, 0.85, 0.9),
                )
            )
        pub_markers.publish(arr)

    def _lidar_cb(self, msg: PointCloud2) -> None:
        ros_t = stamp_to_sec(msg.header.stamp)
        t_nav = ros_t + self.time_offset
        stamp = msg.header.stamp
        host = self._query_host(t_nav)
        if host is None:
            self._log_missing_host(t_nav)
            return

        self._publish_cooperative_targets(stamp, t_nav)

        if not self._enu_set:
            self.pipe.set_enu_origin_if_needed(host)
            self._enu_set = True

        frame_id = self.ros_cfg.output_frame_id or msg.header.frame_id or "lidar"

        if self.multi_cfg.enabled:
            self._publish_all_partner_nav(stamp, t_nav, host, frame_id)
            target = self._query_link_target(t_nav)
            if target is None:
                self._log_missing_link_target(t_nav)
                self._publish_uwb_multi(stamp, t_nav, host)
                return

            pts = self._downsample(read_cloud_numpy(msg))
            out = self.pipe.process_frame_with_poses(
                self._frame_idx, ros_t, t_nav, pts, host, target
            )
            self._frame_idx += 1
            self._publish_link_pipeline_outputs(
                stamp, frame_id, out, self._link_target_id
            )
            self._publish_uwb_multi(stamp, t_nav, host)
            return

        target = self._query_link_target(t_nav)
        if target is None:
            self._log_missing_link_target(t_nav)
            return

        pts = self._downsample(read_cloud_numpy(msg))
        out = self.pipe.process_frame_with_poses(
            self._frame_idx, ros_t, t_nav, pts, host, target
        )
        self._frame_idx += 1
        self._publish_dynamic_outputs(stamp, frame_id, out)
        link_ok = self._link_allowed(out)

        yaw_nav = float(out.prior.yaw_lidar)
        nav_xy = out.step.center_nav_xy
        has_cluster = link_ok and out.step.best_cluster is not None
        det_xy = (
            out.step.best_cluster.centroid_xy.copy()
            if has_cluster
            else None
        )

        assert self.pub_state is not None
        self.pub_state.publish(String(data=out.step.state.value))

        assert self.pub_pose_nav is not None
        self.pub_pose_nav.publish(
            _make_pose(
                stamp, frame_id,
                float(nav_xy[0]), float(nav_xy[1]), self.gt_z, yaw_nav,
            )
        )

        if has_cluster and det_xy is not None:
            assert self.pub_pose_det is not None
            self.pub_pose_det.publish(
                _make_pose(
                    stamp, frame_id,
                    float(det_xy[0]), float(det_xy[1]), self.gt_z, 0.0,
                )
            )

        fx, fy = float(out.step.center_fused_xy[0]), float(out.step.center_fused_xy[1])
        valid = link_ok and _output_valid(out.step.state, has_cluster)

        assert self.pub_polar is not None
        polar = Float32MultiArray()
        polar.data = [
            float(np.hypot(fx, fy)),
            float(np.arctan2(fy, fx)),
            float(out.step.match_score),
            1.0 if valid else 0.0,
        ]
        self.pub_polar.publish(polar)

        if valid:
            yaw_fused = yaw_nav if out.step.state == LinkState.LOCKED else 0.0
            assert self.pub_pose_fused is not None
            self.pub_pose_fused.publish(
                _make_pose(stamp, frame_id, fx, fy, self.gt_z, yaw_fused)
            )
            if self.pub_marker is not None:
                self.pub_marker.publish(
                    _make_sphere_marker(
                        stamp, frame_id, 0, "cooperative_target_legacy",
                        fx, fy, self.gt_z, 0.5,
                        (1.0, 0.85, 0.0, 0.9) if out.step.state == LinkState.LOCKED
                        else (1.0, 0.3, 0.1, 0.9),
                    )
                )

        if link_ok:
            self._publish_post_outputs(
                stamp, frame_id, out, int(self._link_target_id), yaw_nav
            )
        self._publish_markers(
            stamp, frame_id, nav_xy, det_xy, out.step.center_fused_xy,
            out.step.state, has_cluster,
            filtered_xy=out.post.filtered_xy if out.post and link_ok else None,
        )
        self._publish_uwb(stamp, out.prior)

    def _reading_to_msg(self, stamp: Any, reading: Any) -> Any:
        msg = self._UwbRange()
        msg.header.stamp = stamp
        msg.header.frame_id = "enu"
        msg.from_id = int(reading.from_id)
        msg.to_id = int(reading.to_id)
        msg.range_m = float(reading.range_m)
        msg.range_std = float(reading.range_std)
        msg.valid = bool(reading.valid)
        msg.source = str(reading.source)
        return msg

    def _publish_uwb_multi(self, stamp: Any, t_nav: float, host: Dict[str, float]) -> None:
        if self.pub_uwb_ranges is None or not self.uwb_cfg.enabled:
            return
        assert self.nav_multi is not None
        partner_poses = self.nav_multi.query_all_partners(t_nav)
        lat0 = float(getattr(self.pipe, "lat0_deg", host["lat_deg"]))
        lon0 = float(getattr(self.pipe, "lon0_deg", host["lon_deg"]))
        readings = build_uwb_range_readings(
            host_id=self.uwb_cfg.host_agent_id,
            partners=self.multi_cfg.partners,
            host_pose=host,
            partner_poses=partner_poses,
            lat0_deg=lat0,
            lon0_deg=lon0,
            cfg=self.uwb_cfg,
            geometry=self._uwb_geometry(),
        )
        if not readings and self.uwb_cfg.publish_only_when_nav_ok:
            return

        arr = self._UwbRangeArray()
        arr.header.stamp = stamp
        arr.header.frame_id = "enu"
        arr.ranges = [self._reading_to_msg(stamp, r) for r in readings]
        self.pub_uwb_ranges.publish(arr)

        if self.uwb_cfg.publish_per_partner_topics:
            for reading in readings:
                pub = self.pub_uwb_per_partner.get(reading.to_id)
                if pub is not None:
                    pub.publish(self._reading_to_msg(stamp, reading))

        valid_readings = [r for r in readings if r.valid]
        if self.pub_uwb is not None and self.uwb_cfg.publish_legacy_single:
            if len(valid_readings) == 1:
                self.pub_uwb.publish(self._reading_to_msg(stamp, valid_readings[0]))

    def _publish_uwb(self, stamp: Any, prior: Any) -> None:
        if self.pub_uwb is None or not self.uwb_cfg.enabled:
            return
        range_m = enu_range_from_prior(prior)
        range_m = apply_range_noise(range_m, self.uwb_cfg.range_noise_std)
        msg = self._UwbRange()
        msg.header.stamp = stamp
        msg.header.frame_id = "enu"
        msg.from_id = int(self.uwb_cfg.from_agent_id)
        msg.to_id = int(self.uwb_cfg.to_agent_id)
        msg.range_m = range_m
        msg.range_std = float(self.uwb_cfg.range_noise_std)
        msg.valid = True
        msg.source = self.uwb_cfg.source_label
        self.pub_uwb.publish(msg)


def main() -> None:
    rospy.init_node("cooperative_link_node")
    CooperativeLinkRosNode()
    rospy.spin()


if __name__ == "__main__":
    main()
