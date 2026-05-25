"""Parse cooperative_link_ros section from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from cooperative_link.partner_config import load_multi_link_config

_MULTI_NAV_DEFAULT = "/cooperative_target/{id}/relative_pose_nav"
_MULTI_DET_DEFAULT = "/cooperative_target/{id}/relative_pose_det"
_MULTI_FUSED_DEFAULT = "/cooperative_target/{id}/relative_pose"
_MULTI_POLAR_DEFAULT = "/cooperative_target/{id}/relative_polar"
_MULTI_STATE_DEFAULT = "/cooperative_target/{id}/link_state"
_MULTI_MARKERS_DEFAULT = "/cooperative_target/{id}/markers"

_FLAT_NAV_DEFAULT = "/cooperative_target/relative_pose_nav"
_FLAT_DET_DEFAULT = "/cooperative_target/relative_pose_det"
_FLAT_FUSED_DEFAULT = "/cooperative_target/relative_pose"
_FLAT_POLAR_DEFAULT = "/cooperative_target/relative_polar"
_FLAT_STATE_DEFAULT = "/cooperative_link/state"
_FLAT_MARKERS_DEFAULT = "/cooperative_target/markers"
_MULTI_FILTERED_DEFAULT = "/cooperative_target/{id}/relative_pose_filtered"
_MULTI_COORD_DEFAULT = "/cooperative_target/{id}/coord_frame_estimate"
_FLAT_FILTERED_DEFAULT = "/cooperative_target/relative_pose_filtered"
_FLAT_COORD_DEFAULT = "/cooperative_target/coord_frame_estimate"


@dataclass
class RosLinkConfig:
    lidar_topic: str = "/lslidar_point_cloud"
    host_navsat_topic: str = "/host/navsat"
    host_yaw_topic: str = "/host/yaw_deg"
    target_navsat_topic: str = "/target/navsat"
    target_yaw_topic: str = "/target/yaw_deg"
    pub_relative_pose: str = "/cooperative_target/relative_pose"
    pub_relative_pose_nav: str = "/cooperative_target/relative_pose_nav"
    pub_relative_pose_det: str = "/cooperative_target/relative_pose_det"
    pub_relative_polar: str = "/cooperative_target/relative_polar"
    pub_link_state: str = "/cooperative_link/state"
    pub_cooperative_targets: str = "/cooperative_targets"
    pub_marker: str = "/cooperative_target/marker"
    pub_markers: str = "/cooperative_target/markers"
    pub_relative_pose_filtered: str = "/cooperative_target/relative_pose_filtered"
    pub_coord_frame_estimate: str = "/cooperative_target/coord_frame_estimate"
    pub_dynamic_targets: str = "/cooperative_dynamic/targets"
    pub_dynamic_markers: str = "/cooperative_dynamic/markers"
    publish_marker: bool = True
    publish_markers: bool = True
    publish_dynamic: bool = True
    nav_buffer_sec: float = 60.0
    max_query_slop_sec: float = 0.15
    max_points: int = 80000
    output_frame_id: str = ""


def load_ros_link_config(cfg: Dict[str, Any]) -> RosLinkConfig:
    raw = cfg.get("cooperative_link_ros") or {}
    multi = load_multi_link_config(cfg)
    if multi.enabled:
        nav_d = _MULTI_NAV_DEFAULT
        det_d = _MULTI_DET_DEFAULT
        fused_d = _MULTI_FUSED_DEFAULT
        polar_d = _MULTI_POLAR_DEFAULT
        state_d = _MULTI_STATE_DEFAULT
        markers_d = _MULTI_MARKERS_DEFAULT
        filtered_d = _MULTI_FILTERED_DEFAULT
        coord_d = _MULTI_COORD_DEFAULT
    else:
        nav_d = _FLAT_NAV_DEFAULT
        det_d = _FLAT_DET_DEFAULT
        fused_d = _FLAT_FUSED_DEFAULT
        polar_d = _FLAT_POLAR_DEFAULT
        state_d = _FLAT_STATE_DEFAULT
        markers_d = _FLAT_MARKERS_DEFAULT
        filtered_d = _FLAT_FILTERED_DEFAULT
        coord_d = _FLAT_COORD_DEFAULT
    cal_raw = cfg.get("cooperative_link_calibration") or {}
    return RosLinkConfig(
        lidar_topic=str(raw.get("lidar_topic", "/lslidar_point_cloud")),
        host_navsat_topic=str(raw.get("host_navsat_topic", "/host/navsat")),
        host_yaw_topic=str(raw.get("host_yaw_topic", "/host/yaw_deg")),
        target_navsat_topic=str(raw.get("target_navsat_topic", "/target/navsat")),
        target_yaw_topic=str(raw.get("target_yaw_topic", "/target/yaw_deg")),
        pub_relative_pose=str(raw.get("pub_relative_pose", fused_d)),
        pub_relative_pose_nav=str(raw.get("pub_relative_pose_nav", nav_d)),
        pub_relative_pose_det=str(raw.get("pub_relative_pose_det", det_d)),
        pub_relative_polar=str(raw.get("pub_relative_polar", polar_d)),
        pub_link_state=str(raw.get("pub_link_state", state_d)),
        pub_cooperative_targets=str(
            raw.get("pub_cooperative_targets", "/cooperative_targets")
        ),
        pub_marker=str(raw.get("pub_marker", "/cooperative_target/marker")),
        pub_markers=str(raw.get("pub_markers", markers_d)),
        pub_relative_pose_filtered=str(
            cal_raw.get("pub_relative_pose_filtered", filtered_d)
        ),
        pub_coord_frame_estimate=str(
            cal_raw.get("pub_coord_frame", cal_raw.get("pub_coord_frame_estimate", coord_d))
        ),
        pub_dynamic_targets=str(
            raw.get("pub_dynamic_targets", "/cooperative_dynamic/targets")
        ),
        pub_dynamic_markers=str(
            raw.get("pub_dynamic_markers", "/cooperative_dynamic/markers")
        ),
        publish_marker=bool(raw.get("publish_marker", False)),
        publish_markers=bool(raw.get("publish_markers", True)),
        publish_dynamic=bool(raw.get("publish_dynamic", True)),
        nav_buffer_sec=float(raw.get("nav_buffer_sec", 60.0)),
        max_query_slop_sec=float(raw.get("max_query_slop_sec", 0.15)),
        max_points=int(raw.get("max_points", 80000)),
        output_frame_id=str(raw.get("output_frame_id", "")),
    )
