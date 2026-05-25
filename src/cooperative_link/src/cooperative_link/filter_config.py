"""Parse cooperative_link_filter and cooperative_link_calibration YAML sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FilterConfig:
    enabled: bool = True
    score_thresh: float = 0.5
    assoc_gate_m: float = 2.5
    max_missed: int = 5
    q_scale: float = 0.1
    r_scale: float = 0.2
    fit_win: int = 10
    poly_order: int = 2
    filter_output: str = "ekf"  # ekf | poly
    predict_on_miss: bool = True
    max_speed_mps: float = 15.0
    coast_publish: bool = True
    coast_gate_extra_m: float = 1.0


@dataclass
class DynamicConfig:
    enabled: bool = True
    publish_score_thresh: float = 0.0
    track_score_thresh: float = 0.0
    min_motion_score: float = 0.0
    rviz_tracks_only: bool = True
    max_dynamic_markers: int = 6
    marker_lifetime_sec: float = 0.25
    marker_delete_stale: bool = True


@dataclass
class PartnerMatchConfig:
    enabled: bool = True
    window_frames: int = 30
    min_frames: int = 10
    max_align_cost: float = 5.0
    partner_lock_frames: int = 5
    partner_unlock_miss: int = 10
    require_traj_valid_for_link: bool = True
    require_match_score: bool = True


@dataclass
class CalibrationConfig:
    enabled: bool = True
    window_frames: int = 30
    min_frames: int = 10
    lambda_t: float = 2.0
    lambda_v: float = 3.5
    dtw_window: float = 0.3
    skip_penalty: float = 1.2
    max_iter: int = 50
    tol: float = 1e-6
    pub_coord_frame: str = "/cooperative_target/{id}/coord_frame_estimate"
    pub_relative_pose_filtered: str = "/cooperative_target/{id}/relative_pose_filtered"


def load_filter_config(cfg: Dict[str, Any]) -> FilterConfig:
    raw = cfg.get("cooperative_link_filter") or {}
    return FilterConfig(
        enabled=bool(raw.get("enabled", True)),
        score_thresh=float(raw.get("score_thresh", 0.5)),
        assoc_gate_m=float(raw.get("assoc_gate_m", 2.5)),
        max_missed=int(raw.get("max_missed", 5)),
        q_scale=float(raw.get("q_scale", 0.1)),
        r_scale=float(raw.get("r_scale", 0.2)),
        fit_win=int(raw.get("fit_win", 10)),
        poly_order=int(raw.get("poly_order", 2)),
        filter_output=str(raw.get("filter_output", "ekf")).lower(),
        predict_on_miss=bool(raw.get("predict_on_miss", True)),
        max_speed_mps=float(raw.get("max_speed_mps", 15.0)),
        coast_publish=bool(raw.get("coast_publish", True)),
        coast_gate_extra_m=float(raw.get("coast_gate_extra_m", 1.0)),
    )


def load_dynamic_config(cfg: Dict[str, Any]) -> DynamicConfig:
    raw = cfg.get("cooperative_link_dynamic") or {}
    return DynamicConfig(
        enabled=bool(raw.get("enabled", True)),
        publish_score_thresh=float(raw.get("publish_score_thresh", 0.0)),
        track_score_thresh=float(raw.get("track_score_thresh", 0.0)),
        min_motion_score=float(raw.get("min_motion_score", 0.0)),
        rviz_tracks_only=bool(raw.get("rviz_tracks_only", True)),
        max_dynamic_markers=int(raw.get("max_dynamic_markers", 6)),
        marker_lifetime_sec=float(raw.get("marker_lifetime_sec", 0.25)),
        marker_delete_stale=bool(raw.get("marker_delete_stale", True)),
    )


def load_partner_match_config(cfg: Dict[str, Any]) -> PartnerMatchConfig:
    raw = cfg.get("cooperative_link_partner_match") or {}
    cal_raw = cfg.get("cooperative_link_calibration") or {}
    return PartnerMatchConfig(
        enabled=bool(raw.get("enabled", True)),
        window_frames=int(raw.get("window_frames", cal_raw.get("window_frames", 30))),
        min_frames=int(raw.get("min_frames", cal_raw.get("min_frames", 10))),
        max_align_cost=float(raw.get("max_align_cost", 5.0)),
        partner_lock_frames=int(raw.get("partner_lock_frames", 5)),
        partner_unlock_miss=int(raw.get("partner_unlock_miss", 10)),
        require_traj_valid_for_link=bool(raw.get("require_traj_valid_for_link", True)),
        require_match_score=bool(raw.get("require_match_score", True)),
    )


def load_calibration_config(cfg: Dict[str, Any]) -> CalibrationConfig:
    raw = cfg.get("cooperative_link_calibration") or {}
    return CalibrationConfig(
        enabled=bool(raw.get("enabled", True)),
        window_frames=int(raw.get("window_frames", 30)),
        min_frames=int(raw.get("min_frames", 10)),
        lambda_t=float(raw.get("lambda_t", 2.0)),
        lambda_v=float(raw.get("lambda_v", 3.5)),
        dtw_window=float(raw.get("dtw_window", 0.3)),
        skip_penalty=float(raw.get("skip_penalty", 1.2)),
        max_iter=int(raw.get("max_iter", 50)),
        tol=float(raw.get("tol", 1e-6)),
        pub_coord_frame=str(
            raw.get("pub_coord_frame", "/cooperative_target/{id}/coord_frame_estimate")
        ),
        pub_relative_pose_filtered=str(
            raw.get(
                "pub_relative_pose_filtered",
                "/cooperative_target/{id}/relative_pose_filtered",
            )
        ),
    )
