"""End-to-end per-frame cooperative link processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from cooperative_link.associate import CooperativeAssociator, LinkState, StepResult
from cooperative_link.config import LinkConfig, load_link_config
from cooperative_link.dynamic_detect import detect_dynamic_clusters
from cooperative_link.dynamic_targets import DynamicTargetInfo, build_dynamic_targets
from cooperative_link.ego_motion import compensate_points_to_current, motion_residual_nn
from cooperative_link.filter_config import (
    CalibrationConfig,
    DynamicConfig,
    FilterConfig,
    PartnerMatchConfig,
    load_calibration_config,
    load_dynamic_config,
    load_filter_config,
    load_partner_match_config,
)
from cooperative_link.partner_match import PartnerMatchResult, PartnerTrajectoryMatcher
from cooperative_link.prior import InstantPrior, compute_instant_prior
from cooperative_link.nav_pose import NavSeries, pose_dict_2d
from cooperative_link.post_process import CooperativePostProcessor, PostProcessResult
from cooperative_link.track_filter import TrackFilterManager, TrackOutput


@dataclass
class FrameOutput:
    frame_idx: int
    ros_t: float
    t_nav: float
    prior: InstantPrior
    step: StepResult
    n_clusters: int
    n_gated: int
    clusters: List[Any] = field(default_factory=list)
    dynamic_targets: List[DynamicTargetInfo] = field(default_factory=list)
    partner_match: Optional[PartnerMatchResult] = None
    track_outputs: List[TrackOutput] = field(default_factory=list)
    post: Optional[PostProcessResult] = None
    log_record: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CooperativeLinkPipeline:
    alt0_m: float
    R_imu_to_vehicle: np.ndarray
    R_lidar_vehicle: np.ndarray
    t_lidar: np.ndarray
    target_offset: np.ndarray
    lidar_yaw_bias: float
    gt_center_z_m: float
    link_cfg: LinkConfig
    lat0_deg: Optional[float] = None
    lon0_deg: Optional[float] = None
    nav_h: Optional[NavSeries] = None
    nav_t: Optional[NavSeries] = None
    associator: CooperativeAssociator = field(init=False)
    post_processor: Optional[CooperativePostProcessor] = None
    tracker: Optional[TrackFilterManager] = None
    partner_matcher: Optional[PartnerTrajectoryMatcher] = None
    dynamic_cfg: DynamicConfig = field(default_factory=DynamicConfig)
    partner_match_cfg: PartnerMatchConfig = field(default_factory=PartnerMatchConfig)
    filter_cfg: Optional[FilterConfig] = None
    _prev_pts: Optional[np.ndarray] = None
    _prev_host: Optional[Dict[str, float]] = None
    _prev_ros: Optional[float] = None

    def __post_init__(self) -> None:
        self.associator = CooperativeAssociator(
            link_cfg=self.link_cfg,
            gt_center_z_m=self.gt_center_z_m,
        )

    def attach_post_processor(self, cfg: Dict[str, Any]) -> None:
        self.filter_cfg = load_filter_config(cfg)
        cal_cfg = load_calibration_config(cfg)
        self.dynamic_cfg = load_dynamic_config(cfg)
        self.partner_match_cfg = load_partner_match_config(cfg)
        self.post_processor = CooperativePostProcessor.from_yaml(
            cfg, gt_z=self.gt_center_z_m
        )
        if self.filter_cfg.enabled:
            self.tracker = TrackFilterManager(self.filter_cfg, z_default=self.gt_center_z_m)
        self.partner_matcher = PartnerTrajectoryMatcher(
            self.partner_match_cfg,
            cal_cfg,
            link_score_thresh=self.link_cfg.score_thresh,
        )

    @classmethod
    def _geometry_from_cfg(cls, cfg: Dict[str, Any], lat0_deg: Optional[float], lon0_deg: Optional[float],
                           nav_h: Optional[NavSeries] = None,
                           nav_t: Optional[NavSeries] = None) -> "CooperativeLinkPipeline":
        def _mat3(key: str) -> np.ndarray:
            m = np.asarray(cfg[key], dtype=np.float64)
            if m.shape != (3, 3):
                raise ValueError(f"{key} must be 3x3")
            return m

        return cls(
            nav_h=nav_h,
            nav_t=nav_t,
            lat0_deg=lat0_deg,
            lon0_deg=lon0_deg,
            alt0_m=0.0,
            R_imu_to_vehicle=_mat3("R_imu_to_vehicle"),
            R_lidar_vehicle=_mat3("R_lidar_vehicle"),
            t_lidar=np.asarray(cfg["t_lidar_in_vehicle"], dtype=np.float64).reshape(3),
            target_offset=np.asarray(cfg["target_imu_to_center_offset"], dtype=np.float64).reshape(3),
            lidar_yaw_bias=float(cfg.get("lidar_yaw_bias", 0.0)),
            gt_center_z_m=float(cfg.get("gt_center_z_m", 0.0)),
            link_cfg=load_link_config(cfg),
            dynamic_cfg=load_dynamic_config(cfg),
            partner_match_cfg=load_partner_match_config(cfg),
        )

    @classmethod
    def from_yaml_cfg(cls, cfg: Dict[str, Any], nav_h: NavSeries, nav_t: NavSeries,
                      lat0_deg: float, lon0_deg: float) -> "CooperativeLinkPipeline":
        pipe = cls._geometry_from_cfg(cfg, lat0_deg, lon0_deg, nav_h=nav_h, nav_t=nav_t)
        pipe.attach_post_processor(cfg)
        return pipe

    @classmethod
    def from_yaml_geometry_only(cls, cfg: Dict[str, Any]) -> "CooperativeLinkPipeline":
        """ROS node: poses from topics; ENU origin set on first host pose."""
        return cls._geometry_from_cfg(cfg, None, None)

    def set_enu_origin_if_needed(self, host: Dict[str, float]) -> None:
        if self.lat0_deg is None or self.lon0_deg is None:
            self.lat0_deg = float(host["lat_deg"])
            self.lon0_deg = float(host["lon_deg"])

    def _require_enu_origin(self) -> Tuple[float, float]:
        if self.lat0_deg is None or self.lon0_deg is None:
            raise RuntimeError("ENU origin not set; call set_enu_origin_if_needed first")
        return float(self.lat0_deg), float(self.lon0_deg)

    def process_frame(
        self,
        frame_idx: int,
        ros_t: float,
        t_nav: float,
        pts: np.ndarray,
    ) -> FrameOutput:
        if self.nav_h is None or self.nav_t is None:
            raise RuntimeError("process_frame requires nav_h/nav_t; use process_frame_with_poses for ROS")
        host = pose_dict_2d(self.nav_h, float(t_nav))
        target = pose_dict_2d(self.nav_t, float(t_nav))
        return self.process_frame_with_poses(frame_idx, ros_t, t_nav, pts, host, target)

    def process_frame_with_poses(
        self,
        frame_idx: int,
        ros_t: float,
        t_nav: float,
        pts: np.ndarray,
        host: Dict[str, float],
        target: Dict[str, float],
    ) -> FrameOutput:
        lat0, lon0 = self._require_enu_origin()
        prior = compute_instant_prior(
            host,
            target,
            lat0,
            lon0,
            self.alt0_m,
            self.R_imu_to_vehicle,
            self.R_lidar_vehicle,
            self.t_lidar,
            self.target_offset,
            self.lidar_yaw_bias,
            gt_center_z_m=self.gt_center_z_m,
        )

        if self._prev_pts is not None and self._prev_host is not None:
            prev_comp, _ = compensate_points_to_current(
                self._prev_pts,
                self._prev_host,
                host,
                lat0,
                lon0,
                self.R_imu_to_vehicle,
                self.R_lidar_vehicle,
                self.t_lidar,
            )
            motion_res = motion_residual_nn(
                pts, prev_comp, self.link_cfg.nn_max_dist_m
            )
        else:
            motion_res = np.zeros(pts.shape[0], dtype=np.float64)

        gate, clusters = detect_dynamic_clusters(
            pts,
            prior,
            self.link_cfg,
            motion_res,
            gt_center_z_m=self.gt_center_z_m,
        )

        dt = 0.0
        if self._prev_ros is not None:
            dt = float(ros_t - self._prev_ros)

        nav_xy = prior.center_lidar[:2].copy()
        track_outputs: List[TrackOutput] = []
        if self.tracker is not None and self.filter_cfg is not None:
            track_outputs = self.tracker.update(
                clusters,
                dt,
                score_thresh=self.dynamic_cfg.track_score_thresh,
                min_motion_score=self.dynamic_cfg.min_motion_score,
            )

        partner_match = PartnerMatchResult()
        if self.partner_matcher is not None:
            partner_match = self.partner_matcher.step(nav_xy, clusters, track_outputs)

        link_ok = (
            partner_match.partner_cluster is not None
            and (
                not self.partner_match_cfg.require_traj_valid_for_link
                or partner_match.traj_valid
            )
        )
        fsm_clusters = (
            [partner_match.partner_cluster]
            if link_ok and partner_match.partner_cluster is not None
            else []
        )
        step = self.associator.step(prior, fsm_clusters, float(t_nav), dt)

        self._prev_pts = pts
        self._prev_host = host
        self._prev_ros = float(ros_t)

        dynamic_targets = build_dynamic_targets(
            clusters,
            track_outputs,
            self.dynamic_cfg,
            partner_match,
            self.gt_center_z_m,
        )

        best = step.best_cluster
        log_record = {
            "frame_idx": frame_idx,
            "ros_t": float(ros_t),
            "t_nav": float(t_nav),
            "state": step.state.value,
            "r_prior_m": prior.r_prior_m,
            "bearing_prior_rad": prior.bearing_prior_rad,
            "r_det_m": best.r_det_m if best else None,
            "bearing_det_rad": best.bearing_det_rad if best else None,
            "match_score": step.match_score,
            "consecutive_hits": step.consecutive_hits,
            "miss_frames": step.miss_frames,
            "locked": step.state == LinkState.LOCKED,
            "n_gated": int(gate.sum()),
            "n_clusters": len(clusters),
            "n_dynamic": len(dynamic_targets),
            "n_coasted_tracks": sum(1 for t in track_outputs if t.coasted),
            "partner_track_id": partner_match.partner_track_id,
            "traj_valid": partner_match.traj_valid,
            "partner_align_cost": partner_match.align_cost,
            "center_nav_x": float(step.center_nav_xy[0]),
            "center_nav_y": float(step.center_nav_xy[1]),
            "center_fused_x": float(step.center_fused_xy[0]),
            "center_fused_y": float(step.center_fused_xy[1]),
            "used_coast": step.used_coast,
        }

        post_result: Optional[PostProcessResult] = None
        if self.post_processor is not None and link_ok and partner_match.partner_cluster is not None:
            post_result = self.post_processor.process_frame(
                best_cluster=partner_match.partner_cluster,
                nav_xy=step.center_nav_xy,
                t_nav=float(t_nav),
                track_outputs=track_outputs,
                tracker=self.tracker,
            )
            log_record["track_id"] = post_result.track_id
            log_record["align_valid"] = post_result.align_valid
            log_record["align_cost"] = post_result.align_cost
            log_record["yaw_error_rad"] = post_result.yaw_error_rad
            log_record["tx"] = post_result.tx
            log_record["ty"] = post_result.ty
            if post_result.filtered_xy is not None:
                log_record["filtered_x"] = float(post_result.filtered_xy[0])
                log_record["filtered_y"] = float(post_result.filtered_xy[1])

        return FrameOutput(
            frame_idx=frame_idx,
            ros_t=float(ros_t),
            t_nav=float(t_nav),
            prior=prior,
            step=step,
            n_clusters=len(clusters),
            n_gated=int(gate.sum()),
            clusters=clusters,
            dynamic_targets=dynamic_targets,
            partner_match=partner_match,
            track_outputs=track_outputs,
            post=post_result,
            log_record=log_record,
        )
