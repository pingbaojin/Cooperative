#!/usr/bin/env python3
"""
Bag replay with cooperative link: prior gate + motion residual + lock FSM (offline / Open3D).

在线 ROS1 运行并发布相对位置请用:
  data_generate/cooperative_link_ros_node.py
  data_generate/nav_file_to_ros_publisher.py  (bag 无 NAV topic 时)
  data_generate/launch/cooperative_link.launch

用法:
  cd /docker_ws/ubuntu20_04/CenterPoint
  python3 data_generate/replay_dynamic_assoc.py --config data_generate/example_config.yaml
  python3 data_generate/replay_dynamic_assoc.py --config data_generate/example_config.yaml --log /tmp/cooperative_link.jsonl
  python3 data_generate/replay_dynamic_assoc.py --config data_generate/example_config.yaml --no_vis
"""

from __future__ import annotations


def _default_config_path() -> Path:
    try:
        import rospkg
        return Path(rospkg.RosPack().get_path("cooperative_link")) / "config" / "default.yaml"
    except Exception:
        return Path(__file__).resolve().parents[2] / "config" / "default.yaml"

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from cooperative_link.lidar_filter import filter_decimated_lidar_messages
from cooperative_link.associate import LinkState
from cooperative_link.dynamic_detect import prior_gate_mask
from cooperative_link.pipeline import CooperativeLinkPipeline
from cooperative_link.export_pointcloud_bin import iter_bag_lidar, read_cloud_numpy
from cooperative_link.nav_pose import load_nav_pair_from_cfg, unwrap_yaw_interp, warn_if_bag_starts_before_nav
from cooperative_link.time_sync import clamp_to_nav_range, ros_to_nav_time


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _downsample_points(pts: np.ndarray, max_n: int) -> np.ndarray:
    n = pts.shape[0]
    if n <= max_n:
        return pts
    idx = np.random.choice(n, size=max_n, replace=False)
    return pts[idx]


def _sector_arc_xy(
    r: float,
    bearing: float,
    half_deg: float,
    n: int = 32,
) -> np.ndarray:
    half = np.deg2rad(half_deg)
    angles = np.linspace(bearing - half, bearing + half, n)
    return np.stack([r * np.cos(angles), r * np.sin(angles)], axis=1)


def _make_vis_geometries(
    o3d: Any,
    pts: np.ndarray,
    out: Any,
    link_cfg: Any,
    gt_z: float,
    max_points: int,
) -> List[Any]:
    geoms: List[Any] = []
    pts_ds = _downsample_points(pts, max_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_ds[:, :3].astype(np.float64))
    colors = np.tile(np.array([[0.45, 0.45, 0.5]]), (pts_ds.shape[0], 1))
    gate = prior_gate_mask(pts_ds, out.prior, link_cfg, gt_z)
    colors[gate] = np.array([0.2, 0.75, 0.95])
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    geoms.append(pcd)

    nav_c = out.prior.center_lidar.astype(np.float64)
    sph_nav = o3d.geometry.TriangleMesh.create_sphere(radius=0.25)
    sph_nav.paint_uniform_color([0.0, 1.0, 0.2])
    sph_nav.translate(nav_c, relative=False)
    geoms.append(sph_nav)

    fused = np.array(
        [out.step.center_fused_xy[0], out.step.center_fused_xy[1], nav_c[2]],
        dtype=np.float64,
    )
    sph_f = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    color = [1.0, 0.2, 0.0]
    if out.step.state == LinkState.LOCKED:
        color = [1.0, 0.85, 0.0]
    sph_f.paint_uniform_color(color)
    sph_f.translate(fused, relative=False)
    geoms.append(sph_f)

    if out.step.best_cluster is not None:
        det = out.step.best_cluster.centroid_xy
        sph_d = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sph_d.paint_uniform_color([1.0, 0.0, 0.8])
        sph_d.translate(
            np.array([det[0], det[1], nav_c[2]], dtype=np.float64), relative=False
        )
        geoms.append(sph_d)

    if out.post is not None and out.post.filtered_xy is not None:
        filt = np.array(
            [out.post.filtered_xy[0], out.post.filtered_xy[1], nav_c[2]],
            dtype=np.float64,
        )
        sph_kf = o3d.geometry.TriangleMesh.create_sphere(radius=0.22)
        sph_kf.paint_uniform_color([0.85, 0.2, 0.85])
        sph_kf.translate(filt, relative=False)
        geoms.append(sph_kf)

    r0 = max(out.prior.r_prior_m - link_cfg.range_tol_m, 0.1)
    r1 = out.prior.r_prior_m + link_cfg.range_tol_m
    for r_ring, col in ((r0, [0.3, 0.8, 0.3]), (r1, [0.3, 0.8, 0.3])):
        arc = _sector_arc_xy(r_ring, out.prior.bearing_prior_rad, link_cfg.bearing_tol_deg)
        z = np.full((arc.shape[0], 1), gt_z)
        pts_arc = np.hstack([arc, z])
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts_arc.astype(np.float64)),
            lines=o3d.utility.Vector2iVector(
                np.array([[i, (i + 1) % arc.shape[0]] for i in range(arc.shape[0])], dtype=np.int32)
            ),
        )
        ls.colors = o3d.utility.Vector3dVector([col for _ in range(arc.shape[0])])
        geoms.append(ls)

    return geoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay bag + cooperative link viz")
    default_cfg = _default_config_path()
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--bag", type=str, default=None)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--max_points", type=int, default=80000)
    parser.add_argument("--sleep_cap", type=float, default=0.25)
    parser.add_argument("--log", type=str, default=None, help="write JSONL frame log")
    parser.add_argument("--max_frames", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bag_path = args.bag or cfg["bag_path"]
    topic = cfg["lidar_topic"]
    time_offset = float(cfg.get("time_offset_sec", 0.0))
    decimate = max(1, int(cfg.get("lidar_decimate", 1)))
    skip_start_sec = float(cfg.get("skip_start_sec", 0.0))
    lidar_output_hz = float(cfg.get("lidar_output_hz", 0.0))
    gt_z = float(cfg.get("gt_center_z_m", 0.0))

    nav_h, nav_t = load_nav_pair_from_cfg(cfg)
    t_min = max(nav_h.t_gps.min(), nav_t.t_gps.min())
    t_max = min(nav_h.t_gps.max(), nav_t.t_gps.max())

    messages: List[Tuple[float, Any]] = []
    for i, (ts, msg) in enumerate(iter_bag_lidar(bag_path, topic)):
        if i % decimate != 0:
            continue
        messages.append((ts, msg))
    messages = filter_decimated_lidar_messages(messages, skip_start_sec, lidar_output_hz)
    if not messages:
        print("No lidar after filters.", file=sys.stderr)
        sys.exit(1)

    stamps = np.array([m[0] for m in messages], dtype=np.float64)
    warn_if_bag_starts_before_nav(stamps, time_offset, t_min)
    t_nav_arr = clamp_to_nav_range(ros_to_nav_time(stamps, time_offset), t_min, t_max)

    mode = cfg.get("enu_origin_mode", "host_first")
    if mode == "host_first":
        lat0, lon0, _ = unwrap_yaw_interp(nav_h, np.array([t_nav_arr[0]]))
        lat0_deg, lon0_deg = float(lat0[0]), float(lon0[0])
    else:
        raise ValueError(f"Unknown enu_origin_mode: {mode}")

    pipe = CooperativeLinkPipeline.from_yaml_cfg(cfg, nav_h, nav_t, lat0_deg, lon0_deg)
    pipe.attach_post_processor(cfg)
    link_cfg = pipe.link_cfg

    log_f = None
    if args.log:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "w", encoding="utf-8")

    vis = None
    o3d = None
    frame_geoms: List[Any] = []
    if not args.no_vis:
        import open3d as o3d_mod

        o3d = o3d_mod
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Cooperative link replay", width=1280, height=720)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.05, 0.05, 0.08])
        opt.point_size = 1.5

    prev_ros: float | None = None
    n_locked = 0

    for frame_idx, ((ros_t, msg), tn) in enumerate(zip(messages, t_nav_arr)):
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break
        pts = read_cloud_numpy(msg)
        out = pipe.process_frame(frame_idx, float(ros_t), float(tn), pts)
        if out.step.state == LinkState.LOCKED:
            n_locked += 1

        if log_f is not None:
            log_f.write(json.dumps(out.log_record) + "\n")

        line = (
            f"frame {frame_idx:5d} state={out.step.state.value:10s} "
            f"score={out.step.match_score:.3f} hits={out.step.consecutive_hits} "
            f"gated={out.n_gated} clusters={out.n_clusters} "
            f"r_prior={out.prior.r_prior_m:.2f}m"
        )
        if args.no_vis:
            print(line)
        elif o3d is not None and vis is not None:
            for g in frame_geoms:
                vis.remove_geometry(g, reset_bounding_box=False)
            frame_geoms.clear()
            frame_geoms = _make_vis_geometries(o3d, pts, out, link_cfg, gt_z, args.max_points)
            reset_box = frame_idx == 0
            for g in frame_geoms:
                vis.add_geometry(g, reset_bounding_box=reset_box)
            vis.poll_events()
            vis.update_renderer()

        if args.realtime and prev_ros is not None:
            dt = float(ros_t - prev_ros)
            if dt > 0:
                time.sleep(min(dt, args.sleep_cap))
        prev_ros = float(ros_t)

    if log_f is not None:
        log_f.close()
        print(f"Wrote log: {args.log}")

    total = min(len(messages), args.max_frames) if args.max_frames > 0 else len(messages)
    print(f"Done. locked_frames={n_locked}/{total}")

    if not args.no_vis and vis is not None:
        print("关闭窗口结束。")
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()
