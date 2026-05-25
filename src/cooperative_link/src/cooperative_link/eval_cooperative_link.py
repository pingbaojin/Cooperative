#!/usr/bin/env python3
"""
Offline metrics for cooperative link JSONL logs (no rotated IoU).

用法:
  python3 data_generate/replay_dynamic_assoc.py --config data_generate/example_config.yaml \\
    --no_vis --log /tmp/cooperative_link.jsonl
  python3 data_generate/eval_cooperative_link.py --config data_generate/example_config.yaml \\
    --log /tmp/cooperative_link.jsonl
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
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from cooperative_link.config import load_link_config
from cooperative_link.dynamic_detect import wrap_rad
from cooperative_link.lidar_filter import filter_decimated_lidar_messages
from cooperative_link.export_pointcloud_bin import iter_bag_lidar
from cooperative_link.nav_pose import load_nav_pair_from_cfg, unwrap_yaw_interp
from cooperative_link.time_sync import clamp_to_nav_range, ros_to_nav_time


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_with_config(cfg: Dict[str, Any], records: List[Dict[str, Any]]) -> Dict[str, float]:
    link_cfg = load_link_config(cfg)
    if not records:
        return {}
    gate_ok = 0
    err_locked: List[float] = []
    first_lock = None
    locked_count = 0
    for i, r in enumerate(records):
        nav = np.array([r["center_nav_x"], r["center_nav_y"]], dtype=np.float64)
        rp, bp = r["r_prior_m"], r["bearing_prior_rad"]
        r_ok = abs(np.hypot(nav[0], nav[1]) - rp) <= link_cfg.range_tol_m
        b_ok = abs(wrap_rad(np.arctan2(nav[1], nav[0]) - bp)) <= np.deg2rad(link_cfg.bearing_tol_deg)
        if r_ok and b_ok:
            gate_ok += 1
        if r.get("locked"):
            locked_count += 1
            if first_lock is None:
                first_lock = i
            fused = np.array([r["center_fused_x"], r["center_fused_y"]], dtype=np.float64)
            err_locked.append(float(np.linalg.norm(fused - nav)))
    n = len(records)
    out = {
        "n_frames": n,
        "gate_recall": gate_ok / n if n else 0.0,
        "lock_success": 1.0 if first_lock is not None else 0.0,
        "frames_to_lock": first_lock if first_lock is not None else -1,
        "locked_frame_ratio": locked_count / n if n else 0.0,
    }
    if err_locked:
        arr = np.asarray(err_locked, dtype=np.float64)
        out["locked_xy_err_median_m"] = float(np.median(arr))
        out["locked_xy_err_p95_m"] = float(np.percentile(arr, 95))
    if first_lock is not None and n > 0:
        ros0 = records[0].get("ros_t", 0.0)
        ros_lock = records[first_lock].get("ros_t", ros0)
        out["time_to_lock_sec"] = float(ros_lock - ros0)
    return out


def regenerate_log(cfg: Dict[str, Any], bag_path: str | None, max_frames: int) -> List[Dict[str, Any]]:
    """Run pipeline without viz to produce records (same as replay --no_vis --log)."""
    from cooperative_link.pipeline import CooperativeLinkPipeline
    from cooperative_link.export_pointcloud_bin import read_cloud_numpy

    topic = cfg["lidar_topic"]
    time_offset = float(cfg.get("time_offset_sec", 0.0))
    decimate = max(1, int(cfg.get("lidar_decimate", 1)))
    skip_start_sec = float(cfg.get("skip_start_sec", 0.0))
    lidar_output_hz = float(cfg.get("lidar_output_hz", 0.0))
    bag = bag_path or cfg["bag_path"]

    nav_h, nav_t = load_nav_pair_from_cfg(cfg)
    t_min = max(nav_h.t_gps.min(), nav_t.t_gps.min())
    t_max = min(nav_h.t_gps.max(), nav_t.t_gps.max())

    messages = []
    for i, (ts, msg) in enumerate(iter_bag_lidar(bag, topic)):
        if i % decimate != 0:
            continue
        messages.append((ts, msg))
    messages = filter_decimated_lidar_messages(messages, skip_start_sec, lidar_output_hz)
    stamps = np.array([m[0] for m in messages], dtype=np.float64)
    t_nav_arr = clamp_to_nav_range(ros_to_nav_time(stamps, time_offset), t_min, t_max)
    lat0, lon0, _ = unwrap_yaw_interp(nav_h, np.array([t_nav_arr[0]]))
    pipe = CooperativeLinkPipeline.from_yaml_cfg(cfg, nav_h, nav_t, float(lat0[0]), float(lon0[0]))

    records: List[Dict[str, Any]] = []
    for frame_idx, ((ros_t, msg), tn) in enumerate(zip(messages, t_nav_arr)):
        if max_frames > 0 and frame_idx >= max_frames:
            break
        pts = read_cloud_numpy(msg)
        out = pipe.process_frame(frame_idx, float(ros_t), float(tn), pts)
        records.append(out.log_record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cooperative link JSONL log")
    default_cfg = _default_config_path()
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument("--log", type=str, required=True, help="JSONL from replay_dynamic_assoc")
    parser.add_argument("--regenerate", action="store_true", help="re-run pipeline instead of reading log")
    parser.add_argument("--bag", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.regenerate:
        records = regenerate_log(cfg, args.bag, args.max_frames)
    else:
        records = []
        with open(args.log, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    metrics = eval_with_config(cfg, records)
    print("Cooperative link evaluation")
    print("-" * 40)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float) and k not in ("n_frames", "frames_to_lock"):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
