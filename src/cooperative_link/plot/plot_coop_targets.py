#!/usr/bin/env python3
"""
Plot cooperative_target topics from a coop_targets.bag (offline analysis).

Usage:
  source /opt/ros/noetic/setup.bash
  source /path/to/Cooperative/devel/setup.bash
  rosrun cooperative_link plot_coop_targets \\
    --bag /path/to/coop_targets.bag \\
    --out-dir /path/to/coop_targets_plots \\
    --partner-id 1
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rosbag


MAX_PLOT_POINTS = 8000
STATE_ORDER = ["unlocked", "locking", "locked", "reacquire"]
STATE_COLORS = {
    "unlocked": "#888888",
    "locking": "#f39c12",
    "locked": "#27ae60",
    "reacquire": "#e74c3c",
}


@dataclass
class PoseSeries:
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.t:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        return (
            np.asarray(self.t, dtype=np.float64),
            np.asarray(self.x, dtype=np.float64),
            np.asarray(self.y, dtype=np.float64),
        )


@dataclass
class PolarSeries:
    t: List[float] = field(default_factory=list)
    range_m: List[float] = field(default_factory=list)
    bearing_rad: List[float] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    valid: List[float] = field(default_factory=list)


@dataclass
class StateSeries:
    t: List[float] = field(default_factory=list)
    state: List[str] = field(default_factory=list)


def _downsample(t: np.ndarray, *ys: np.ndarray, max_pts: int = MAX_PLOT_POINTS):
    if t.size <= max_pts:
        return (t,) + ys
    idx = np.linspace(0, t.size - 1, max_pts, dtype=np.int64)
    return (t[idx],) + tuple(y[idx] for y in ys)


def _interp_xy_at(t_ref: np.ndarray, t_src: np.ndarray, x_src: np.ndarray, y_src: np.ndarray):
    if t_ref.size == 0 or t_src.size < 2:
        return np.array([]), np.array([])
    x_i = np.interp(t_ref, t_src, x_src)
    y_i = np.interp(t_ref, t_src, y_src)
    return x_i, y_i


def _error_vs_nav(
    t_nav: np.ndarray,
    x_nav: np.ndarray,
    y_nav: np.ndarray,
    t_other: np.ndarray,
    x_other: np.ndarray,
    y_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if t_nav.size < 2 or t_other.size < 2:
        return np.array([]), np.array([])
    t_min = max(t_nav.min(), t_other.min())
    t_max = min(t_nav.max(), t_other.max())
    mask = (t_nav >= t_min) & (t_nav <= t_max)
    t_m = t_nav[mask]
    if t_m.size == 0:
        return np.array([]), np.array([])
    x_i, y_i = _interp_xy_at(t_m, t_other, x_other, y_other)
    err = np.hypot(x_i - x_nav[mask], y_i - y_nav[mask])
    return t_m, err


def _is_pose_stamped(msg: object) -> bool:
    return hasattr(msg, "pose") and hasattr(getattr(msg, "pose", None), "position")


def _is_float32_array(msg: object) -> bool:
    data = getattr(msg, "data", None)
    return data is not None and not isinstance(data, str) and hasattr(data, "__len__")


def _is_string_msg(msg: object) -> bool:
    return isinstance(getattr(msg, "data", None), str)


def topic_names(partner_id: int) -> Dict[str, str]:
    pid = int(partner_id)
    return {
        "nav": f"/cooperative_target/{pid}/relative_pose_nav",
        "det": f"/cooperative_target/{pid}/relative_pose_det",
        "fused": f"/cooperative_target/{pid}/relative_pose",
        "filtered": f"/cooperative_target/{pid}/relative_pose_filtered",
        "polar": f"/cooperative_target/{pid}/relative_polar",
        "state": f"/cooperative_target/{pid}/link_state",
    }


def load_bag_series(bag_path: str, partner_id: int) -> Tuple[float, Dict[str, object], List[str]]:
    topics = topic_names(partner_id)
    nav = PoseSeries()
    det = PoseSeries()
    fused = PoseSeries()
    filtered = PoseSeries()
    polar = PolarSeries()
    state = StateSeries()
    warnings: List[str] = []

    t0: Optional[float] = None
    with rosbag.Bag(bag_path, "r") as bag:
        for _topic, msg, t in bag.read_messages(
            topics=list(topics.values()),
            raw=False,
        ):
            if t0 is None:
                t0 = float(t.to_sec())
            tr = float(t.to_sec()) - t0

            if _topic == topics["nav"] and _is_pose_stamped(msg):
                nav.t.append(tr)
                nav.x.append(float(msg.pose.position.x))
                nav.y.append(float(msg.pose.position.y))
            elif _topic == topics["det"] and _is_pose_stamped(msg):
                det.t.append(tr)
                det.x.append(float(msg.pose.position.x))
                det.y.append(float(msg.pose.position.y))
            elif _topic == topics["fused"] and _is_pose_stamped(msg):
                fused.t.append(tr)
                fused.x.append(float(msg.pose.position.x))
                fused.y.append(float(msg.pose.position.y))
            elif _topic == topics["filtered"] and _is_pose_stamped(msg):
                filtered.t.append(tr)
                filtered.x.append(float(msg.pose.position.x))
                filtered.y.append(float(msg.pose.position.y))
            elif _topic == topics["polar"] and _is_float32_array(msg):
                if len(msg.data) >= 4:
                    polar.t.append(tr)
                    polar.range_m.append(float(msg.data[0]))
                    polar.bearing_rad.append(float(msg.data[1]))
                    polar.score.append(float(msg.data[2]))
                    polar.valid.append(float(msg.data[3]))
            elif _topic == topics["state"] and _is_string_msg(msg):
                state.t.append(tr)
                state.state.append(str(msg.data).strip())

    if t0 is None:
        t0 = 0.0

    for name, series in [
        ("nav", nav),
        ("det", det),
        ("fused", fused),
        ("filtered", filtered),
        ("polar", polar),
        ("state", state),
    ]:
        if name == "polar":
            if not polar.t:
                warnings.append(f"no messages on {topics['polar']}")
        elif name == "state":
            if not state.t:
                warnings.append(f"no messages on {topics['state']}")
        else:
            s = series  # type: ignore
            if not s.t:
                warnings.append(f"no messages on {topics[name]}")

    data = {
        "nav": nav,
        "det": det,
        "fused": fused,
        "filtered": filtered,
        "polar": polar,
        "state": state,
        "topics": topics,
    }
    return t0, data, warnings


def plot_xy_trajectories(out_dir: Path, data: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    styles = [
        ("nav", "NAV reference", "#2ecc71", "-", 1.2),
        ("det", "Radar det", "#3498db", ".", 0.6),
        ("fused", "FSM fused", "#e67e22", "-", 1.0),
        ("filtered", "KF filtered", "#9b59b6", "-", 1.0),
    ]
    for key, label, color, marker, lw in styles:
        s: PoseSeries = data[key]
        t, x, y = s.to_arrays()
        if t.size == 0:
            continue
        t, x, y = _downsample(t, x, y)
        ax.plot(
            x,
            y,
            marker,
            label=label,
            color=color,
            linewidth=lw,
            markersize=2 if marker == "." else 0,
            alpha=0.85,
        )
    ax.set_xlabel("x (m, host lidar frame)")
    ax.set_ylabel("y (m)")
    ax.set_title("Relative pose trajectories")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "01_xy_trajectories.png", dpi=150)
    plt.close(fig)


def plot_error_vs_time(out_dir: Path, data: Dict[str, object]) -> None:
    nav: PoseSeries = data["nav"]
    t_nav, x_nav, y_nav = nav.to_arrays()
    if t_nav.size < 2:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    for key, label, color in [
        ("det", "det vs NAV", "#3498db"),
        ("fused", "fused vs NAV", "#e67e22"),
        ("filtered", "filtered vs NAV", "#9b59b6"),
    ]:
        s: PoseSeries = data[key]
        t_o, x_o, y_o = s.to_arrays()
        t_e, err = _error_vs_nav(t_nav, x_nav, y_nav, t_o, x_o, y_o)
        if t_e.size == 0:
            continue
        t_e, err = _downsample(t_e, err)
        ax.plot(t_e, err, "-", label=label, color=color, linewidth=0.9, alpha=0.9)

    ax.set_xlabel("time since bag start (s)")
    ax.set_ylabel("planar error vs NAV (m)")
    ax.set_title("Position error relative to NAV reference")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "02_error_vs_time.png", dpi=150)
    plt.close(fig)


def plot_polar(out_dir: Path, data: Dict[str, object]) -> None:
    polar: PolarSeries = data["polar"]
    if not polar.t:
        return
    t = np.asarray(polar.t, dtype=np.float64)
    r = np.asarray(polar.range_m, dtype=np.float64)
    b_deg = np.rad2deg(np.asarray(polar.bearing_rad, dtype=np.float64))
    t, r, b_deg = _downsample(t, r, b_deg)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(t, r, color="#2980b9", linewidth=0.8)
    axes[0].set_ylabel("range (m)")
    axes[0].set_title("Fused polar (from relative_polar)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, b_deg, color="#8e44ad", linewidth=0.8)
    axes[1].set_xlabel("time since bag start (s)")
    axes[1].set_ylabel("bearing (deg)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_polar_range_bearing.png", dpi=150)
    plt.close(fig)


def plot_match_score_valid(out_dir: Path, data: Dict[str, object]) -> None:
    polar: PolarSeries = data["polar"]
    if not polar.t:
        return
    t = np.asarray(polar.t, dtype=np.float64)
    score = np.asarray(polar.score, dtype=np.float64)
    valid = np.asarray(polar.valid, dtype=np.float64)
    t, score, valid = _downsample(t, score, valid)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, score, color="#16a085", linewidth=0.8, label="match_score")
    ax1.set_ylabel("match_score")
    ax1.set_xlabel("time since bag start (s)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(t, valid, color="#c0392b", linewidth=0.6, alpha=0.7, label="valid")
    ax2.set_ylabel("valid (0/1)")
    ax2.set_ylim(-0.05, 1.15)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
    ax1.set_title("Association score and output valid flag")
    fig.tight_layout()
    fig.savefig(out_dir / "04_match_score_valid.png", dpi=150)
    plt.close(fig)


def plot_link_state(out_dir: Path, data: Dict[str, object]) -> None:
    st: StateSeries = data["state"]
    if not st.t:
        return
    t = np.asarray(st.t, dtype=np.float64)
    codes = np.array(
        [STATE_ORDER.index(s) if s in STATE_ORDER else -1 for s in st.state],
        dtype=np.float64,
    )
    colors = [
        STATE_COLORS.get(s, "#333333") if s in STATE_COLORS else "#333333"
        for s in st.state
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(t, codes, where="post", color="#555555", linewidth=0.6, alpha=0.5)
    for name in STATE_ORDER:
        mask = np.array([s == name for s in st.state])
        if not np.any(mask):
            continue
        ax.scatter(
            t[mask],
            codes[mask],
            c=STATE_COLORS[name],
            s=6,
            label=name,
            alpha=0.8,
        )
    ax.set_yticks(range(len(STATE_ORDER)))
    ax.set_yticklabels(STATE_ORDER)
    ax.set_xlabel("time since bag start (s)")
    ax.set_title("Link FSM state")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_link_state.png", dpi=150)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    bag_path: str,
    partner_id: int,
    data: Dict[str, object],
    warnings: List[str],
    errors: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    nav: PoseSeries = data["nav"]
    st: StateSeries = data["state"]
    duration = 0.0
    for key in ("nav", "det", "fused", "filtered", "polar", "state"):
        s = data[key]
        ts = getattr(s, "t", [])
        if ts:
            duration = max(duration, max(ts))

    locked_frac = 0.0
    if st.state:
        locked_frac = sum(1 for x in st.state if x == "locked") / len(st.state)

    lines = [
        f"bag: {bag_path}",
        f"partner_id: {partner_id}",
        f"duration_s: {duration:.1f}",
        "",
        "message counts:",
    ]
    for key in ("nav", "det", "fused", "filtered", "polar", "state"):
        s = data[key]
        lines.append(f"  {key}: {len(s.t)}")
    lines.append("")
    lines.append(f"link_state locked ratio: {locked_frac:.3f}")
    lines.append("")
    lines.append("planar error vs NAV (median / P95 m), on overlapping time:")
    for name, (t_e, err) in errors.items():
        if err.size == 0:
            lines.append(f"  {name}: n/a")
        else:
            lines.append(
                f"  {name}: median={float(np.median(err)):.4f} "
                f"p95={float(np.percentile(err, 95)):.4f} n={err.size}"
            )
    if warnings:
        lines.append("")
        lines.append("warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot cooperative_target bag analysis")
    parser.add_argument("--bag", type=str, required=True, help="path to coop_targets.bag")
    parser.add_argument("--out-dir", type=str, required=True, help="output directory for PNGs")
    parser.add_argument("--partner-id", type=int, default=1, help="cooperative partner id")
    args = parser.parse_args()

    bag_path = str(Path(args.bag).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {bag_path} (partner_id={args.partner_id})...")
    _t0, data, warnings = load_bag_series(bag_path, args.partner_id)
    for w in warnings:
        print(f"WARN: {w}")

    nav: PoseSeries = data["nav"]
    t_nav, x_nav, y_nav = nav.to_arrays()
    err_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key in ("det", "fused", "filtered"):
        s: PoseSeries = data[key]
        t_o, x_o, y_o = s.to_arrays()
        err_stats[key] = _error_vs_nav(t_nav, x_nav, y_nav, t_o, x_o, y_o)

    print(f"Writing plots to {out_dir}...")
    plot_xy_trajectories(out_dir, data)
    plot_error_vs_time(out_dir, data)
    plot_polar(out_dir, data)
    plot_match_score_valid(out_dir, data)
    plot_link_state(out_dir, data)
    write_summary(out_dir, bag_path, args.partner_id, data, warnings, err_stats)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
