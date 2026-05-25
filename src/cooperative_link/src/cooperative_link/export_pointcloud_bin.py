"""Export sensor_msgs/PointCloud2 from ROS1 bag to float32 Nx5 .bin (x,y,z,intensity,ring)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Generator, List, Tuple

import numpy as np


def read_cloud_numpy(msg: Any) -> np.ndarray:
    """
    Return (N, 5) float32 [x,y,z,intensity,ring].

    Prefer sensor_msgs.point_cloud2.read_points; else parse PointCloud2 layout.
    """
    try:
        from sensor_msgs import point_cloud2

        try:
            rows = list(
                point_cloud2.read_points(
                    msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
                )
            )
            arr = np.array([[r[0], r[1], r[2], float(r[3])] for r in rows], dtype=np.float32)
        except Exception:
            rows = list(
                point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            )
            arr = np.array([[r[0], r[1], r[2], 0.0] for r in rows], dtype=np.float32)
        ring = np.zeros((arr.shape[0], 1), dtype=np.float32)
        return np.hstack([arr[:, :4], ring]).astype(np.float32)
    except Exception:
        pass

    fields = msg.fields
    off = {f.name: f.offset for f in fields}
    if not all(k in off for k in ("x", "y", "z")):
        raise ValueError("PointCloud2 must contain x,y,z fields with known offsets")
    step = msg.point_step
    data = msg.data
    n = msg.width * msg.height
    out = np.zeros((n, 5), dtype=np.float32)
    inten_off = off.get("intensity")
    for i in range(n):
        base = i * step
        x, y, z = struct.unpack_from("fff", data, base + off["x"])
        inte = 0.0
        if inten_off is not None:
            try:
                (inte,) = struct.unpack_from("f", data, base + inten_off)
            except struct.error:
                try:
                    (iu,) = struct.unpack_from("H", data, base + inten_off)
                    inte = float(iu) / 255.0
                except struct.error:
                    inte = 0.0
        out[i] = (x, y, z, float(inte), 0.0)
    mask = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & np.isfinite(out[:, 2])
    return out[mask]


def write_lidar_bin(path: Path, points_n5: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points_n5, dtype=np.float32).reshape(-1, 5)
    pts.tofile(str(path))


def iter_bag_lidar(bag_path: str, topic: str) -> Generator[Tuple[float, Any], None, None]:
    import rosbag

    bag = rosbag.Bag(bag_path, "r")
    try:
        for _, msg, t in bag.read_messages(topics=[topic]):
            yield t.to_sec(), msg
    finally:
        bag.close()
