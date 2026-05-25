"""Tests for trajectory alignment (Identification_en style)."""

import unittest

import numpy as np

from cooperative_link.filter_config import CalibrationConfig
from cooperative_link.trajectory_align import align_trajectories


class TestTrajectoryAlign(unittest.TestCase):
    def test_aligned_trajectory_lower_cost(self) -> None:
        cfg = CalibrationConfig()
        t = np.linspace(0, np.pi, 50)
        TS = np.vstack([np.cumsum(np.cos(t) * 0.5), np.cumsum(np.sin(t) * 0.5)])
        theta = np.deg2rad(30)
        R_true = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        t_vec = np.array([5.0, 2.0])
        TD_good = R_true @ (TS - t_vec.reshape(2, 1))
        TD_bad = np.vstack([np.linspace(-2, -4, 50), np.linspace(-1.5, 3.5, 50)])
        r_good = align_trajectories(TS, TD_good, cfg)
        r_bad = align_trajectories(TS, TD_bad, cfg)
        self.assertTrue(r_good.valid and r_bad.valid)
        self.assertLess(r_good.cost, r_bad.cost)

    def test_recover_transform_matlab_convention(self) -> None:
        """TD = R_true * (TS - t_true) as in Identification_en.m."""
        cfg = CalibrationConfig(max_iter=50)
        theta = np.deg2rad(30)
        R_true = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        t_true = np.array([5.0, 2.0])
        N = 40
        TS = np.random.randn(2, N)
        TD = R_true @ (TS - t_true.reshape(2, 1))
        r = align_trajectories(TS, TD, cfg)
        self.assertTrue(r.valid)
        aligned = r.R @ TD + r.t.reshape(2, 1)
        rmse = float(np.sqrt(np.mean((aligned - TS) ** 2)))
        self.assertLess(rmse, 0.5)


if __name__ == "__main__":
    unittest.main()
