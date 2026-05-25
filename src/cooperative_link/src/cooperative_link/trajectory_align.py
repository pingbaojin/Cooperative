"""Trajectory alignment via DTW + iterative weighted SVD (Identification_en port)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from cooperative_link.filter_config import CalibrationConfig


@dataclass
class AlignResult:
    R: np.ndarray  # (2,2)
    t: np.ndarray  # (2,)
    theta_rad: float
    cost: float
    valid: bool


def _normalize_tau(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def align_trajectories(
    TS: np.ndarray,
    TD: np.ndarray,
    cfg: CalibrationConfig,
) -> AlignResult:
    """
    Align candidate trajectory TD (2,M) to reference TS (2,N) in reference frame.

    Returns R, t such that TS ~= R @ TD + t (column vectors).
    """
    TS = np.asarray(TS, dtype=np.float64)
    TD = np.asarray(TD, dtype=np.float64)
    if TS.ndim != 2 or TD.ndim != 2 or TS.shape[0] != 2 or TD.shape[0] != 2:
        return AlignResult(
            R=np.eye(2), t=np.zeros(2), theta_rad=0.0, cost=float("inf"), valid=False
        )
    N, M = TS.shape[1], TD.shape[1]
    if N < 2 or M < 2:
        return AlignResult(
            R=np.eye(2), t=np.zeros(2), theta_rad=0.0, cost=float("inf"), valid=False
        )

    tau_s = _normalize_tau(N)
    tau_d = _normalize_tau(M)
    t = np.mean(TS, axis=1) - np.mean(TD, axis=1)
    R = np.eye(2, dtype=np.float64)

    Vs = np.diff(TS, axis=1)
    Vs = np.hstack([Vs, Vs[:, -1:]])
    Vd = np.diff(TD, axis=1)
    Vd = np.hstack([Vd, Vd[:, -1:]])

    last_cost = float("inf")
    norm_geom = 1.0

    for _iter in range(cfg.max_iter):
        TD_trans = R @ TD + t.reshape(2, 1)
        C_geom = np.sum((TS.T[:, None, :] - TD_trans.T[None, :, :]) ** 2, axis=2)
        C_time = np.abs(tau_s.reshape(N, 1) - tau_d.reshape(1, M))
        Vd_rot = R @ Vd
        vs_norm = np.sqrt(np.sum(Vs**2, axis=0)) + 1e-6
        vd_norm = np.sqrt(np.sum(Vd_rot**2, axis=0)) + 1e-6
        cos_sim = np.sum(Vs * Vd_rot, axis=0) / (vs_norm * vd_norm)
        C_vec_unit = 1.0 - cos_sim
        C_vector = np.tile(C_vec_unit.reshape(1, -1), (N, 1))

        if _iter == 0:
            norm_geom = max(float(np.mean(C_geom)), 0.1)

        C = (C_geom / norm_geom) + cfg.lambda_t * C_time + cfg.lambda_v * C_vector
        mask = C_time <= cfg.dtw_window
        C_dtw = C.copy()
        C_dtw[~mask] = float(np.max(C)) + 10.0

        D = np.full((N, M), np.inf, dtype=np.float64)
        Prev = np.zeros((N, M), dtype=np.int32)
        D[0, 0] = C_dtw[0, 0]
        for i in range(1, N):
            D[i, 0] = D[i - 1, 0] + C_dtw[i, 0] + cfg.skip_penalty
            Prev[i, 0] = 2
        for j in range(1, M):
            D[0, j] = D[0, j - 1] + C_dtw[0, j] + cfg.skip_penalty
            Prev[0, j] = 3
        for i in range(1, N):
            for j in range(1, M):
                opts = [
                    D[i - 1, j - 1] + C_dtw[i, j],
                    D[i - 1, j] + C_dtw[i, j] + cfg.skip_penalty,
                    D[i, j - 1] + C_dtw[i, j] + cfg.skip_penalty,
                ]
                idx = int(np.argmin(opts)) + 1
                D[i, j] = opts[idx - 1]
                Prev[i, j] = idx

        P = np.zeros((N, M), dtype=np.float64)
        ci, cj = N - 1, M - 1
        while ci >= 0 and cj >= 0:
            if mask[ci, cj]:
                P[ci, cj] = 1.0
            if ci == 0 and cj == 0:
                break
            if ci == 0:
                cj -= 1
            elif cj == 0:
                ci -= 1
            else:
                step = Prev[ci, cj]
                if step == 1:
                    ci -= 1
                    cj -= 1
                elif step == 2:
                    ci -= 1
                else:
                    cj -= 1
        P_norm = P / max(P.sum(), 1e-10)

        mu_s = TS @ P_norm.sum(axis=1)
        mu_d = TD @ P_norm.sum(axis=0)
        H = (TS - mu_s.reshape(2, 1)) @ P_norm @ (TD - mu_d.reshape(2, 1)).T
        U, _, Vt = np.linalg.svd(H)
        R_new = U @ Vt
        if np.linalg.det(R_new) < 0:
            Vt = Vt.copy()
            Vt[-1, :] *= -1
            R_new = U @ Vt
        t_new = mu_s - R_new @ mu_d

        R = R_new
        t = t_new
        curr_cost = float(np.sum(P_norm * C))
        if abs(last_cost - curr_cost) < cfg.tol:
            last_cost = curr_cost
            break
        last_cost = curr_cost

    theta = float(np.arctan2(R[1, 0], R[0, 0]))
    return AlignResult(R=R, t=t, theta_rad=theta, cost=last_cost, valid=True)
