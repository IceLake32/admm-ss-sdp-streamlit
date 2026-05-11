"""Python translation target for the Matlab ADMM solver path.

Matlab source files:
    - ../proj_psd_largescale.m
    - ../dual_admm3c_test.m
    - ../dual_admm3c.m
"""

from __future__ import annotations

import time

import numpy as np


def proj_psd_largescale(is_real: bool, x: np.ndarray, t: float = 1.0):
    """Project a symmetric matrix onto the positive semidefinite cone.

    This is the first-pass dense translation target for Matlab's
    `proj_psd_largescale`. The Matlab helper uses `eigs` adaptively for large
    matrices; this version starts with a dense eigendecomposition so validation
    is simpler and deterministic for the 200x200 sample data.
    """
    x = np.asarray(x)
    sym_x = (x + x.T.conj()) / 2
    if is_real:
        sym_x = np.real(sym_x)

    eig_tol = 1e-10

    if t > 0:
        evals, evecs = np.linalg.eigh(sym_x)
        keep = evals > eig_tol
        if not np.any(keep):
            projected = np.zeros_like(sym_x)
        else:
            weighted = evecs[:, keep] * np.sqrt(evals[keep])
            projected = weighted @ weighted.T.conj()
            if is_real:
                projected = np.real(projected)
        return 0.0, projected

    min_eval = np.linalg.eigvalsh(sym_x).min()
    cone_value = np.inf if min_eval < -10 * np.finfo(float).eps else 0.0
    return cone_value, sym_x


def dual_admm3c_test(
    x0: np.ndarray,
    g: np.ndarray,
    k: int,
    eps: float,
    p_iter: int,
):
    """Wrapper matching `dual_admm3c_test.m`."""
    n = x0.shape[0]
    costmax = _trace_product(g, x0)
    max_iter = 10000

    eta = np.zeros((n, n), dtype=x0.dtype)
    v = 0.01
    s = np.zeros((n, n), dtype=x0.dtype)
    x1 = x0.copy()
    x2 = x0.copy()
    sigma = 0.01
    tau = 1.618

    start = time.perf_counter()
    s, x, p_value, v = dual_admm3c(
        x0,
        g,
        k,
        costmax,
        eta,
        v,
        s,
        x1,
        x2,
        sigma,
        tau,
        max_iter,
        eps,
        p_iter,
    )
    elapsed = _elapsed_since(start)
    return s, x, p_value, elapsed, v


def dual_admm3c(
    x0: np.ndarray,
    g: np.ndarray,
    k: int,
    costmax: float,
    eta: np.ndarray,
    v: float,
    s: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    sigma: float,
    tau: float,
    max_iter: int,
    eps: float,
    p_iter: int,
):
    """Core solver matching `dual_admm3c.m`."""
    n = x0.shape[0]
    p_value = 0.0

    ones = np.ones(n, dtype=x0.dtype)
    eye = np.eye(n, dtype=x0.dtype)

    sum_alpha = (np.sum(eta + s) - np.trace(eta + s + v * g) + k - n) / (n - 1)
    z = (np.trace(s + v * g + eta) - k - sum_alpha) / n
    alpha = (2 * ((s + eta) @ ones) - 2 - sum_alpha - 2 * z) / n
    a_alpha = (np.outer(ones, alpha) + np.outer(alpha, ones)) / 2
    beta = 0.5 * (x0 - s - v * g + eta + a_alpha + z * eye)

    norm_g = np.linalg.norm(g, "fro") ** 2

    for i in range(1, max_iter + 1):
        t_1 = v * g - z * eye - a_alpha + beta - x0
        m = -t_1 - x1 / sigma

        _, s = proj_psd_largescale(True, -m, 1)
        s = s + m
        s = (s + s.T) / 2

        eta = np.maximum(beta - x2 / sigma, 0)

        sum_alpha = (np.sum(eta + s) - np.trace(eta + s + v * g) + k - n) / (n - 1)
        z = (np.trace(s + v * g + eta) - k - sum_alpha) / n
        alpha = (2 * ((s + eta) @ ones) - 2 - sum_alpha - 2 * z) / n
        a_alpha = (np.outer(ones, alpha) + np.outer(alpha, ones)) / 2
        beta = 0.5 * (x0 - s - v * g + eta + a_alpha + z * eye)

        t_2 = s - z * eye - a_alpha + beta - x0
        v_candidate = float(
            (costmax - _trace_product(g, x1) - sigma * _trace_product(g, t_2))
            / (sigma * norm_g)
        )
        v = max(v_candidate, 0.0)

        sum_alpha = (np.sum(eta + s) - np.trace(eta + s + v * g) + k - n) / (n - 1)
        z = (np.trace(s + v * g + eta) - k - sum_alpha) / n
        alpha = (2 * ((s + eta) @ ones) - 2 - sum_alpha - 2 * z) / n
        a_alpha = (np.outer(ones, alpha) + np.outer(alpha, ones)) / 2
        beta = 0.5 * (x0 - s - v * g + eta + a_alpha + z * eye)

        x1 = x1 + tau * sigma * (s + v * g + beta - a_alpha - z * eye - x0)
        x2 = x2 + tau * sigma * (eta - beta)

        p_value = _trace_product(x0, x1)
        dual_value = k * z + np.sum(alpha) - v * costmax

        eta_e = 0.0
        eta_i = abs(min(float(np.real(_trace_product(g, x1) - costmax)), 0.0)) / (
            1 + abs(costmax)
        )
        eta_p = max(eta_e, eta_i)

        eta_d = np.linalg.norm(s + v * g + beta - a_alpha - z * eye - x0, "fro") / (
            1 + np.sqrt(k)
        )

        stop_eta = max(float(eta_d), float(eta_p))
        if i % p_iter == 0 or stop_eta <= eps:
            print(f"after {i} iteration, p_value is {p_value:f}")
            print(f"after {i} iteration, dual value is {-dual_value:f}")
            print(f"the dual violation is {eta_d:f}; the primal violation is {eta_p:f}")
            print(f"primal nonnegativity violation is {abs(np.min(x1)):f}")
            print()

        if stop_eta <= eps:
            print(f"the multiplier value sigma is {sigma:f}")
            break

    return s, x1, p_value, v


def _trace_product(a: np.ndarray, b: np.ndarray) -> float:
    """Return trace(a @ b), matching Matlab's trace(A * B)."""
    return float(np.trace(a @ b))


def _elapsed_since(start: float) -> float:
    return time.perf_counter() - start
