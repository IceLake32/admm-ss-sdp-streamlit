"""Small API layer for running the clustering SDP solvers from UI code."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from scipy.io import loadmat, savemat

from admm_solver import dual_admm3c_test
from cg_solver import cg_ss_test


@dataclass
class SolverResult:
    solver: str
    objective: float
    elapsed: float
    metrics: dict[str, float]
    output_name: str
    output_matrix: np.ndarray
    extra_outputs: dict[str, np.ndarray | float]


def load_problem(file: str | Path | BinaryIO) -> tuple[np.ndarray, np.ndarray]:
    """Load X0 and G from a Matlab .mat file."""
    data = loadmat(file)
    missing = [name for name in ("X0", "G") if name not in data]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"Uploaded .mat file must contain variables: X0 and G. Missing: {missing_names}.")

    x0 = np.asarray(data["X0"], dtype=float)
    g = np.asarray(data["G"], dtype=float)
    _validate_problem_matrices(x0, g)
    return x0, g


def infer_cluster_count(x0: np.ndarray, tol: float = 1e-6) -> int:
    """Infer K from a clustering matrix using trace(X0) = K."""
    trace_x0 = float(np.trace(x0))
    k = int(round(trace_x0))
    if k < 2:
        raise ValueError(f"`trace(X0)` must identify at least 2 clusters. Got {trace_x0:.12g}.")
    if abs(trace_x0 - k) > tol:
        raise ValueError(
            "`trace(X0)` should be an integer equal to the number of clusters. "
            f"Got trace(X0) = {trace_x0:.12g}."
        )
    return k


def run_admm(
    x0: np.ndarray,
    g: np.ndarray,
    k: int,
    eps: float = 1e-4,
    p_iter: int = 100,
) -> SolverResult:
    s, x, p_value, elapsed, v = dual_admm3c_test(x0, g, k, eps, p_iter)
    metrics = _matrix_metrics("X", x0, g, x)
    metrics["v"] = float(v)
    return SolverResult(
        solver="ADMM",
        objective=float(p_value),
        elapsed=float(elapsed),
        metrics=metrics,
        output_name="X",
        output_matrix=x,
        extra_outputs={"S": s, "v": float(v)},
    )


def run_cg(
    x0: np.ndarray,
    g: np.ndarray,
    k: int,
    max_iter: int = 500,
    p_iter: int = 50,
    eigen_mode: str = "eigsh",
) -> SolverResult:
    p, epsilon, obj_value, elapsed = cg_ss_test(
        x0,
        g,
        k,
        max_iter,
        p_iter,
        stop_on_convergence=True,
        eigen_mode=eigen_mode,
    )
    metrics = _matrix_metrics("P", x0, g, p)
    nonzero_obj = obj_value[np.flatnonzero(obj_value)]
    if nonzero_obj.size:
        metrics["last_recorded_objective"] = float(nonzero_obj[-1])
    return SolverResult(
        solver="CG",
        objective=float(epsilon),
        elapsed=float(elapsed),
        metrics=metrics,
        output_name="P",
        output_matrix=p,
        extra_outputs={"obj_value": obj_value},
    )


def result_to_mat_bytes(result: SolverResult) -> bytes:
    """Serialize solver outputs into an in-memory .mat file."""
    payload: dict[str, np.ndarray | float] = {
        result.output_name: result.output_matrix,
        "objective": result.objective,
        "elapsed": result.elapsed,
    }
    payload.update(result.extra_outputs)
    for key, value in result.metrics.items():
        payload[f"metric_{key}"] = float(value)

    buffer = BytesIO()
    savemat(buffer, payload)
    return buffer.getvalue()


def _validate_problem_matrices(x0: np.ndarray, g: np.ndarray) -> None:
    if x0.ndim != 2 or g.ndim != 2:
        raise ValueError("X0 and G must both be 2D matrices.")
    if x0.shape[0] != x0.shape[1]:
        raise ValueError(f"X0 must be square. Got shape {x0.shape}.")
    if g.shape[0] != g.shape[1]:
        raise ValueError(f"G must be square. Got shape {g.shape}.")
    if x0.shape != g.shape:
        raise ValueError(f"X0 and G must have the same shape. Got {x0.shape} and {g.shape}.")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(g)):
        raise ValueError("X0 and G must contain only finite numeric values.")


def _matrix_metrics(name: str, x0: np.ndarray, g: np.ndarray, matrix: np.ndarray) -> dict[str, float]:
    return {
        f"min_{name}": float(np.min(matrix)),
        f"trace_{name}": float(np.trace(matrix)),
        f"trace_G{name}": float(np.trace(g @ matrix)),
        "trace_GX0": float(np.trace(g @ x0)),
        f"trace_X0{name}": float(np.trace(x0 @ matrix)),
    }
