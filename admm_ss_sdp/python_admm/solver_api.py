"""Small API layer for running the clustering SDP solvers from UI code."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
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


def load_problem(file: str | Path | BinaryIO, filename: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load X0 and G from a supported problem file."""
    suffix = _problem_suffix(file, filename)
    if suffix == ".mat":
        data = loadmat(file)
    elif suffix == ".npz":
        data = np.load(file)
    elif suffix == ".csv":
        return _load_csv_problem(file)
    else:
        raise ValueError("Unsupported file format. Upload a `.mat`, `.npz`, or `.csv` file.")

    return _problem_arrays_from_mapping(data, suffix)


def _problem_arrays_from_mapping(data, suffix: str) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in ("X0", "G") if name not in data]
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"Uploaded {suffix} file must contain variables: X0 and G. Missing: {missing_names}.")

    x0 = np.asarray(data["X0"], dtype=float)
    g = np.asarray(data["G"], dtype=float)
    _validate_problem_matrices(x0, g)
    return x0, g


def _load_csv_problem(file: str | Path | BinaryIO) -> tuple[np.ndarray, np.ndarray]:
    text = _read_problem_text(file)
    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Uploaded .csv file must include a header row.")

    label_column = _find_label_column(reader.fieldnames)
    feature_columns = [name for name in reader.fieldnames if name != label_column]
    if not feature_columns:
        raise ValueError("Uploaded .csv file must include at least one numeric feature column.")

    labels: list[str] = []
    features: list[list[float]] = []
    for row_number, row in enumerate(reader, start=2):
        label = (row.get(label_column) or "").strip()
        if not label:
            raise ValueError(f"Missing label in row {row_number}.")
        labels.append(label)

        feature_row: list[float] = []
        for column in feature_columns:
            value = (row.get(column) or "").strip()
            if not value:
                raise ValueError(f"Missing numeric value for column `{column}` in row {row_number}.")
            try:
                feature_row.append(float(value))
            except ValueError as exc:
                raise ValueError(
                    f"Column `{column}` must contain numeric values. "
                    f"Could not parse `{value}` in row {row_number}."
                ) from exc
        features.append(feature_row)

    if len(labels) < 2:
        raise ValueError("Uploaded .csv file must contain at least two data rows.")

    return _problem_from_features_and_labels(np.asarray(features, dtype=float), np.asarray(labels, dtype=str))


def _read_problem_text(file: str | Path | BinaryIO) -> str:
    if isinstance(file, (str, Path)):
        return Path(file).read_text(encoding="utf-8-sig")

    raw = file.read()
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8-sig")


def _find_label_column(fieldnames: list[str]) -> str:
    for name in fieldnames:
        if name.lower() == "label":
            return name
    raise ValueError("Uploaded .csv file must include a `label` column.")


def _problem_from_features_and_labels(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    if unique_labels.size < 2:
        raise ValueError("CSV labels must contain at least two clusters.")

    cluster_sizes = np.bincount(inverse)
    same_cluster = inverse[:, None] == inverse[None, :]
    x0 = same_cluster.astype(float) / cluster_sizes[inverse][:, None]

    centered = features - features.mean(axis=0, keepdims=True)
    g = centered @ centered.T
    _validate_problem_matrices(x0, g)
    return x0, g


def _problem_suffix(file: str | Path | BinaryIO, filename: str | None) -> str:
    if filename:
        return Path(filename).suffix.lower()
    if isinstance(file, (str, Path)):
        return Path(file).suffix.lower()
    return ".mat"


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
