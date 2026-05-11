"""Validate the Python ADMM translation against saved Matlab benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from admm_solver import dual_admm3c_test


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "X0_200.mat"
BENCHMARK_PATH = ROOT / "matlab_benchmark.mat"


def _mat_scalar(value):
    return float(np.asarray(value).squeeze())


def _print_comparison(name: str, python_value: float, matlab_value: float) -> None:
    diff = abs(python_value - matlab_value)
    print(f"  {name:<10} python={python_value:.12g} matlab={matlab_value:.12g} diff={diff:.3g}")


def main() -> None:
    data = loadmat(DATA_PATH)
    benchmark = loadmat(BENCHMARK_PATH, squeeze_me=True, struct_as_record=False)

    x0 = data["X0"]
    g = data["G"]

    print("Loaded input data")
    print(f"  X0 shape: {x0.shape}")
    print(f"  G shape:  {g.shape}")

    print("\nLoaded Matlab benchmark")
    print(f"  p_value: {_mat_scalar(benchmark['p_value']):.12g}")
    print(f"  v:       {_mat_scalar(benchmark['v']):.12g}")
    print(f"  epsilon: {_mat_scalar(benchmark['epsilon']):.12g}")

    admm_benchmark = benchmark["admm_benchmark"]
    print("\nADMM benchmark fields")
    print(f"  time:       {float(admm_benchmark.time):.12g}")
    print(f"  min_X:      {float(admm_benchmark.min_X):.12g}")
    print(f"  trace_X:    {float(admm_benchmark.trace_X):.12g}")
    print(f"  trace_GX:   {float(admm_benchmark.trace_GX):.12g}")
    print(f"  trace_GX0:  {float(admm_benchmark.trace_GX0):.12g}")
    print(f"  trace_X0X:  {float(admm_benchmark.trace_X0X):.12g}")

    print("\nRunning Python ADMM solver")
    s, x, p_value, elapsed, v = dual_admm3c_test(x0, g, 4, 1e-4, 100)

    print("\nPython vs Matlab ADMM")
    _print_comparison("p_value", p_value, _mat_scalar(benchmark["p_value"]))
    _print_comparison("v", v, _mat_scalar(benchmark["v"]))
    _print_comparison("min_X", float(np.min(x)), float(admm_benchmark.min_X))
    _print_comparison("trace_X", float(np.trace(x)), float(admm_benchmark.trace_X))
    _print_comparison("trace_GX", float(np.trace(g @ x)), float(admm_benchmark.trace_GX))
    _print_comparison("trace_GX0", float(np.trace(g @ x0)), float(admm_benchmark.trace_GX0))
    _print_comparison("trace_X0X", float(np.trace(x0 @ x)), float(admm_benchmark.trace_X0X))
    print(f"  elapsed    python={elapsed:.12g} matlab={float(admm_benchmark.time):.12g}")
    print(f"  S max diff {np.max(np.abs(s - benchmark['S'])):.3g}")
    print(f"  X max diff {np.max(np.abs(x - benchmark['X'])):.3g}")


if __name__ == "__main__":
    main()
