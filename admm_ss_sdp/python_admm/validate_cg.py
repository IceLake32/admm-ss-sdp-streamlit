"""Validate the Python conditional-gradient translation against Matlab."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from cg_solver import cg_ss_test, prepare_cg_inputs


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "X0_200.mat"
BENCHMARK_PATH = ROOT / "matlab_benchmark.mat"


def _mat_scalar(value):
    return float(np.asarray(value).squeeze())


def _print_comparison(name: str, python_value: float, matlab_value: float) -> None:
    diff = abs(python_value - matlab_value)
    print(f"  {name:<13} python={python_value:.12g} matlab={matlab_value:.12g} diff={diff:.3g}")


def main() -> None:
    data = loadmat(DATA_PATH)
    benchmark = loadmat(BENCHMARK_PATH, squeeze_me=True, struct_as_record=False)

    x0 = data["X0"]
    g = data["G"]
    cg_benchmark = benchmark["cg_benchmark"]

    print("Loaded input data")
    print(f"  X0 shape: {x0.shape}")
    print(f"  G shape:  {g.shape}")

    print("\nLoaded Matlab CG benchmark")
    print(f"  epsilon:       {_mat_scalar(benchmark['epsilon']):.12g}")
    print(f"  time:          {float(cg_benchmark.time):.12g}")
    print(f"  min_P:         {float(cg_benchmark.min_P):.12g}")
    print(f"  trace_P:       {float(cg_benchmark.trace_P):.12g}")
    print(f"  trace_GP:      {float(cg_benchmark.trace_GP):.12g}")
    print(f"  trace_GX0:     {float(cg_benchmark.trace_GX0):.12g}")
    print(f"  trace_X0P:     {float(cg_benchmark.trace_X0P):.12g}")
    print(f"  obj_value_end: {float(cg_benchmark.obj_value_end):.12g}")

    print("\nPreparing Python CG inputs")
    g_half, x0_half, costmax = prepare_cg_inputs(x0, g)
    print(f"  G_half shape:  {g_half.shape}")
    print(f"  X0_half shape: {x0_half.shape}")
    print(f"  costmax:       {costmax:.12g}")

    print("\nRunning Python CG solver")
    p, epsilon, obj_value, elapsed = cg_ss_test(
        x0, g, 4, 5000, 50, stop_on_convergence=False, eigen_mode="eigs"
    )
    last_obj = obj_value[np.flatnonzero(obj_value)[-1]]

    print("\nPython vs Matlab CG")
    _print_comparison("epsilon", epsilon, _mat_scalar(benchmark["epsilon"]))
    _print_comparison("min_P", float(np.min(p)), float(cg_benchmark.min_P))
    _print_comparison("trace_P", float(np.trace(p)), float(cg_benchmark.trace_P))
    _print_comparison("trace_GP", float(np.trace(g @ p)), float(cg_benchmark.trace_GP))
    _print_comparison("trace_GX0", float(np.trace(g @ x0)), float(cg_benchmark.trace_GX0))
    _print_comparison("trace_X0P", float(np.trace(x0 @ p)), float(cg_benchmark.trace_X0P))
    _print_comparison("obj_end", float(last_obj), float(cg_benchmark.obj_value_end))
    print(f"  elapsed       python={elapsed:.12g} matlab={float(cg_benchmark.time):.12g}")
    print(f"  P max diff    {np.max(np.abs(p - benchmark['P'])):.3g}")


if __name__ == "__main__":
    main()
