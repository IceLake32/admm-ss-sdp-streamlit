"""Run short CG checks for comparing against Matlab's early iterations."""

from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat

from cg_solver import cg_ss_test


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "X0_200.mat"


def run(mode: str) -> None:
    data = loadmat(DATA_PATH)
    print(f"\n=== CG short run: eigen_mode={mode} ===")
    cg_ss_test(
        data["X0"],
        data["G"],
        4,
        max_iter=4,
        p_iter=1,
        stop_on_convergence=False,
        eigen_mode=mode,
        print_eigvals=True,
    )


def main() -> None:
    run("eigsh")
    run("eigs")
    run("dense")


if __name__ == "__main__":
    main()
