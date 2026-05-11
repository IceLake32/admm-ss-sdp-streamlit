"""Python translation target for the Matlab conditional-gradient solver path.

Matlab source files:
    - ../cg_ss_test.m
    - ../cg_ss.m
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs, eigsh, lobpcg


def cg_ss_test(
    x0: np.ndarray,
    g: np.ndarray,
    k: int,
    max_iter: int,
    p_iter: int,
    stop_on_convergence: bool = True,
    eigen_mode: str = "eigsh",
    print_eigvals: bool = False,
):
    """Wrapper matching `cg_ss_test.m`."""
    x2 = -0.01
    n_inner = 10

    v_max = eigsh(g, k=1, which="LA", return_eigenvectors=False)[0]
    v_max = v_max / 10
    g_norm = g / v_max

    g_evals, g_evecs = np.linalg.eigh(g_norm)
    g_keep = g_evals > 1e-10
    g_half = g_evecs[:, g_keep] * np.sqrt(g_evals[g_keep])

    x0_evals, x0_evecs = np.linalg.eigh(x0)
    x0_keep = x0_evals > 1e-10
    x0_half = x0_evecs[:, x0_keep] * np.sqrt(x0_evals[x0_keep])

    temp = g_half.T @ x0_half
    costmax = np.linalg.norm(temp, "fro") ** 2

    return cg_ss(
        x0,
        g_half,
        x0_half,
        k,
        costmax,
        x2,
        max_iter,
        n_inner,
        p_iter,
        stop_on_convergence=stop_on_convergence,
        eigen_mode=eigen_mode,
        print_eigvals=print_eigvals,
    )


def cg_ss(
    x0: np.ndarray,
    g_half: np.ndarray,
    x0_half: np.ndarray,
    k: int,
    costmax: float,
    x2: float,
    max_iter: int,
    n_inner: int,
    p_iter: int,
    stop_on_convergence: bool = True,
    eigen_mode: str = "eigsh",
    print_eigvals: bool = False,
):
    """Core solver matching `cg_ss.m`."""
    n = x0.shape[0]
    p = np.zeros((n, n), dtype=x0.dtype)
    trace_g_p = 0.0
    trace_x0_p = 0.0
    gamma_matrix = np.zeros((n, n), dtype=x0.dtype)
    tau = 1.618

    one_v = np.ones(n, dtype=x0.dtype)
    one_over_n = 1.0 / n
    obj_value = np.zeros(max_iter)

    start = time.perf_counter()
    for i in range(1, max_iter + 1):
        penalty = np.sqrt(i)
        for j in range(1, n_inner + 1):
            p_plus_en = p + one_over_n
            g_p_minus = trace_g_p - costmax

            x2_temp = min(x2 + penalty * g_p_minus, 0.0)
            g_grad = np.minimum(gamma_matrix + penalty * p_plus_en, 0)
            temp = one_over_n * (g_grad @ one_v)
            g_grad_mean = float(np.mean(g_grad))

            def apply_vector(vector):
                vector = np.asarray(vector).reshape(-1)
                b1 = np.sum(vector) * (g_grad_mean - one_over_n - temp) - temp @ vector
                c1 = x0_half @ (x0_half.T @ vector) + x2_temp * (
                    g_half @ (g_half.T @ vector)
                )
                return g_grad @ vector + c1 + b1

            def matvec(vector):
                return apply_vector(vector)

            def matmat(matrix):
                matrix = np.asarray(matrix)
                return np.column_stack(
                    [apply_vector(matrix[:, col]) for col in range(matrix.shape[1])]
                )

            if eigen_mode == "eigsh":
                operator = LinearOperator(
                    (n, n), matvec=matvec, matmat=matmat, dtype=x0.dtype
                )
                eigvals, eigvecs = eigsh(operator, k=1, which="SA", tol=1.0 / j)
                eigval = float(eigvals[0])
                u = eigvecs[:, 0]
            elif eigen_mode == "eigs":
                operator = LinearOperator(
                    (n, n), matvec=matvec, matmat=matmat, dtype=x0.dtype
                )
                eigvals, eigvecs = eigs(operator, k=1, which="SR", tol=1.0 / j)
                eigval = float(np.real(eigvals[0]))
                u = np.real(eigvecs[:, 0])
                u = u / np.linalg.norm(u)
            elif eigen_mode == "lobpcg":
                operator = LinearOperator(
                    (n, n), matvec=matvec, matmat=matmat, dtype=x0.dtype
                )
                initial = np.ones((n, 1), dtype=x0.dtype)
                eigvals, eigvecs = lobpcg(
                    operator,
                    initial,
                    largest=False,
                    tol=1.0 / j,
                    maxiter=200,
                )
                eigval = float(eigvals[0])
                u = eigvecs[:, 0]
                u = u / np.linalg.norm(u)
            elif eigen_mode == "dense":
                basis = np.eye(n, dtype=x0.dtype)
                dense_operator = np.column_stack(
                    [apply_vector(basis[:, col]) for col in range(n)]
                )
                dense_operator = (dense_operator + dense_operator.T) / 2
                eigvals, eigvecs = np.linalg.eigh(dense_operator)
                eigval = float(eigvals[0])
                u = eigvecs[:, 0]
            else:
                raise ValueError(f"Unknown eigen_mode: {eigen_mode}")

            if print_eigvals:
                print(f"outer {i} inner {j} eigval {eigval:.6f}")

            if eigval > 0:
                print("positive eigenvalue for A")
                continue

            h = (k - 1) * np.outer(u, u)
            alpha_step = 2.0 / ((i - 1) * n_inner + j - 1 + 2)
            p = (1 - alpha_step) * p + alpha_step * h
            trace_g_p = (1 - alpha_step) * trace_g_p + (k - 1) * alpha_step * (
                np.linalg.norm(g_half.T @ u) ** 2
            )
            trace_x0_p = (1 - alpha_step) * trace_x0_p + (k - 1) * alpha_step * (
                np.linalg.norm(x0_half.T @ u) ** 2
            )

        p_nneg = p + one_over_n
        gamma_matrix = np.minimum(gamma_matrix + tau * penalty * p_nneg, 0)
        x2 = min(x2 + tau * penalty * (trace_g_p - costmax), 0.0)
        max_error = max(abs(float(np.min(p_nneg))), abs(trace_g_p - costmax))
        obj_value[i - 1] = trace_x0_p + 1

        if i % p_iter == 0 or max_error <= 1e-5:
            print(f"after {i} iteration, cost difference is {trace_g_p - costmax:f}")
            print(f"after {i} iteration, max_error is {max_error:f}")
            print(f"after {i} iteration, obj value is {obj_value[i - 1]:f}")
            if i > 1:
                print(
                    f"after {i} iteration, difference in obj_value is "
                    f"{obj_value[i - 1] - obj_value[i - 2]:f}"
                )
            print()

        should_stop = max_error <= 1e-5 or (
            i > 1
            and abs(obj_value[i - 1] - obj_value[i - 2]) <= 1e-5
            and max_error <= 1e-4
        )
        if stop_on_convergence and should_stop:
            print(
                f"after {i} iteration, difference in obj_value is "
                f"{obj_value[i - 1] - obj_value[i - 2]:f}"
            )
            break

    p = p + one_over_n
    elapsed = _elapsed_since(start)
    epsilon = trace_x0_p + 1
    print(f"final objective would be {epsilon:f}")
    return p, epsilon, obj_value, elapsed


def prepare_cg_inputs(x0: np.ndarray, g: np.ndarray):
    """Return the preprocessing outputs from `cg_ss_test.m` for validation."""
    v_max = eigsh(g, k=1, which="LA", return_eigenvectors=False)[0]
    v_max = v_max / 10
    g_norm = g / v_max

    g_evals, g_evecs = np.linalg.eigh(g_norm)
    g_keep = g_evals > 1e-10
    g_half = g_evecs[:, g_keep] * np.sqrt(g_evals[g_keep])

    x0_evals, x0_evecs = np.linalg.eigh(x0)
    x0_keep = x0_evals > 1e-10
    x0_half = x0_evecs[:, x0_keep] * np.sqrt(x0_evals[x0_keep])

    temp = g_half.T @ x0_half
    costmax = np.linalg.norm(temp, "fro") ** 2
    return g_half, x0_half, float(costmax)


def _elapsed_since(start: float) -> float:
    return time.perf_counter() - start
