"""Streamlit demo for the sublevel-set SDP clustering solvers."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st

from solver_api import load_problem, result_to_mat_bytes, run_admm, run_cg


st.set_page_config(page_title="K-means Stability Guarantee", layout="wide")

ADMM_EPS = 1e-4
ADMM_PRINT_INTERVAL = 100
ADMM_N_LIMIT = 500
CG_MAX_ITER = 500
CG_PRINT_INTERVAL = 50
CG_N_LIMIT = 1000


def infer_cluster_count(x0, tol: float = 1e-6) -> int:
    """Infer K from a clustering matrix using trace(X0) = K."""
    trace_x0 = float(x0.trace())
    k = int(round(trace_x0))
    if k < 2:
        raise ValueError(f"`trace(X0)` must identify at least 2 clusters. Got {trace_x0:.12g}.")
    if abs(trace_x0 - k) > tol:
        raise ValueError(
            "`trace(X0)` should be an integer equal to the number of clusters. "
            f"Got trace(X0) = {trace_x0:.12g}."
        )
    return k


def cluster_proportions(x0) -> tuple[float, float]:
    """Infer p_min and p_max from a clustering matrix."""
    diagonal = np.asarray(np.diag(x0), dtype=float)
    if np.any(diagonal <= 0):
        raise ValueError("X0 diagonal entries must be positive to infer cluster sizes.")
    point_cluster_sizes = np.rint(1.0 / diagonal).astype(int)
    if np.any(point_cluster_sizes <= 0):
        raise ValueError("Could not infer valid cluster sizes from X0.")
    n = x0.shape[0]
    return float(np.min(point_cluster_sizes) / n), float(np.max(point_cluster_sizes) / n)


def stability_certificate(objective: float, x0, k: int) -> dict[str, float | str | bool]:
    """Compute the epsilon certificate quantities used in the SS guarantee."""
    p_min, p_max = cluster_proportions(x0)
    epsilon = max(float(k - objective), 0.0) * p_max
    guaranteed = epsilon <= p_min + 1e-10
    if guaranteed:
        return {
            "status": "Guaranteed",
            "color": "green",
            "epsilon": epsilon,
            "p_min": p_min,
            "p_max": p_max,
            "bottom_line": (
                "Every equal-or-better clustering is certified to be epsilon-close to the uploaded "
                "clustering, under this SDP run."
            ),
            "guaranteed": True,
        }
    return {
        "status": "Not guaranteed",
        "color": "orange",
        "epsilon": epsilon,
        "p_min": p_min,
        "p_max": p_max,
        "bottom_line": (
            "This run did not certify stability. This does not prove the clustering is wrong; it means "
            "the guarantee condition was not met."
        ),
        "guaranteed": False,
    }


def render_epsilon_gauge(epsilon: float, p_min: float) -> None:
    """Render a compact gauge for epsilon relative to p_min."""
    ratio = 0.0 if p_min <= 0 else epsilon / p_min
    marker_left = max(1.0, min(100.0 * ratio / 2.0, 99.0))
    st.markdown(
        f"""
        <div style="margin: 0.75rem 0 1.25rem 0;">
            <div style="
                position: relative;
                height: 1.1rem;
                border-radius: 999px;
                background: linear-gradient(90deg, #2e7d32 0%, #2e7d32 50%, #f0ad4e 50%, #f0ad4e 100%);
                box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
            ">
                <div style="
                    position: absolute;
                    left: {marker_left:.2f}%;
                    top: 50%;
                    transform: translate(-50%, -50%);
                    width: 1.35rem;
                    height: 1.35rem;
                    border-radius: 999px;
                    background: #111;
                    border: 3px solid white;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.25);
                "></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.45rem; font-size: 0.86rem;">
                <span>epsilon = 0</span>
                <span>guarantee threshold p_min</span>
                <span>2 x p_min</span>
            </div>
            <div style="margin-top: 0.35rem; color: #666; font-size: 0.84rem;">
                Smaller epsilon is better. The formal guarantee requires epsilon <= p_min.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Stability Guarantee for a K-means Clustering")
    st.caption("Sublevel Set (SS) verification for an uploaded clustering.")
    st.markdown(
        """
        This software verifies whether your clustering `C` is approximately correct. It is based on
        Meila (2018),
        **["How to tell when a clustering is (approximately) correct using convex relaxations"](https://sites.stat.washington.edu/mmp/Papers/sdp-kmeans-nips18.pdf)**,
        and the original [ADMM SS SDP code](https://github.com/mathcg/admm_ss_sdp/) by Gang Cheng.

        """
    )

    with st.expander("How it works", expanded=True):
        st.markdown(
            """
            1. Upload your Clustering data with accepted formats.
            2. Click **Run**. The app sets up and solves an optimization
               problem in the background.
            3. Get the answer. The app returns one of two forms:

               - **Guaranteed `epsilon = 0.04`**  
                 The clustering has a deterministic stability guarantee.

               - **Not guaranteed (`epsilon = 0.22`, `p_min = 0.18`)**  
                 The run did not certify stability. This does not prove the clustering is wrong.

            `epsilon` is the Optimality Interval (OI). The smaller it is, the better. It is not a
            confidence interval; it is a deterministic bound returned by the optimization certificate.
            """
        )

    with st.expander("Data formats", expanded=True):
        st.markdown(
            """
            Upload a MATLAB `.mat`, NumPy `.npz`, or CSV `.csv` file containing the problem data.

            For `.mat` and `.npz`, provide:

            - `X0`: an `n x n` clustering matrix for the clustering you want to validate.
            - `G`: an `n x n` centered Gram matrix describing the data geometry.

            For `.csv`, provide one row per data point, numeric feature columns, and one cluster label
            column named `label`. The app will construct `X0` and `G` automatically.

            `X0` is built from cluster labels. If points `i` and `j` are in the same cluster of size
            `m`, then `X0[i, j] = 1 / m`; otherwise `X0[i, j] = 0`. Its trace equals the number of
            clusters, so this app infers `K` from `trace(X0)`.

            `G` is computed from the centered data matrix:

            ```python
            Y_centered = Y - Y.mean(axis=0, keepdims=True)
            G = Y_centered @ Y_centered.T
            ```

            The app checks that `X0` and `G` are square, finite numeric matrices with the same shape,
            and that `trace(X0)` is an integer cluster count.
            """
        )

    with st.expander("What does epsilon actually mean?"):
        st.markdown(
            """
            Remember that a clustering is evaluated by its k-means cost:

            $$
            Cost(C)=\\sum_{k=1}^{K}\\sum_{i\\in \\text{cluster }k}\\|x_i-\\mu_k\\|^2.
            $$

            **What we know:** data `D`, clustering `C`, and its `Cost(C)`.

            **What we want to know, first version:** Can there be another `C'` so that
            `Cost(C') <= Cost(C)`?

            The answer, if we could know it, would not be very informative. If we reassign a single
            point to a different cluster, the change in cost may be very small.

            **What we want to know, better version:** Can there be another `C'`, very different from
            `C`, so that `Cost(C') <= Cost(C)`?

            This is what the SS algorithm searches for. When it returns a guaranteed `epsilon`, then
            any clustering `C'` with `Cost(C') <= Cost(C)` must be `epsilon`-close to `C`.

            `epsilon` is a difference between two clusterings, measured by the fraction of the `n`
            data points that must change cluster assignment to turn `C'` into `C`. For example, if
            `n = 200` and `epsilon = 0.05`, then any clustering as good as `C` or better must differ
            from `C` in at most 10 points.
            """
        )

    with st.expander("Technical optimization problem"):
        st.markdown(
            """
            The solver searches over relaxed clustering matrices `X` and solves:

            ```text
            minimize    <X0, X>

            subject to  trace(X) = K
                        X 1 = 1
                        X >= 0
                        X is positive semidefinite
                        <G, X> >= <G, X0>
            ```

            The constraint `<G, X> >= <G, X0>` means the candidate has equal-or-better k-means quality.
            The objective `<X0, X>` tries to find the least similar candidate. Exact clustering is
            combinatorial, so the app uses a convex SDP relaxation.
            """
        )

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload problem data", type=["mat", "npz", "csv"])
        solver = st.selectbox("Solver", ["ADMM", "CG"])

        if solver == "ADMM":
            st.caption(f"Tolerance is fixed at `{ADMM_EPS:g}`.")
            st.caption(f"Demo size limit: `n <= {ADMM_N_LIMIT}`.")
        else:
            st.caption("CG is experimental in this Python demo.")
            st.caption(f"Maximum iterations are fixed at `{CG_MAX_ITER}`.")
            st.caption(f"Demo size limit: `n <= {CG_N_LIMIT}`.")
            # eigen_mode = st.selectbox("Eigen solver", ["eigsh", "eigs"])

        run_clicked = st.button("Run Sublevel Set (SS) algorithm", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a `.mat`, `.npz`, or `.csv` file.")
        return

    try:
        x0, g = load_problem(BytesIO(uploaded_file.getvalue()), uploaded_file.name)
        k = infer_cluster_count(x0)
    except Exception as exc:
        st.error("Invalid input file format.")
        st.error(str(exc))
        return

    n = x0.shape[0]
    st.subheader("Input")
    st.markdown(
        "\n".join(
            [
                f"- Clusters `K = trace(X0)`: `{k}`",
                f"- `n`: `{n}`",
                f"- `X0 shape`: `{x0.shape}`",
                f"- `G shape`: `{g.shape}`",
            ]
        )
    )

    n_limit = ADMM_N_LIMIT if solver == "ADMM" else CG_N_LIMIT
    if n > n_limit:
        st.warning(f"This demo is configured for n <= {n_limit}. Uploaded problem has n = {n}.")
        return

    if not run_clicked:
        return

    with st.spinner(f"Running {solver} solver..."):
        try:
            if solver == "ADMM":
                result = run_admm(x0, g, int(k), eps=ADMM_EPS, p_iter=ADMM_PRINT_INTERVAL)
            else:
                result = run_cg(
                    x0,
                    g,
                    int(k),
                    max_iter=CG_MAX_ITER,
                    p_iter=CG_PRINT_INTERVAL,
                    eigen_mode="eigs",
                )
        except Exception as exc:
            st.exception(exc)
            return

    objective_minus_k = result.objective - k
    if result.solver == "ADMM":
        certificate = stability_certificate(result.objective, x0, int(k))
    else:
        p_min, p_max = cluster_proportions(x0)
        certificate = {
                "status": "Not guaranteed",
            "color": "gray",
            "epsilon": float(result.objective),
            "p_min": p_min,
            "p_max": p_max,
            "bottom_line": (
                "This CG run is experimental in the Python demo. Treat the result as a diagnostic, "
                "not as the primary stability certificate."
            ),
            "guaranteed": False,
        }

    st.subheader("Get the answer")
    st.markdown(
        f"""
        <div style="border-left: 0.5rem solid {certificate['color']}; padding: 1rem 1.25rem; background: #f8f9fa;">
            <h3 style="margin-top: 0;">{certificate['status']} epsilon = {float(certificate['epsilon']):.8g}</h3>
            <p style="font-size: 1.05rem; margin-bottom: 0;">{certificate['bottom_line']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Optimality Interval Gauge")
    render_epsilon_gauge(float(certificate["epsilon"]), float(certificate["p_min"]))

    st.markdown("#### Key Numbers")
    col1, col2, col3 = st.columns(3)
    col1.metric("epsilon", f"{float(certificate['epsilon']):.8g}")
    col2.metric("p_min", f"{float(certificate['p_min']):.8g}")
    col3.metric("Runtime", f"{result.elapsed:.2f} s")

    st.markdown(
        """
        Smaller `epsilon` is better. The clustering is guaranteed when `epsilon <= p_min`.
        """
    )

    with st.expander("How to interpret this answer", expanded=True):
        st.markdown(
            f"""
            `epsilon` is the Optimality Interval. The smaller it is, the better.

            If the answer is **Guaranteed**, then any clustering `C'` with `Cost(C') <= Cost(C)` must
            be `epsilon`-close to the uploaded clustering `C`. For this dataset with `n = {n}`, that
            corresponds to at most about `{float(certificate['epsilon']) * n:.3g}` data points changing
            cluster assignment.

            `p_min` is the smallest cluster size divided by `n`. The guarantee condition used here is
            `epsilon <= p_min`.

            If the answer is **Not Guaranteed**, this can happen because the data are not strongly
            clusterable, the uploaded clustering is only a local minimum, or the certificate is a
            borderline case. It does not prove that the clustering is wrong. A small `epsilon` can still
            be useful as a heuristic stability signal.
            """
        )

    with st.expander("If the clustering is not guaranteed"):
        st.markdown(
            """
            A few things to check:

            - Is `epsilon` close to `p_min`? If `epsilon` exceeds `p_min`, a formal guarantee cannot be
              returned, but a small `epsilon` still suggests more stability.
            - Try nearby values of `K`; the OI can be useful heuristically for selecting the number of
              clusters.
            - If it makes sense for your application, remove clear outliers and try again. Outliers often
              make the OI worse.
            """
        )

    with st.expander("Advanced solver diagnostics"):
        st.markdown(
            "\n".join(
                [
                    f"- `K`: `{k}`",
                    f"- `Objective - K`: `{objective_minus_k:.12g}`",
                    f"- `p_max`: `{float(certificate['p_max']):.12g}`",
                ]
            )
        )
        st.markdown(
            "\n".join(
                f"- `{key}`: `{value:.12g}`" for key, value in result.metrics.items()
            )
        )
        if result.solver == "ADMM":
            st.markdown(f"- `tolerance`: `{ADMM_EPS:g}`")
        else:
            st.markdown(f"- `maximum_iterations`: `{CG_MAX_ITER}`")
        st.caption("These diagnostics are for optimization validity, not the primary user verdict.")

    st.download_button(
        "Download Full Solver Diagnostics (.mat)",
        data=result_to_mat_bytes(result),
        file_name=f"{result.solver.lower()}_result.mat",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
