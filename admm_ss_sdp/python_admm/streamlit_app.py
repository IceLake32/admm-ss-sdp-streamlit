"""Streamlit demo for the sublevel-set SDP clustering solvers."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from solver_api import load_problem, result_to_mat_bytes, run_admm, run_cg


st.set_page_config(page_title="ADMM SS SDP Solver", layout="wide")

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


def main() -> None:
    st.title("Sublevel-Set SDP Clustering Solver")
    st.markdown(
        """
        This app implements the sublevel-set SDP idea from Meila (2018),
        **"How to tell when a clustering is (approximately) correct using convex relaxations."**
        It is a validation tool for an existing clustering, not a tool that runs k-means from scratch.

        The main question is: **among all clusterings that fit the data at least as well as the uploaded
        clustering, can any of them be very different?** If the answer is no, the uploaded clustering has
        a stronger stability certificate. If the answer is yes, the data may support multiple different
        clusterings with comparable k-means quality.
        """
    )
    with st.expander("Methodology overview", expanded=True):
        st.markdown(
            """
            The workflow has four steps:

            1. Start with a clustering you already have, usually produced by k-means or another clustering
               method. This clustering is encoded as `X0`.
            2. Convert the original data geometry into the centered Gram matrix `G`. The value `<G, X>`
               is the SDP form of the k-means data-fit score for a clustering matrix `X`.
            3. Define the sublevel set: all candidate clusterings whose k-means quality is at least as good
               as the uploaded clustering. In this app that condition is written as `<G, X> >= <G, X0>`.
            4. Solve a semidefinite program that searches inside that set for the candidate least similar
               to `X0`. This is the adversarial check: the solver is trying to find a credible alternative
               clustering that still fits the data well.

            The important interpretation is qualitative: if even this adversarial search stays close to
            `X0`, then the uploaded clustering is stable under the SDP relaxation. If the solver finds a
            far-away candidate, the uploaded clustering is harder to certify.
            """
        )

    with st.expander("What problem is this solver checking?", expanded=True):
        st.markdown(
            """
            The uploaded clustering is represented by a matrix `X0`. The solver searches over relaxed
            clustering matrices `X` and solves:

            ```text
            minimize    <X0, X>

            subject to  trace(X) = K
                        X 1 = 1
                        X >= 0
                        X is positive semidefinite
                        <G, X> >= <G, X0>
            ```

            The constraint `<G, X> >= <G, X0>` means that the candidate `X` must be at least as good as
            `X0` under the k-means/Gram-matrix score. The objective `<X0, X>` tries to make `X` as
            different from `X0` as possible. This is a "stress test" for the uploaded clustering.

            Exact clustering is combinatorial, so the app uses an SDP relaxation: it searches over a
            larger convex set of matrix candidates. A strong certificate from this relaxed problem is
            meaningful because the relaxed search is allowed to be more flexible than real hard clusterings.
            """
        )

    with st.expander("Required `.mat` format", expanded=True):
        st.markdown(
            """
            Upload a MATLAB `.mat` file containing exactly the problem data variables:

            - `X0`: an `n x n` clustering matrix for the clustering you want to validate.
            - `G`: an `n x n` centered Gram matrix describing the data geometry.

            `X0` is built from cluster labels. If points `i` and `j` are in the same cluster of size
            `m`, then `X0[i, j] = 1 / m`; otherwise `X0[i, j] = 0`. Its trace equals the number of
            clusters, so this app infers `K` from `trace(X0)`.

            `G` is computed from the centered data matrix. If the raw data matrix is `Y` with one point
            per row, first subtract the column-wise mean:

            ```python
            Y_centered = Y - Y.mean(axis=0, keepdims=True)
            G = Y_centered @ Y_centered.T
            ```

            The app checks that `X0` and `G` are square, finite numeric matrices with the same shape,
            and that `trace(X0)` is an integer cluster count.
            """
        )

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload MATLAB data", type=["mat"])
        solver = st.selectbox("Solver", ["ADMM", "CG"])

        if solver == "ADMM":
            st.caption(f"Tolerance is fixed at `{ADMM_EPS:g}`.")
            st.caption(f"Demo size limit: `n <= {ADMM_N_LIMIT}`.")
        else:
            st.caption("CG is experimental in Python because iterative eigensolvers can diverge from Matlab.")
            st.caption(f"Maximum iterations are fixed at `{CG_MAX_ITER}`.")
            st.caption(f"Demo size limit: `n <= {CG_N_LIMIT}`.")
            eigen_mode = st.selectbox("Eigen solver", ["eigsh", "eigs"])

        run_clicked = st.button("Run Solver", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a `.mat` file containing variables `X0` and `G`.")
        return

    try:
        x0, g = load_problem(BytesIO(uploaded_file.getvalue()))
        k = infer_cluster_count(x0)
    except Exception as exc:
        st.error("Invalid `.mat` file format.")
        st.error(str(exc))
        return

    n = x0.shape[0]
    st.subheader("Input")
    st.markdown(
        "\n".join(
            [
                f"- `n`: `{n}`",
                f"- inferred `K = trace(X0)`: `{k}`",
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
                    eigen_mode=eigen_mode,
                )
        except Exception as exc:
            st.exception(exc)
            return

    st.subheader("Result")
    objective_minus_k = result.objective - k
    col1, col2, col3 = st.columns(3)
    col1.metric("Solver", result.solver)
    col2.metric("Objective", f"{result.objective:.8g}")
    col3.metric("Runtime", f"{result.elapsed:.2f} s")
    st.metric("Objective - K", f"{objective_minus_k:.8g}")

    st.markdown(
        "\n".join(
            f"- `{key}`: `{value:.12g}`" for key, value in result.metrics.items()
        )
    )

    st.subheader("Result Meaning")
    meaning_lines = [
        f"- `K`: inferred from `trace(X0)`, here `K = {k}`.",
        "- `Objective`: solver objective value. For ADMM this is `<X0, X>`, the overlap between the uploaded clustering and the returned candidate.",
        "- `Objective - K`: for ADMM this is `<X0, X> - K`. Values near `0` mean the solver could not move far away from the uploaded clustering while keeping equal-or-better k-means quality. More negative values mean it found a more different candidate, so the uploaded clustering is less strongly certified.",
        "- `Runtime`: wall-clock time spent inside the selected solver.",
        "- `min_X` or `min_P`: minimum entry of the returned matrix; values slightly below zero can be numerical tolerance error.",
        "- `trace_X` or `trace_P`: trace of the returned matrix; for ADMM, `trace_X` should be close to `K`.",
        "- `trace_GX` or `trace_GP`: data-fit score of the returned matrix.",
        "- `trace_GX0`: data-fit score of the uploaded clustering matrix `X0`; ADMM targets `trace_GX >= trace_GX0`.",
        "- `trace_X0X` or `trace_X0P`: overlap between `X0` and the returned matrix; for ADMM this is the same quantity as the objective.",
    ]
    if result.solver == "ADMM":
        meaning_lines.extend(
            [
                f"- `Tolerance`: fixed at `{ADMM_EPS:g}` for ADMM in this demo.",
                "- `v`: ADMM multiplier for the sublevel constraint `trace(G X) >= trace(G X0)`.",
            ]
        )
    else:
        meaning_lines.append(f"- `Maximum iterations`: fixed at `{CG_MAX_ITER}` for CG in this demo.")
    st.markdown("\n".join(meaning_lines))

    st.download_button(
        "Download result .mat",
        data=result_to_mat_bytes(result),
        file_name=f"{result.solver.lower()}_result.mat",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
