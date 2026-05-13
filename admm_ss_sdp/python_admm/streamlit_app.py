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


def stability_verdict(objective_minus_k: float) -> dict[str, str]:
    """Return UI-only stability labels from the ADMM primary metric."""
    if objective_minus_k > -0.05:
        return {
            "status": "Certified Stable",
            "confidence": "High",
            "color": "green",
            "bottom_line": (
                "No substantially different equally-good clustering was found. "
                "Your uploaded clustering appears structurally robust."
            ),
        }
    if objective_minus_k > -0.30:
        return {
            "status": "Moderately Stable",
            "confidence": "Moderate",
            "color": "orange",
            "bottom_line": (
                "The solver found somewhat different alternatives, but not drastically different ones. "
                "Your clustering appears reasonably credible, though not uniquely certified."
            ),
        }
    return {
        "status": "Weak / Potentially Ambiguous",
        "confidence": "Low",
        "color": "red",
        "bottom_line": (
            "The solver found meaningfully different alternatives with similar quality. "
            "Multiple plausible cluster structures may exist."
        ),
    }


def main() -> None:
    st.title("Sublevel-Set SDP Clustering Solver")
    st.markdown(
        """
        This app implements the sublevel-set SDP idea from Meila (2018),
        **"How to tell when a clustering is (approximately) correct using convex relaxations."**
        It is a validation tool for an existing clustering, not a tool that runs k-means from scratch.

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

    with st.expander("Required data format", expanded=True):
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
        uploaded_file = st.file_uploader("Upload problem data", type=["mat", "npz", "csv"])
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

    objective_minus_k = result.objective - k
    if result.solver == "ADMM":
        verdict = stability_verdict(objective_minus_k)
    else:
        verdict = {
            "status": "No Guarantee",
            "confidence": "Low",
            "color": "gray",
            "bottom_line": (
                "This CG run is experimental in the Python demo. Treat the result as a diagnostic, "
                "not as the primary stability certificate."
            ),
        }

    st.subheader("Uploaded Clustering Stress-Test Result")
    st.markdown(
        f"""
        <div style="border-left: 0.5rem solid {verdict['color']}; padding: 1rem 1.25rem; background: #f8f9fa;">
            <h3 style="margin-top: 0;">{verdict['status']}</h3>
            <p style="font-size: 1.05rem; margin-bottom: 0;">{verdict['bottom_line']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Primary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Solver Backend", f"{result.solver} / SDP")
    col2.metric("K", f"{k}")
    col3.metric("Objective - K", f"{objective_minus_k:.8g}")
    col4.metric("Runtime", f"{result.elapsed:.2f} s")
    st.metric("Confidence Level", verdict["confidence"])

    with st.expander("How to read Objective - K", expanded=True):
        st.markdown(
            """
            `Objective - K` is the primary ADMM stability readout for this demo.

            - Near `0`: the solver could not move far from your clustering while preserving
              equal-or-better k-means quality. This supports stronger structural stability.
            - Moderately negative: the solver found somewhat different alternatives. This is a
              moderate certificate.
            - Very negative: the solver found substantially different alternatives. The uploaded
              clustering may not be unique.

            Heuristic UI bands, not theorem thresholds:

            - `Objective - K > -0.05`: Strongly stable
            - `-0.05 >= Objective - K > -0.30`: Moderately stable
            - `Objective - K <= -0.30`: Weak / ambiguous
            """
        )

    st.subheader("Plain English Interpretation")
    st.markdown(
        """
        This solver searches for an alternative clustering that:

        1. Matches or improves your uploaded clustering's k-means quality.
        2. Is as structurally different as possible.

        If the search fails to move far away, your clustering is harder to replace. If it succeeds,
        your clustering may be one of several plausible explanations for the same data.
        """
    )

    st.subheader("Recommended Next Steps")
    if verdict["status"] == "Certified Stable":
        st.markdown(
            """
            - Try nearby values of `K`.
            - Test sensitivity to outliers.
            - Compare ADMM with the experimental CG backend on larger examples.
            """
        )
    elif verdict["status"] == "Moderately Stable":
        st.markdown(
            """
            - Inspect boundary or ambiguous points.
            - Test multiple values of `K`.
            - Compare results from different clustering initializations.
            """
        )
    else:
        st.markdown(
            """
            - Check for overlap, outliers, or weak separation.
            - Revisit the chosen number of clusters `K`.
            - Consider whether the data truly has a strong cluster structure.
            """
        )

    with st.expander("Advanced Solver Diagnostics"):
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

    st.warning(
        "Important: Weak or No Guarantee does not prove the clustering is incorrect. "
        "It means this convex stress-test found limited evidence for uniqueness or robustness. "
        "This tool is a verifier, not a ground-truth oracle."
    )

    st.markdown("### Are your clusters real, or just one convenient partition?")

    st.download_button(
        "Download Full Solver Diagnostics (.mat)",
        data=result_to_mat_bytes(result),
        file_name=f"{result.solver.lower()}_result.mat",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
