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
            "status": "Stable in this SDP Run",
            "confidence": "High",
            "color": "green",
            "bottom_line": (
                "This SDP run did not find a substantially different equal-or-better candidate. "
                "Your uploaded clustering has strong stability evidence."
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


def stability_score(objective_minus_k: float) -> float:
    """Map Objective - K to a 0-100 display score for the gauge."""
    clipped = min(max(objective_minus_k, -0.60), 0.0)
    return 100.0 * (1.0 + clipped / 0.60)


def render_stability_gauge(objective_minus_k: float) -> None:
    """Render a compact gauge for the UI-only stability heuristic."""
    score = stability_score(objective_minus_k)
    marker_left = max(1.0, min(score, 99.0))
    st.markdown(
        f"""
        <div style="margin: 0.75rem 0 1.25rem 0;">
            <div style="
                position: relative;
                height: 1.1rem;
                border-radius: 999px;
                background: linear-gradient(90deg, #d9534f 0%, #d9534f 50%, #f0ad4e 50%, #f0ad4e 91.67%, #2e7d32 91.67%, #2e7d32 100%);
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
                <span>Weak / Ambiguous</span>
                <span>Moderately Stable</span>
                <span>Strongly Stable</span>
            </div>
            <div style="margin-top: 0.35rem; color: #666; font-size: 0.84rem;">
                Visual index: {score:.0f}/100. Derived from Objective - K = {objective_minus_k:.6g}.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.title("Clustering Stability Verifier")
    st.caption("A sublevel-set SDP stress test for uploaded clustering results.")
    st.markdown(
        """
        This app asks whether another clustering can fit the data just as well while being
        structurally different from the clustering you uploaded.

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

            `G` is computed from the centered data matrix:

            ```python
            Y_centered = Y - Y.mean(axis=0, keepdims=True)
            G = Y_centered @ Y_centered.T
            ```

            The app checks that `X0` and `G` are square, finite numeric matrices with the same shape,
            and that `trace(X0)` is an integer cluster count.
            """
        )

    with st.expander("Methodology overview"):
        st.markdown(
            """
            The method is based on Meila (2018), **"How to tell when a clustering is
            (approximately) correct using convex relaxations."**

            The workflow is:

            1. Start with an existing clustering, encoded as `X0`.
            2. Encode the data geometry as the centered Gram matrix `G`.
            3. Search the sublevel set: candidates with k-means quality at least as good as `X0`.
            4. Look for the candidate least similar to `X0`.

            If this stress test cannot move far away from `X0`, the uploaded clustering has stronger
            stability evidence. This is a verifier, not a ground-truth oracle.
            """
        )

    with st.expander("Optimization problem"):
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
                f"- Clusters `K = trace(X0)`: `{k}`",
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

    st.markdown("#### Stability Evidence Gauge")
    render_stability_gauge(objective_minus_k)

    st.markdown("#### Key Numbers")
    col1, col2, col3 = st.columns(3)
    col1.metric("Objective - K", f"{objective_minus_k:.8g}")
    col2.metric("K", f"{k}")
    col3.metric("Runtime", f"{result.elapsed:.2f} s")

    st.markdown(
        """
        Closer to `0` means stronger stability evidence; more negative means the uploaded clustering
        is easier to replace. The gauge is a visual aid, not a theorem score or probability.
        """
    )

    with st.expander("What does this mean?"):
        st.markdown(
            """
            The solver tries to find an alternative clustering that has equal-or-better k-means quality
            and is as different from the uploaded clustering as possible.

            `Objective - K` is the primary ADMM stability readout for this demo:

            - Near `0`: stronger stability evidence.
            - More negative: a more replaceable clustering structure.

            Heuristic UI bands, not theorem thresholds:

            - `> -0.05`: strong stability evidence
            - `-0.30` to `-0.05`: moderate stability evidence
            - `<= -0.30`: weak or ambiguous evidence

            Weak evidence does not prove the clustering is wrong. It means this convex stress test did
            not certify stability in this run. The gauge is a visual aid, not a theorem score or
            probability.
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

    st.download_button(
        "Download Full Solver Diagnostics (.mat)",
        data=result_to_mat_bytes(result),
        file_name=f"{result.solver.lower()}_result.mat",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
