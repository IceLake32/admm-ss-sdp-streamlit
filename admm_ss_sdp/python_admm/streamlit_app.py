"""Streamlit demo for the sublevel-set SDP clustering solvers."""

from __future__ import annotations

from io import BytesIO

import streamlit as st

from solver_api import load_problem, result_to_mat_bytes, run_admm, run_cg


st.set_page_config(page_title="ADMM SS SDP Solver", layout="wide")


def main() -> None:
    st.title("Sublevel-Set SDP Clustering Solver")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload MATLAB data", type=["mat"])
        solver = st.selectbox("Solver", ["ADMM", "CG"])
        k = st.number_input("Number of clusters", min_value=2, max_value=100, value=4, step=1)

        if solver == "ADMM":
            eps = st.selectbox("Tolerance", [1e-3, 1e-4, 1e-5], index=1, format_func=lambda x: f"{x:g}")
            p_iter = st.number_input("Print interval", min_value=1, max_value=1000, value=100, step=10)
            n_limit = st.number_input("Maximum n", min_value=50, max_value=1000, value=500, step=50)
        else:
            st.caption("CG is experimental in Python because iterative eigensolvers can diverge from Matlab.")
            max_iter = st.number_input("Maximum iterations", min_value=10, max_value=5000, value=500, step=50)
            p_iter = st.number_input("Print interval", min_value=1, max_value=1000, value=50, step=10)
            eigen_mode = st.selectbox("Eigen solver", ["eigsh", "eigs"])
            n_limit = st.number_input("Maximum n", min_value=50, max_value=2000, value=1000, step=50)

        run_clicked = st.button("Run Solver", type="primary", use_container_width=True)

    if uploaded_file is None:
        st.info("Upload a `.mat` file containing variables `X0` and `G`.")
        return

    try:
        x0, g = load_problem(BytesIO(uploaded_file.getvalue()))
    except Exception as exc:
        st.error(str(exc))
        return

    n = x0.shape[0]
    st.subheader("Input")
    st.markdown(
        "\n".join(
            [
                f"- `n`: `{n}`",
                f"- `X0 shape`: `{x0.shape}`",
                f"- `G shape`: `{g.shape}`",
            ]
        )
    )

    if n > n_limit:
        st.warning(f"This demo is configured for n <= {n_limit}. Uploaded problem has n = {n}.")
        return

    if not run_clicked:
        return

    with st.spinner(f"Running {solver} solver..."):
        try:
            if solver == "ADMM":
                result = run_admm(x0, g, int(k), eps=float(eps), p_iter=int(p_iter))
            else:
                result = run_cg(
                    x0,
                    g,
                    int(k),
                    max_iter=int(max_iter),
                    p_iter=int(p_iter),
                    eigen_mode=eigen_mode,
                )
        except Exception as exc:
            st.exception(exc)
            return

    st.subheader("Result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Solver", result.solver)
    col2.metric("Objective", f"{result.objective:.8g}")
    col3.metric("Runtime", f"{result.elapsed:.2f} s")

    st.markdown(
        "\n".join(
            f"- `{key}`: `{value:.12g}`" for key, value in result.metrics.items()
        )
    )

    st.download_button(
        "Download result .mat",
        data=result_to_mat_bytes(result),
        file_name=f"{result.solver.lower()}_result.mat",
        mime="application/octet-stream",
    )


if __name__ == "__main__":
    main()
