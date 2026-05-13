# Python SDP Clustering Solvers

This folder contains Python implementations and a Streamlit interface for the
sublevel-set SDP clustering validation method from Meila (2018). The app does
not run k-means from scratch. It takes an existing clustering and checks whether
there is another equal-or-better clustering candidate that is very different
from it.

## Main Files

- `admm_solver.py`: ADMM implementation for the sublevel-set SDP. This is the
  main validated solver path and is intended for smaller dense problems.
- `cg_solver.py`: Conditional-gradient implementation for a larger-scale SDP
  formulation. This version is runnable but experimental because Python
  iterative eigensolver behavior can differ from Matlab.
- `solver_api.py`: API layer used by the app. It loads `.mat` files, validates
  inputs, runs ADMM or CG, computes summary metrics, and exports results.
- `streamlit_app.py`: Streamlit web interface for uploading data, running a
  solver, and reading the validation result.
- `validate_admm.py`: Compares the Python ADMM implementation against the saved
  Matlab benchmark.
- `validate_cg.py`: Compares Python CG summary metrics against Matlab output.
- `debug_cg_short.py`: Small diagnostic script for CG eigensolver behavior.

## Input `.mat` Format

The Streamlit app expects a MATLAB `.mat` file containing:

- `X0`: an `n x n` clustering matrix for the clustering being validated.
- `G`: an `n x n` centered Gram matrix with the same shape as `X0`.

For `X0`, if points `i` and `j` are in the same cluster of size `m`, then
`X0[i, j] = 1 / m`; otherwise `X0[i, j] = 0`. The number of clusters is inferred
from `trace(X0)`.

For `G`, if the raw data matrix is `Y` with one point per row:

```python
Y_centered = Y - Y.mean(axis=0, keepdims=True)
G = Y_centered @ Y_centered.T
```

## Streamlit App

Use `streamlit_app.py` as the Streamlit Community Cloud entry point:

```text
admm_ss_sdp/python_admm/streamlit_app.py
```

The app workflow is:

1. Upload a `.mat` file containing `X0` and `G`.
2. Select `ADMM` or `CG`.
3. Click `Run Solver`.
4. Read the objective, `Objective - K`, runtime, and matrix diagnostics.
5. Download the result `.mat` file if needed.

For ADMM, `Objective - K` is `<X0, X> - K`. Values near `0` mean the solver did
not find a very different equal-or-better candidate, which supports clustering
stability. More negative values mean the SDP found a less similar candidate, so
the uploaded clustering is less strongly certified.
