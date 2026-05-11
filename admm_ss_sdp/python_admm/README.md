# Python Solver Demo

This folder contains the Python translation and Streamlit demo for the
sublevel-set SDP clustering solvers.

## Files

- `admm_solver.py`: Python implementation of the ADMM solver path from
  `proj_psd_largescale.m`, `dual_admm3c_test.m`, and `dual_admm3c.m`.
- `cg_solver.py`: Python implementation of the conditional-gradient solver path
  from `cg_ss_test.m` and `cg_ss.m`.
- `solver_api.py`: Small API layer used by the Streamlit app. It loads `.mat`
  files, validates `X0` and `G`, runs solvers, computes metrics, and exports
  result `.mat` files.
- `streamlit_app.py`: Upload-and-run web demo.
- `validate_admm.py`: Validates Python ADMM against the saved Matlab benchmark.
- `validate_cg.py`: Runs Python CG and compares summary metrics against Matlab.
- `debug_cg_short.py`: Short CG eigensolver diagnostic.

## Solver Status

- ADMM: validated against Matlab on `X0_200.mat`; results match to numerical
  precision.
- CG: translated and runnable, but marked experimental because iterative
  eigensolver choices diverge from Matlab after the first few iterations.

## Local Run

From the repository root:

```powershell
D:/Anaconda/python.exe -m streamlit run admm_ss_sdp/python_admm/streamlit_app.py --global.developmentMode false --browser.gatherUsageStats false
```

Then open:

```text
http://localhost:8501
```

Upload a `.mat` file containing variables:

- `X0`
- `G`

For the included sample, use:

```text
admm_ss_sdp/X0_200.mat
```

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Select the GitHub repo and branch.
4. Set the main file path to:

```text
admm_ss_sdp/python_admm/streamlit_app.py
```

5. Keep `requirements.txt` at the repository root.

The root `requirements.txt` is intentionally small:

```text
streamlit
numpy<2
scipy
```
