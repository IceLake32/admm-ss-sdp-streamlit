"""Generate demo clustering examples for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples"

COLORS = {
    1: "#f2c200",
    2: "#9ee7f2",
}


def problem_from_points(points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the app's X0 and G matrices from data points and cluster labels."""
    _, inverse = np.unique(labels, return_inverse=True)
    cluster_sizes = np.bincount(inverse)
    same_cluster = inverse[:, None] == inverse[None, :]
    x0 = same_cluster.astype(float) / cluster_sizes[inverse][:, None]

    centered = points - points.mean(axis=0, keepdims=True)
    g = centered @ centered.T
    return x0, g


def save_example(stem: str, points: np.ndarray, labels: np.ndarray, title: str) -> None:
    x0, g = problem_from_points(points, labels)
    payload = {
        "Y": points,
        "labels": labels.reshape(-1, 1),
        "clustering": labels.reshape(-1, 1),
        "X0": x0,
        "G": g,
    }

    savemat(OUT / f"{stem}.mat", payload)
    np.savez(OUT / f"{stem}.npz", **payload)

    with (OUT / f"{stem}.csv").open("w", encoding="utf-8", newline="") as file:
        file.write("x1,x2,label\n")
        for (x1, x2), label in zip(points, labels):
            file.write(f"{x1:.6g},{x2:.6g},{label}\n")

    is_good_example = stem.startswith("good-")
    fig_size = (4.2, 4.2) if is_good_example else (4.8, 4.2)
    fig, ax = plt.subplots(figsize=fig_size, dpi=220)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    x_pad = max((x_max - x_min) * 0.16, 0.45)
    y_pad = max((y_max - y_min) * 0.40, 0.55)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if is_good_example:
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()
        x_center = (x_left + x_right) / 2
        y_center = (y_bottom + y_top) / 2
        half_span = max(x_right - x_left, y_top - y_bottom) / 2
        ax.set_xlim(x_center - half_span, x_center + half_span)
        ax.set_ylim(y_center - half_span, y_center + half_span)

    marker_size = 64 if is_good_example else 92
    for label in sorted(set(labels)):
        cluster_points = points[labels == label]
        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=marker_size,
            color=COLORS[int(label)],
            edgecolors="white",
            linewidths=0.7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, color="white", fontsize=16, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.06, top=0.84)
    fig.savefig(OUT / f"{stem}.png", transparent=True)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)

    good_points = np.array(
        [
            [-3.42, -0.48],
            [-3.38, 0.38],
            [-2.95, -0.06],
            [-2.52, 0.42],
            [-2.48, -0.50],
            [2.32, -0.42],
            [2.45, 0.44],
            [2.90, 0.02],
            [3.12, 0.58],
            [3.42, 0.24],
            [3.48, -0.44],
            [2.88, -0.66],
            [3.72, -0.08],
        ],
        dtype=float,
    )
    good_labels = np.array([1] * 5 + [2] * 8)

    bad_points = np.array(
        [
            [-0.80, 0.62],
            [-0.62, -0.35],
            [-0.42, 0.18],
            [-0.20, -0.72],
            [0.02, 0.52],
            [0.20, -0.12],
            [0.38, 0.78],
            [0.55, -0.55],
            [0.76, 0.20],
            [0.94, -0.02],
            [1.10, 0.58],
            [1.28, -0.42],
            [1.48, 0.08],
        ],
        dtype=float,
    )
    bad_labels = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2])

    save_example("good-n13-k2", good_points, good_labels, "Example A: good clustering")
    save_example("bad-n13-k2", bad_points, bad_labels, "Example B: bad clustering")

    print(f"Created examples in {OUT}")


if __name__ == "__main__":
    main()
