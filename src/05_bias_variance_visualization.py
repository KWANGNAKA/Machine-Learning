"""
05_bias_variance_visualization.py

Create a Bias-Variance Decomposition figure similar to the lecture-style plot:
- Green curve: true target function f(x)
- Gray curves: hypotheses learned from many random training sets
- Red dashed curve: average hypothesis g_bar(x)
- Red transparent band: pointwise standard deviation of hypotheses around g_bar(x)

This script uses least squares / normal equation through 00_models.py.
No gradient descent is used.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

models = import_module("00_models")

ROOT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS = ["sin_pi_x", "x_squared"]
MODELS = ["constant", "linear", "origin"]

TARGET_LABELS = {
    "sin_pi_x": r"sin($\pi$ x)",
    "x_squared": r"x^2",
}

MODEL_LABELS = {
    "constant": "constant",
    "linear": "linear",
    "origin": "linear_origin",
}


def make_one_hypothesis(
    x_grid: np.ndarray,
    target_name: str,
    model_name: str,
    rng: np.random.Generator,
    noise_std: float,
    n_train: int = 2,
) -> np.ndarray:
    """
    Randomly generate one training set, fit one model, and predict on x_grid.

    x_train is sampled from Uniform[-1, 1].
    y_train = f(x_train) + Gaussian noise.
    """
    # For a readable visualization, avoid nearly singular two-point designs.
    # Example: in the full linear model, x1 very close to x2 makes the slope explode.
    # This filtering only affects the plotted figure, not the analytical/simulation tables.
    while True:
        x_train = rng.uniform(-1.0, 1.0, size=n_train)
        if model_name == "linear" and n_train == 2 and abs(x_train[0] - x_train[1]) < 0.08:
            continue
        if model_name == "origin" and np.sum(x_train**2) < 0.03:
            continue
        break

    y_true = models.target_function(x_train, target_name)
    noise = rng.normal(loc=0.0, scale=noise_std, size=n_train)
    y_train = y_true + noise

    theta = models.fit_model(x_train, y_train, model_name)
    y_pred = models.predict(x_grid, theta, model_name)
    return y_pred


def sample_hypotheses(
    x_grid: np.ndarray,
    target_name: str,
    model_name: str,
    rng: np.random.Generator,
    noise_std: float,
    n_hypotheses: int,
    n_train: int = 2,
) -> np.ndarray:
    """Create many hypotheses from many random training sets."""
    predictions: list[np.ndarray] = []

    while len(predictions) < n_hypotheses:
        y_pred = make_one_hypothesis(
            x_grid=x_grid,
            target_name=target_name,
            model_name=model_name,
            rng=rng,
            noise_std=noise_std,
            n_train=n_train,
        )

        # Skip extremely unstable numerical results, if any.
        if np.all(np.isfinite(y_pred)):
            predictions.append(y_pred)

    return np.vstack(predictions)


def plot_bias_variance_decomposition(
    noise_std: float = 0.20,
    n_train: int = 2,
    n_examples: int = 80,
    n_average: int = 1800,
    seed: int = 2026,
    show: bool = False,
) -> tuple[Path, Path]:
    """
    Create and save the decomposition figure.

    The left panel of each model shows individual hypotheses.
    The right panel shows the average hypothesis and the variance band.
    """
    rng = np.random.default_rng(seed)
    x_grid = np.linspace(-1.0, 1.0, 500)

    y_min, y_max = -2.0, 2.0
    fig, axes = plt.subplots(
        nrows=2,
        ncols=6,
        figsize=(18, 9),
        sharex=True,
        sharey=True,
    )

    rows_for_csv: list[dict[str, float | str]] = []

    for row_index, target_name in enumerate(TARGETS):
        target_y = models.target_function(x_grid, target_name)

        for model_index, model_name in enumerate(MODELS):
            left_ax = axes[row_index, model_index * 2]
            right_ax = axes[row_index, model_index * 2 + 1]

            example_preds = sample_hypotheses(
                x_grid=x_grid,
                target_name=target_name,
                model_name=model_name,
                rng=rng,
                noise_std=noise_std,
                n_hypotheses=n_examples,
                n_train=n_train,
            )
            average_preds = sample_hypotheses(
                x_grid=x_grid,
                target_name=target_name,
                model_name=model_name,
                rng=rng,
                noise_std=noise_std,
                n_hypotheses=n_average,
                n_train=n_train,
            )

            g_bar = np.mean(average_preds, axis=0)
            g_std = np.std(average_preds, axis=0)

            # Numeric summary based on the same grid.
            bias = float(np.mean((g_bar - target_y) ** 2))
            variance = float(np.mean(g_std**2))
            expected_error = bias + variance
            rows_for_csv.append(
                {
                    "target": target_name,
                    "model": model_name,
                    "noise_std": noise_std,
                    "n_train": n_train,
                    "bias_grid_estimate": bias,
                    "variance_grid_estimate": variance,
                    "bias_plus_variance": expected_error,
                }
            )

            # Left panel: many learned hypotheses.
            for y_pred in example_preds:
                left_ax.plot(x_grid, y_pred, color="gray", alpha=0.18, linewidth=0.8)
            left_ax.plot(x_grid, target_y, color="limegreen", linewidth=2.6, label="Target")
            left_ax.plot(x_grid, g_bar, color="red", linestyle="--", linewidth=2.2, label="Average")

            # Right panel: average hypothesis and one-standard-deviation variance band.
            lower = np.clip(g_bar - g_std, y_min, y_max)
            upper = np.clip(g_bar + g_std, y_min, y_max)
            right_ax.fill_between(x_grid, lower, upper, color="red", alpha=0.25)
            right_ax.plot(x_grid, target_y, color="limegreen", linewidth=2.6)
            right_ax.plot(x_grid, g_bar, color="red", linestyle="--", linewidth=2.2)

            title = f"Target: {TARGET_LABELS[target_name]} — Model: {MODEL_LABELS[model_name]}"
            left_ax.set_title(title, fontsize=11)

            for ax in (left_ax, right_ax):
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.45)
                ax.set_xlabel("x")
            left_ax.set_ylabel("y")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.96))
    fig.suptitle(
        f"Bias-Variance Decomposition (N={n_train} Samples, Noise Std: {noise_std})",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])

    fig_path = FIGURES_DIR / f"bias_variance_decomposition_noise_{noise_std:.2f}.png"
    csv_path = RESULTS_DIR / f"bias_variance_decomposition_noise_{noise_std:.2f}.csv"

    fig.savefig(fig_path, dpi=170)
    pd.DataFrame(rows_for_csv).to_csv(csv_path, index=False)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig_path, csv_path


def main() -> None:
    fig_path, csv_path = plot_bias_variance_decomposition(noise_std=0.20, show=True)
    print(f"Saved figure to: {fig_path}")
    print(f"Saved summary to: {csv_path}")


if __name__ == "__main__":
    main()
