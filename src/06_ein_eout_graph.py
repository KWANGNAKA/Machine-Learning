"""
06_ein_eout_graph.py

This file creates Ein/Eout graphs for three models:
1) constant model: h(x)=b
2) linear model: h(x)=a*x+b
3) linear-through-origin model: h(x)=a*x

Ein  = in-sample error, measured on training data
Eout = out-of-sample error, measured on unseen test data

The experiment is repeated many times and averaged
to reduce randomness.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Import helper functions from 00_models.py
# We use import_module because the file name starts with a number.
models = import_module("00_models")


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


TARGETS = ["sin_pi_x", "x_squared"]
MODELS = ["constant", "linear", "origin"]

TRAIN_SIZES = [2, 3, 5, 10, 20, 50, 100, 200]


def run_ein_eout_experiment(
    target_name: str,
    noise_std: float = 0.0,
    repeats: int = 500,
    test_size: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Run Ein/Eout experiment for one target function and one noise level.

    Parameters
    ----------
    target_name:
        Target function name, either "sin_pi_x" or "x_squared".

    noise_std:
        Standard deviation of Gaussian noise.
        If noise_std = 0.0, there is no noise.
        If noise_std = 0.30, data are generated as y = f(x) + noise.

    repeats:
        Number of repetitions for each training size and model.

    test_size:
        Number of test points used to estimate Eout.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A table containing target, noise, model, n_train, Ein, and Eout.
    """

    rng = np.random.default_rng(seed)
    rows = []

    for n_train in TRAIN_SIZES:
        for model_name in MODELS:
            ein_values = []
            eout_values = []

            for _ in range(repeats):
                # -----------------------------
                # 1. Generate training data
                # -----------------------------
                x_train = rng.uniform(-1.0, 1.0, size=n_train)

                y_train_true = models.target_function(x_train, target_name)

                noise_train = rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=n_train,
                )

                y_train = y_train_true + noise_train

                # -----------------------------
                # 2. Fit model using least squares
                # -----------------------------
                theta = models.fit_model(
                    x_train=x_train,
                    y_train=y_train,
                    model_name=model_name,
                )

                # -----------------------------
                # 3. Compute Ein
                # -----------------------------
                y_train_pred = models.predict(
                    x=x_train,
                    theta=theta,
                    model_name=model_name,
                )

                ein = np.mean((y_train_pred - y_train) ** 2)
                ein_values.append(ein)

                # -----------------------------
                # 4. Generate test data
                # -----------------------------
                x_test = rng.uniform(-1.0, 1.0, size=test_size)

                y_test_true = models.target_function(x_test, target_name)

                noise_test = rng.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=test_size,
                )

                y_test = y_test_true + noise_test

                # -----------------------------
                # 5. Compute Eout
                # -----------------------------
                y_test_pred = models.predict(
                    x=x_test,
                    theta=theta,
                    model_name=model_name,
                )

                eout = np.mean((y_test_pred - y_test) ** 2)
                eout_values.append(eout)

            rows.append(
                {
                    "target": target_name,
                    "noise_std": noise_std,
                    "model": model_name,
                    "n_train": n_train,
                    "Ein": float(np.mean(ein_values)),
                    "Eout": float(np.mean(eout_values)),
                }
            )

    return pd.DataFrame(rows)


def plot_ein_eout(
    df: pd.DataFrame,
    target_name: str,
    noise_std: float,
) -> Path:
    """
    Plot Ein and Eout curves.

    One figure contains three subplots:
    - constant model
    - linear model
    - linear through origin model
    """

    figure_path = FIGURES_DIR / f"ein_eout_{target_name}_noise_{noise_std:.2f}.png"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 4),
        sharey=True,
    )

    for ax, model_name in zip(axes, MODELS):
        model_df = df[
            (df["target"] == target_name)
            & (df["noise_std"] == noise_std)
            & (df["model"] == model_name)
        ]

        ax.plot(
            model_df["n_train"],
            model_df["Ein"],
            marker="o",
            linestyle="-",
            label="Ein",
        )

        ax.plot(
            model_df["n_train"],
            model_df["Eout"],
            marker="s",
            linestyle="--",
            label="Eout",
        )

        ax.set_xscale("log")
        ax.set_title(f"Model: {model_name}")
        ax.set_xlabel("Number of training examples")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("Mean squared error")

    fig.suptitle(
        f"Ein vs Eout: target = {target_name}, noise std = {noise_std}",
        fontsize=14,
    )

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    return figure_path


def run_all_ein_eout(
    noise_values: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run Ein/Eout experiment for all target functions and noise levels.
    Save results to CSV and save graphs to figures folder.
    """

    if noise_values is None:
        noise_values = [0.0, 0.30]

    all_results = []

    for target_name in TARGETS:
        for noise_std in noise_values:
            df = run_ein_eout_experiment(
                target_name=target_name,
                noise_std=noise_std,
                repeats=500,
                test_size=2000,
                seed=123,
            )

            all_results.append(df)

            figure_path = plot_ein_eout(
                df=df,
                target_name=target_name,
                noise_std=noise_std,
            )

            print(f"Saved figure: {figure_path}")

    result_df = pd.concat(all_results, ignore_index=True)

    csv_path = RESULTS_DIR / "ein_eout_results.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved result CSV: {csv_path}")

    return result_df


if __name__ == "__main__":
    result = run_all_ein_eout()
    print(result)