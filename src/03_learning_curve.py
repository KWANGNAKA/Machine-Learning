"""
03_learning_curve.py

Learning-curve experiment comparing:
1) constant model: h(x)=b
2) linear model: h(x)=a*x+b
3) linear-through-origin model: h(x)=a*x

The program creates learning curves for both target functions:
- f(x)=sin(pi*x)
- f(x)=x^2

It runs both clean data and noisy data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from importlib import import_module

models = import_module("00_models")

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TARGETS = ["sin_pi_x", "x_squared"]
MODELS = ["constant", "linear", "origin"]
TRAIN_SIZES = [2, 3, 5, 10, 20, 50, 100, 200]


def run_one_learning_curve(
    target_name: str,
    noise_std: float = 0.0,
    repeats: int = 500,
    test_size: int = 2000,
    seed: int = 123,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for n_train in TRAIN_SIZES:
        for model_name in MODELS:
            train_errors = []
            test_errors = []

            for _ in range(repeats):
                x_train = rng.uniform(-1.0, 1.0, size=n_train)
                y_train_true = models.target_function(x_train, target_name)
                y_train = y_train_true + rng.normal(0.0, noise_std, size=n_train)

                theta = models.fit_model(x_train, y_train, model_name)

                y_train_pred = models.predict(x_train, theta, model_name)
                train_errors.append(np.mean((y_train_pred - y_train) ** 2))

                x_test = rng.uniform(-1.0, 1.0, size=test_size)
                y_test_true = models.target_function(x_test, target_name)
                y_test = y_test_true + rng.normal(0.0, noise_std, size=test_size)
                y_test_pred = models.predict(x_test, theta, model_name)
                test_errors.append(np.mean((y_test_pred - y_test) ** 2))

            rows.append(
                {
                    "target": target_name,
                    "noise_std": noise_std,
                    "n_train": n_train,
                    "model": model_name,
                    "train_mse": float(np.mean(train_errors)),
                    "test_mse": float(np.mean(test_errors)),
                }
            )

    return pd.DataFrame(rows)


def plot_learning_curve(df: pd.DataFrame, target_name: str, noise_std: float) -> Path:
    fig_path = FIGURES_DIR / f"learning_curve_{target_name}_noise_{noise_std:.2f}.png"

    plt.figure(figsize=(8, 5))
    for model_name in MODELS:
        d = df[(df["target"] == target_name) & (df["noise_std"] == noise_std) & (df["model"] == model_name)]
        plt.plot(d["n_train"], d["test_mse"], marker="o", label=f"{model_name} test MSE")

    plt.xscale("log")
    plt.xlabel("Number of training examples")
    plt.ylabel("Average test MSE")
    plt.title(f"Learning Curve: {target_name}, noise std = {noise_std}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()
    return fig_path


def run_learning_curves(noise_values: list[float] | None = None) -> pd.DataFrame:
    if noise_values is None:
        noise_values = [0.0, 0.30]

    all_results = []
    for target in TARGETS:
        for noise_std in noise_values:
            df = run_one_learning_curve(target, noise_std=noise_std)
            all_results.append(df)
            plot_learning_curve(df, target, noise_std)

    result = pd.concat(all_results, ignore_index=True)
    result.to_csv(RESULTS_DIR / "learning_curve_results.csv", index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    result_df = run_learning_curves()
    print(result_df.head(20))
