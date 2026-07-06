"""
02_simulation.py

Monte Carlo simulation of bias and variance for the same setting as
01_analytical_method.py.

The code estimates the distribution of fitted parameters over many random
training sets D. Because every selected model is linear in x, the final
bias/variance can be estimated efficiently from the sampled parameters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TARGETS = ["sin_pi_x", "x_squared"]
MODELS = ["constant", "linear", "origin"]


def target_function(x: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "sin_pi_x":
        return np.sin(np.pi * x)
    if target_name == "x_squared":
        return x**2
    raise ValueError("target_name must be 'sin_pi_x' or 'x_squared'")


def sample_parameters(target_name: str, model_name: str, n_datasets: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample fitted a,b from many random 2-point datasets."""
    x1 = rng.uniform(-1.0, 1.0, size=n_datasets)
    x2 = rng.uniform(-1.0, 1.0, size=n_datasets)
    y1 = target_function(x1, target_name)
    y2 = target_function(x2, target_name)

    if model_name == "constant":
        a = np.zeros(n_datasets)
        b = (y1 + y2) / 2.0
        return a, b

    if model_name == "origin":
        den = x1**2 + x2**2
        a = (x1 * y1 + x2 * y2) / den
        b = np.zeros(n_datasets)
        return a, b

    if model_name == "linear":
        den = x1 - x2
        # Probability of exact equality is zero, but this protects against rare floating cases.
        den = np.where(np.abs(den) < 1e-14, np.nan, den)
        a = (y1 - y2) / den
        b = (x1 * y2 - x2 * y1) / den
        mask = np.isfinite(a) & np.isfinite(b)
        return a[mask], b[mask]

    raise ValueError("model_name must be 'constant', 'linear', or 'origin'")


def simulate_bias_variance(target_name: str, model_name: str, n_datasets: int = 300_000, seed: int = 42) -> dict[str, float | str]:
    rng = np.random.default_rng(seed)
    a, b = sample_parameters(target_name, model_name, n_datasets, rng)

    # Estimate g_bar(x) = E[a]x + E[b]
    a_bar = float(np.mean(a))
    b_bar = float(np.mean(b))

    # Estimate bias with a dense grid over x ~ Uniform[-1,1]
    x_grid = np.linspace(-1.0, 1.0, 5001)
    f_grid = target_function(x_grid, target_name)
    g_bar_grid = a_bar * x_grid + b_bar
    bias = float(np.mean((g_bar_grid - f_grid) ** 2))

    # Since E[x]=0 and E[x^2]=1/3, average variance is:
    # Var(a*x+b) averaged over x = Var(a)/3 + Var(b)
    variance = float(np.var(a, ddof=0) / 3.0 + np.var(b, ddof=0))

    return {
        "target": target_name,
        "model": model_name,
        "n_datasets": int(len(a)),
        "a_bar_sim": a_bar,
        "b_bar_sim": b_bar,
        "bias_sim": bias,
        "variance_sim": variance,
        "expected_out_error_sim": bias + variance,
    }


def run_simulation(n_datasets: int = 300_000, seed: int = 42) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        for model in MODELS:
            rows.append(simulate_bias_variance(target, model, n_datasets=n_datasets, seed=seed))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "simulation_results.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    pd.set_option("display.precision", 8)
    result = run_simulation()
    print(result)
