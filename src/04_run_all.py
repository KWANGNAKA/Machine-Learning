"""
04_run_all.py

Run every part of the project:
1) analytical method
2) simulation
3) learning curves with and without noise
4) bias-variance decomposition visualization
5) Ein / Eout graphs
"""

from __future__ import annotations

from importlib import import_module

import pandas as pd


analytical = import_module("01_analytical_method")
simulation = import_module("02_simulation")
learning_curve = import_module("03_learning_curve")
visualization = import_module("05_bias_variance_visualization")
ein_eout = import_module("06_ein_eout_graph")


def main() -> None:
    pd.set_option("display.precision", 8)

    print("\n=== Analytical method ===")
    analytical_df = analytical.run_analytical()
    print(
        analytical_df[
            [
                "target",
                "model",
                "g_bar(x)",
                "bias",
                "variance",
                "expected_out_error",
            ]
        ]
    )

    print("\n=== Simulation ===")
    simulation_df = simulation.run_simulation(
        n_datasets=300_000,
        seed=42,
    )
    print(simulation_df)

    print("\n=== Learning curves ===")
    lc_df = learning_curve.run_learning_curves(
        noise_values=[0.0, 0.30],
    )
    print(lc_df.head(12))

    print("\n=== Bias-Variance decomposition figure ===")
    fig_path, csv_path = visualization.plot_bias_variance_decomposition(
        noise_std=0.20,
        show=False,
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {csv_path}")

    print("\n=== Ein / Eout graphs ===")
    ein_eout_df = ein_eout.run_all_ein_eout(
        noise_values=[0.0, 0.30],
    )
    print(ein_eout_df.head(12))

    print("\nDone. See the results/ and figures/ folders.")


if __name__ == "__main__":
    main()