"""
00_models.py

Utility functions for the assignment:
- target functions
- model fitting with normal equation / least squares
- prediction

Models:
1) constant model: h(x) = b
2) linear model: h(x) = a*x + b
3) linear-through-origin model: h(x) = a*x
"""

from __future__ import annotations

import numpy as np


def target_function(x: np.ndarray, target_name: str) -> np.ndarray:
    """Return f(x) for the selected target function."""
    x = np.asarray(x, dtype=float)
    if target_name == "sin_pi_x":
        return np.sin(np.pi * x)
    if target_name == "x_squared":
        return x**2
    raise ValueError("target_name must be 'sin_pi_x' or 'x_squared'")


def design_matrix(x: np.ndarray, model_name: str) -> np.ndarray:
    """Create a design matrix for each model."""
    x = np.asarray(x, dtype=float).reshape(-1)
    if model_name == "constant":
        return np.ones((x.size, 1))
    if model_name == "linear":
        return np.column_stack([x, np.ones_like(x)])
    if model_name == "origin":
        return x.reshape(-1, 1)
    raise ValueError("model_name must be 'constant', 'linear', or 'origin'")


def fit_model(x_train: np.ndarray, y_train: np.ndarray, model_name: str) -> np.ndarray:
    """
    Fit model parameters using the normal equation / numpy least squares.

    Returned parameters:
    - constant: [b]
    - linear:   [a, b]
    - origin:   [a]
    """
    X = design_matrix(x_train, model_name)
    y = np.asarray(y_train, dtype=float).reshape(-1)

    # np.linalg.lstsq solves the normal-equation least-squares problem stably.
    # No gradient descent is used.
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return theta


def predict(x: np.ndarray, theta: np.ndarray, model_name: str) -> np.ndarray:
    """Predict y for the selected model."""
    X = design_matrix(x, model_name)
    return X @ theta
