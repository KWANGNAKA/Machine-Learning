"""
01_analytical_method.py

Analytical bias-variance calculation for three hypothesis sets:
1) constant model: h(x) = b
2) linear model: h(x) = a*x + b
3) linear-through-origin model: h(x) = a*x

Data generating process:
- choose two training inputs x1, x2 ~ Uniform[-1, 1]
- y_i = f(x_i)
- f(x) is either sin(pi*x) or x^2

Bias and variance definitions:
    g_bar(x) = E_D[g_D(x)]
    bias     = E_x[(g_bar(x) - f(x))^2]
    variance = E_x[E_D[(g_D(x) - g_bar(x))^2]]

For the hard expectations, this file evaluates the expected-value integrals
numerically with scipy.integrate. This is still analytical integration, not
Monte Carlo simulation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import pandas as pd
from scipy.integrate import dblquad, quad

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


TARGETS = ["sin_pi_x", "x_squared"]
MODELS = ["constant", "linear", "origin"]


def f(x: float, target_name: str) -> float:
    if target_name == "sin_pi_x":
        return math.sin(math.pi * x)
    if target_name == "x_squared":
        return x * x
    raise ValueError("target_name must be 'sin_pi_x' or 'x_squared'")


def uniform_expectation_1d(func: Callable[[float], float]) -> float:
    """E[func(X)] where X ~ Uniform[-1, 1]."""
    val, _ = quad(lambda x: 0.5 * func(x), -1.0, 1.0, epsabs=1e-10, epsrel=1e-10)
    return val


def uniform_expectation_2d(func: Callable[[float, float], float]) -> float:
    """E[func(X1, X2)] where X1, X2 iid Uniform[-1, 1]."""
    val, _ = dblquad(
        lambda x2, x1: 0.25 * func(x1, x2),
        -1.0,
        1.0,
        lambda x1: -1.0,
        lambda x1: 1.0,
        epsabs=1e-9,
        epsrel=1e-9,
    )
    return val


def common_moments(target_name: str) -> dict[str, float]:
    """Moments needed for bias calculations."""
    ef = uniform_expectation_1d(lambda x: f(x, target_name))
    ef2 = uniform_expectation_1d(lambda x: f(x, target_name) ** 2)
    exf = uniform_expectation_1d(lambda x: x * f(x, target_name))
    return {"E_f": ef, "E_f2": ef2, "E_xf": exf}


def constant_analytical(target_name: str) -> dict[str, float | str]:
    """
    h(x)=b, fitted with two samples.
    b_D = (f(x1)+f(x2))/2
    g_bar(x) = E[f(X)]
    """
    m = common_moments(target_name)
    var_f = m["E_f2"] - m["E_f"] ** 2
    bias = var_f
    variance = var_f / 2.0
    return {
        "target": target_name,
        "model": "constant",
        "g_bar(x)": f"{m['E_f']:.10f}",
        "a_bar": 0.0,
        "b_bar": m["E_f"],
        "bias": bias,
        "variance": variance,
        "expected_out_error": bias + variance,
        "method_note": "closed form from E[f] and Var(f)",
    }


def origin_slope(x1: float, x2: float, target_name: str) -> float:
    """
    h(x)=a*x fitted by least squares through the origin.
    a_D = (x1*f(x1)+x2*f(x2))/(x1^2+x2^2)
    """
    den = x1 * x1 + x2 * x2
    if den < 1e-14:
        # Removable point at (0,0). The exact value does not affect the integral.
        return math.pi if target_name == "sin_pi_x" else 0.0
    return (x1 * f(x1, target_name) + x2 * f(x2, target_name)) / den


def origin_analytical(target_name: str) -> dict[str, float | str]:
    """Analytical expected-value integration for h(x)=a*x."""
    m = common_moments(target_name)
    e_a = uniform_expectation_2d(lambda x1, x2: origin_slope(x1, x2, target_name))
    e_a2 = uniform_expectation_2d(lambda x1, x2: origin_slope(x1, x2, target_name) ** 2)

    # E[x] = 0 and E[x^2] = 1/3 for Uniform[-1,1]
    bias = (e_a**2) / 3.0 - 2.0 * e_a * m["E_xf"] + m["E_f2"]
    variance = (e_a2 - e_a**2) / 3.0
    return {
        "target": target_name,
        "model": "origin",
        "g_bar(x)": f"{e_a:.10f} x",
        "a_bar": e_a,
        "b_bar": 0.0,
        "bias": bias,
        "variance": variance,
        "expected_out_error": bias + variance,
        "method_note": "2D expected-value integral of slope a_D",
    }


def safe_sin_over_d(d: float) -> float:
    """sin(pi*d)/d with removable value at d=0."""
    if abs(d) < 1e-10:
        return math.pi
    return math.sin(math.pi * d) / d


def linear_sin_integrals() -> tuple[float, float, float]:
    """
    Analytical integration for the full linear interpolation model
    h(x)=a*x+b with target f(x)=sin(pi*x).

    Use change of variables:
        m = (x1+x2)/2, d = (x1-x2)/2
    The square [-1,1]^2 becomes |d| <= 1-|m|, with density 1/2.

    a = cos(pi*m) * sin(pi*d)/d
    b = -m*cos(pi*m)*sin(pi*d)/d + sin(pi*m)*cos(pi*d)
    """

    def e_a_integrand_m(m: float) -> float:
        L = 1.0 - abs(m)
        inner, _ = quad(lambda d: safe_sin_over_d(d), 0.0, L, epsabs=1e-10, epsrel=1e-10)
        return math.cos(math.pi * m) * inner

    def e_a2_integrand_m(m: float) -> float:
        L = 1.0 - abs(m)
        inner, _ = quad(lambda d: safe_sin_over_d(d) ** 2, 0.0, L, epsabs=1e-10, epsrel=1e-10)
        return (math.cos(math.pi * m) ** 2) * inner

    def e_b2_integrand_m(m: float) -> float:
        L = 1.0 - abs(m)
        cm = math.cos(math.pi * m)
        sm = math.sin(math.pi * m)

        def b_value(d: float) -> float:
            return -m * cm * safe_sin_over_d(d) + sm * math.cos(math.pi * d)

        inner, _ = quad(lambda d: b_value(d) ** 2, 0.0, L, epsabs=1e-10, epsrel=1e-10)
        return inner

    e_a, _ = quad(e_a_integrand_m, -1.0, 1.0, epsabs=1e-10, epsrel=1e-10)
    e_a2, _ = quad(e_a2_integrand_m, -1.0, 1.0, epsabs=1e-10, epsrel=1e-10)
    e_b2, _ = quad(e_b2_integrand_m, -1.0, 1.0, epsabs=1e-10, epsrel=1e-10)
    return e_a, e_a2, e_b2


def linear_analytical(target_name: str) -> dict[str, float | str]:
    """Analytical calculation for h(x)=a*x+b."""
    m = common_moments(target_name)

    if target_name == "x_squared":
        # With two noiseless samples on f(x)=x^2, the interpolating line is:
        # a_D = x1+x2, b_D = -x1*x2
        e_a = 0.0
        e_b = 0.0
        e_a2 = 2.0 / 3.0
        e_b2 = 1.0 / 9.0
        note = "closed form: a_D=x1+x2, b_D=-x1*x2"
    else:
        e_a, e_a2, e_b2 = linear_sin_integrals()
        e_b = 0.0
        note = "m,d transform expected-value integral for a_D and b_D"

    # g_bar(x) = E[a]x + E[b]
    # E[x]=0 and E[x^2]=1/3
    bias = (e_a**2) / 3.0 + e_b**2 - 2.0 * e_a * m["E_xf"] - 2.0 * e_b * m["E_f"] + m["E_f2"]
    variance = (e_a2 - e_a**2) / 3.0 + (e_b2 - e_b**2)

    return {
        "target": target_name,
        "model": "linear",
        "g_bar(x)": f"{e_a:.10f} x + {e_b:.10f}",
        "a_bar": e_a,
        "b_bar": e_b,
        "bias": bias,
        "variance": variance,
        "expected_out_error": bias + variance,
        "method_note": note,
    }


def run_analytical() -> pd.DataFrame:
    rows = []
    for target_name in TARGETS:
        rows.append(constant_analytical(target_name))
        rows.append(linear_analytical(target_name))
        rows.append(origin_analytical(target_name))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "analytical_results.csv", index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    pd.set_option("display.precision", 8)
    result = run_analytical()
    print(result[["target", "model", "g_bar(x)", "bias", "variance", "expected_out_error"]])
