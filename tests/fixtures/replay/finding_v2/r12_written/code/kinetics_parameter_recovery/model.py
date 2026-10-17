"""An exact first-order exponential forward model with additive Gaussian observations is fitted independently at each temperature, followed by centered weighted or unweighted linear Arrhenius regression."""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares


# E1
def concentration_true(c0: float, k: float, t: float) -> float:
    return c0 * np.exp(-k * t)


# E2
def arrhenius_rate(A: float, Ea: float, R: float, T: float) -> float:
    return A * np.exp(-Ea / (R * T))


# E3
def observe_concentration(c_true: float, noise_std_rel: float, c0: float, rng: np.random.Generator) -> float:
    return float(c_true + rng.normal(0.0, noise_std_rel * c0))


# E4
def arrhenius_parameters(alpha: float, beta: float, R: float, mean_invT: float) -> tuple[float, float]:
    Ea_hat = -R * beta
    ln_A_hat = alpha + Ea_hat * mean_invT / R
    return float(Ea_hat), float(ln_A_hat)


# S1
def construct_temperatures(Ea_true_J_mol: float, R_J_mol_K: float, T_low_K: float, n_temperatures: int) -> np.ndarray:
    inverse_temperatures = np.linspace(
        1.0 / T_low_K,
        1.0 / T_low_K - np.log(10.0) * R_J_mol_K / Ea_true_J_mol,
        n_temperatures,
    )
    temperatures = 1.0 / inverse_temperatures
    _ = arrhenius_rate(1.0, Ea_true_J_mol, R_J_mol_K, temperatures[0])
    return temperatures


# S2
def simulate_trajectory(k_true: float, n_timepoints: int, noise_std_rel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 3.0 / k_true, n_timepoints)
    noiseless_concentration = np.array(
        [concentration_true(1.0, k_true, t) for t in time]
    )
    rng = np.random.default_rng()
    observed_concentration = np.array(
        [
            observe_concentration(c_true, noise_std_rel, 1.0, rng)
            for c_true in noiseless_concentration
        ]
    )
    return time, noiseless_concentration, observed_concentration


# S3
def fit_rate(time: np.ndarray, concentration: np.ndarray) -> tuple[float, float]:
    # Perform one bounded nonlinear concentration fit and return k_hat with propagated log-rate variance.
    c0 = 1.0
    noise_std_rel = 0.01
    solver_gtol = 1e-10
    k_lower_bound = 1e-6
    k_upper_bound = 1000.0

    time = np.asarray(time, dtype=float)
    concentration = np.asarray(concentration, dtype=float)

    if time.ndim != 1 or concentration.ndim != 1 or time.size != concentration.size:
        raise ValueError("time and concentration must be one-dimensional arrays of equal length.")
    if time.size < 2:
        raise ValueError("At least two concentration observations are required.")

    def residuals(k: np.ndarray) -> np.ndarray:
        return concentration - np.array(
            [concentration_true(c0, float(k[0]), float(t)) for t in time]
        )

    result = least_squares(
        residuals,
        x0=np.array([max(k_lower_bound * 2.0, 1.0 / max(time.max(), 1e-12))]),
        bounds=(k_lower_bound, k_upper_bound),
        xtol=1e-10,
        ftol=1e-10,
        gtol=solver_gtol,
    )

    k_hat = float(result.x[0])
    if not (k_hat > 0.0) or abs(result.optimality) >= solver_gtol:
        raise RuntimeError("Rate fit did not satisfy first-order optimality or positivity.")

    jtj = float(result.jac.T @ result.jac)
    if jtj <= 0.0:
        raise RuntimeError("Rate-fit Jacobian is singular.")

    var_k_hat = (noise_std_rel * c0) ** 2 / jtj
    var_ln_k = var_k_hat / k_hat**2
    return k_hat, float(var_ln_k)


# S4
def calibrate_arrhenius(ln_k_hat: np.ndarray, var_ln_k: np.ndarray, temperature_K: np.ndarray) -> dict:
    ln_k_hat = np.asarray(ln_k_hat, dtype=float)
    var_ln_k = np.asarray(var_ln_k, dtype=float)
    temperature_K = np.asarray(temperature_K, dtype=float)

    if ln_k_hat.shape != (5,) or var_ln_k.shape != (5,) or temperature_K.shape != (5,):
        raise ValueError("ln_k_hat, var_ln_k, and temperature_K must each contain five values.")
    if (not np.all(np.isfinite(ln_k_hat)) or
            not np.all(np.isfinite(var_ln_k)) or
            not np.all(np.isfinite(temperature_K)) or
            np.any(var_ln_k <= 0.0) or np.any(temperature_K <= 0.0)):
        raise ValueError("Inputs must be finite, with positive variances and temperatures.")

    R = 8.314462618
    invT = 1.0 / temperature_K
    mean_invT = float(np.mean(invT))
    X = np.column_stack((np.ones(5), invT - mean_invT))
    weights = 1.0 / var_ln_k

    xtwx = X.T @ (weights[:, None] * X)
    covariance_ab = np.linalg.inv(xtwx)
    alpha, beta = covariance_ab @ (X.T @ (weights * ln_k_hat))

    Ea_hat, ln_A_hat = arrhenius_parameters(
        float(alpha), float(beta), R, mean_invT
    )

    jacobian = np.array([[0.0, -R], [1.0, -mean_invT]])
    covariance = jacobian @ covariance_ab @ jacobian.T
    standard_errors = np.sqrt(np.diag(covariance))
    z_critical = 1.959963984540054

    return {
        "Ea_hat": Ea_hat,
        "ln_A_hat": ln_A_hat,
        "covariance": covariance,
        "Ea_interval_95": (
            Ea_hat - z_critical * standard_errors[0],
            Ea_hat + z_critical * standard_errors[0],
        ),
        "ln_A_interval_95": (
            ln_A_hat - z_critical * standard_errors[1],
            ln_A_hat + z_critical * standard_errors[1],
        ),
    }
