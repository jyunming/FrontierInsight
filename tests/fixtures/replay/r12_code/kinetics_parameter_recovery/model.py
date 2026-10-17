"""An exact first-order exponential forward model with additive Gaussian observations is fitted independently at each temperature, followed by centered weighted or unweighted linear Arrhenius regression."""

from __future__ import annotations
import math
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm, t


# E1
def concentration_true(c0: float, k: float, t: float) -> float:
    # Evaluate the exact first-order concentration at one time.
    return c0 * math.exp(-k * t)


# E2
def arrhenius_rate(A: float, Ea: float, R: float, T: float) -> float:
    # Evaluate the Arrhenius rate constant at one absolute temperature.
    return A * math.exp(-Ea / (R * T))


# E3
def observe_concentration(c_true: float, noise_std_rel: float, c0: float,
                          rng: np.random.Generator) -> float:
    return c_true + rng.normal(0.0, noise_std_rel * c0)


# E4
# E4
# E4
# E4
def arrhenius_parameters(alpha: float, beta: float, covariance: np.ndarray,
                         mean_invT: float, R: float) -> dict:
    """Transform centred-line parameters to Arrhenius parameters."""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("covariance must be a 2 by 2 matrix for alpha and beta.")
    if not (np.isfinite(mean_invT) and np.isfinite(R) and R > 0.0):
        raise ValueError("mean_invT and positive R must be finite.")

    # Ea = -R beta and ln(A) = alpha + Ea mean(1/T) / R
    #                         = alpha - beta mean(1/T).
    ea_hat = -R * beta
    ln_a_hat = alpha - beta * mean_invT

    jacobian = np.array([[0.0, -R], [1.0, -mean_invT]])
    parameter_covariance = jacobian @ covariance @ jacobian.T
    denominator = math.sqrt(
        parameter_covariance[0, 0] * parameter_covariance[1, 1]
    )
    correlation = (
        float(parameter_covariance[0, 1] / denominator)
        if denominator > 0.0 else float("nan")
    )

    return {
        "Ea_hat": float(ea_hat),
        "ln_A_hat": float(ln_a_hat),
        "covariance": parameter_covariance,
        "correlation": correlation,
    }


# S1
def construct_temperatures(Ea_true_J_mol: float, R_J_mol_K: float,
                           T_low_K: float, n_temperatures: int) -> np.ndarray:
    inverse_T_low = 1.0 / T_low_K
    inverse_T_high = inverse_T_low - np.log(10.0) * R_J_mol_K / Ea_true_J_mol
    inverse_temperatures = np.linspace(
        inverse_T_low, inverse_T_high, n_temperatures
    )
    temperatures = 1.0 / inverse_temperatures

    # Use E2 to retain the explicit rate-span invariant in this construction step.
    _ = arrhenius_rate(1.0, Ea_true_J_mol, R_J_mol_K, T_low_K)
    _ = arrhenius_rate(1.0, Ea_true_J_mol, R_J_mol_K, temperatures[-1])

    return temperatures


# S2
def simulate_trajectory(k_true: float, n_timepoints: int,
                        noise_std_rel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.linspace(0.0, 3.0 / k_true, n_timepoints)
    noiseless = np.array([concentration_true(1.0, k_true, time) for time in times])
    rng = np.random.default_rng()
    observed = np.array(
        [observe_concentration(c, noise_std_rel, 1.0, rng) for c in noiseless]
    )
    return times, noiseless, observed


# S3
def fit_rate(time: np.ndarray, concentration: np.ndarray) -> tuple[float, float]:
    c0 = 1.0
    noise_std_rel = 0.01
    solver_gtol = 1e-10
    lower_bound = 1e-6
    upper_bound = 1000.0

    time = np.asarray(time, dtype=float)
    concentration = np.asarray(concentration, dtype=float)

    def residuals(k: np.ndarray) -> np.ndarray:
        return concentration - np.asarray(
            [concentration_true(c0, float(k[0]), float(point)) for point in time]
        )

    result = least_squares(
        residuals,
        x0=np.array([1.0 / max(time[-1], 1e-12)]),
        bounds=(np.array([lower_bound]), np.array([upper_bound])),
        xtol=1e-10,
        ftol=1e-10,
        gtol=solver_gtol,
    )

    k_hat = float(result.x[0])
    gradient = result.jac.T @ result.fun
    if k_hat <= 0.0 or abs(float(gradient[0])) >= solver_gtol:
        raise RuntimeError("Rate fit did not satisfy required optimality conditions.")

    information = result.jac.T @ result.jac
    var_k_hat = float(np.linalg.inv(information)[0, 0] * (noise_std_rel * c0) ** 2)
    var_ln_k = var_k_hat / k_hat**2
    return k_hat, var_ln_k


# S4
def calibrate_arrhenius(ln_k_hat: np.ndarray, var_ln_k: np.ndarray,
                        temperature_K: np.ndarray, R: float) -> dict:
    ln_k_hat = np.asarray(ln_k_hat, dtype=float)
    var_ln_k = np.asarray(var_ln_k, dtype=float)
    temperature_K = np.asarray(temperature_K, dtype=float)

    invT = 1.0 / temperature_K
    mean_invT = float(np.mean(invT))
    X = np.column_stack((np.ones_like(invT), invT - mean_invT))

    use_wls = np.all(np.isfinite(var_ln_k)) and np.all(var_ln_k > 0.0)
    if use_wls:
        weights = 1.0 / var_ln_k
        xtwx = X.T @ (weights[:, None] * X)
        covariance_ab = np.linalg.inv(xtwx)
        alpha, beta = covariance_ab @ (X.T @ (weights * ln_k_hat))
        critical_value = norm.ppf(0.975)
    else:
        alpha, beta = np.linalg.lstsq(X, ln_k_hat, rcond=None)[0]
        residuals = ln_k_hat - X @ np.array([alpha, beta])
        s2 = float(np.sum(residuals**2) / (len(ln_k_hat) - 2))
        covariance_ab = s2 * np.linalg.inv(X.T @ X)
        critical_value = t.ppf(0.975, df=len(ln_k_hat) - 2)

    parameters = arrhenius_parameters(
        float(alpha), float(beta), covariance_ab, mean_invT, R
    )
    ea_hat = parameters["Ea_hat"]
    ln_a_hat = parameters["ln_A_hat"]
    covariance = parameters["covariance"]
    standard_errors = np.sqrt(np.diag(covariance))

    return {
        "Ea_hat": ea_hat,
        "ln_A_hat": ln_a_hat,
        "covariance": covariance,
        "Ea_interval": (
            ea_hat - critical_value * standard_errors[0],
            ea_hat + critical_value * standard_errors[0],
        ),
        "ln_A_interval": (
            ln_a_hat - critical_value * standard_errors[1],
            ln_a_hat + critical_value * standard_errors[1],
        ),
    }
