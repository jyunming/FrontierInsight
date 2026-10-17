import numpy as np

from kinetics_parameter_recovery import model


def test_fit_rate_propagates_the_requested_observation_noise_level():
    time = np.linspace(0.0, 4.0, 9)
    concentration = np.exp(-0.5 * time)
    low_noise = 0.005
    high_noise = 0.02

    try:
        _, low_variance = model.fit_rate(
            time, concentration, noise_std_rel=low_noise
        )
        _, high_variance = model.fit_rate(
            time, concentration, noise_std_rel=high_noise
        )
    except Exception as exc:
        assert False, (
            "fit_rate must accept the requested relative observation noise so its "
            f"propagated log-rate variance follows the noise sweep; got {exc!r}."
        )

    expected_ratio = (high_noise / low_noise) ** 2
    actual_ratio = high_variance / low_variance
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-12, atol=0.0), (
        "fit_rate's propagated log-rate variance does not scale with the supplied "
        "relative observation noise level."
    )
