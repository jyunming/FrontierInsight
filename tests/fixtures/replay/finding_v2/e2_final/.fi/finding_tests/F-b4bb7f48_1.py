import numpy as np


def test_noiseless_decay_recovery_is_only_run_for_its_planned_cell_and_uses_half_rate_initialisation():
    import simulate
    from kinetics_parameter_recovery import model

    original_fit = model.fit_first_stage
    calls = []

    def recording_fit(concentration, time_s, *args, **kwargs):
        calls.append(
            (
                np.asarray(concentration, dtype=float).copy(),
                np.asarray(time_s, dtype=float).copy(),
                args,
                kwargs,
            )
        )
        return original_fit(concentration, time_s, *args, **kwargs)

    def is_noiseless_decay(concentration, time_s):
        rate = 3.0 / time_s[-1]
        expected = simulate.C0 * np.exp(-rate * time_s)
        return np.allclose(concentration, expected, rtol=1e-13, atol=1e-15)

    def has_half_rate_initialisation(args, kwargs, rate):
        values = list(args) + list(kwargs.values())

        def contains_half(value):
            array = np.asarray(value)
            return bool(
                np.any(
                    np.isclose(
                        array.astype(float, copy=False),
                        rate / 2.0,
                        rtol=8 * np.finfo(float).eps,
                        atol=0.0,
                    )
                )
            )

        return any(contains_half(value) for value in values)

    model.fit_first_stage = recording_fit
    try:
        try:
            simulate.run_trial(
                {"noise_sigma_over_c0": 0.05, "n_time": 9}, trial_id=0, seed=123
            )
        except Exception as exc:
            assert False, f"run_trial raised instead of returning its check result: {exc}"

        assert not any(
            is_noiseless_decay(concentration, time_s)
            for concentration, time_s, _, _ in calls
        ), (
            "noiseless_decay_recovery is being fitted in a noisy cell; the plan "
            "places this check only at noise=0 and n_time=9"
        )

        calls.clear()
        try:
            simulate.run_trial(
                {"noise_sigma_over_c0": 0.0, "n_time": 9}, trial_id=0, seed=123
            )
        except Exception as exc:
            assert False, f"the planned noiseless check raised unexpectedly: {exc}"

        check_calls = [
            (time_s, args, kwargs)
            for concentration, time_s, args, kwargs in calls
            if is_noiseless_decay(concentration, time_s)
        ]
        assert check_calls, "the planned noiseless_decay_recovery fit was not run"
        assert all(
            has_half_rate_initialisation(args, kwargs, 3.0 / time_s[-1])
            for time_s, args, kwargs in check_calls
        ), (
            "noiseless_decay_recovery does not initialise each first-stage fit "
            "at half of that decay's true rate"
        )
    finally:
        model.fit_first_stage = original_fit
