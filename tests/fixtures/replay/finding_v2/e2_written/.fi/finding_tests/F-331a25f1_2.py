import experiment
import simulate


def test_convergence_check_uses_production_first_stage_fitter():
    assert hasattr(
        simulate, "fit_first_stage"
    ), "The production first-stage fitter must be available for the convergence refit."

    production_fit = simulate.fit_first_stage
    calls = []
    missing = object()
    original_experiment_fit = getattr(experiment, "fit_first_stage", missing)

    def tracked_production_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return production_fit(*args, **kwargs)

    try:
        simulate.fit_first_stage = tracked_production_fit
        experiment.fit_first_stage = tracked_production_fit
        try:
            experiment._convergence_refit_check(
                noise_sigma_over_c0=0.01,
                n_time=6,
                seed=17,
                runs=20,
                n_temperatures=2,
                rate_span=3.0,
            )
        except Exception as exc:
            assert False, f"The convergence refit check raised {type(exc).__name__}: {exc}"

        assert calls, (
            "The convergence check must refit its generated series with the production "
            "fit_first_stage routine; it instead performs an independent curve_fit."
        )
    finally:
        simulate.fit_first_stage = production_fit
        if original_experiment_fit is missing:
            delattr(experiment, "fit_first_stage")
        else:
            experiment.fit_first_stage = original_experiment_fit
