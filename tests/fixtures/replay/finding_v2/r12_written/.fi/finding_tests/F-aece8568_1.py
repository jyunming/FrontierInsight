import numpy as np
import simulate


def test_run_trial_returns_each_recovered_rate_constant():
    try:
        result = simulate.run_trial(
            {"noise_std_rel": 0.02, "n_timepoints": 6}, trial_id=0, seed=123
        )
    except Exception as exc:
        assert False, f"run_trial raised unexpectedly: {exc}"

    khat = result.get("khat")
    assert khat is not None, (
        "run_trial reports an average relative rate error instead of returning the "
        "individual recovered rate constants (khat) for the five temperatures."
    )
    khat = np.asarray(khat, dtype=float)
    assert khat.shape == (5,) and np.all(np.isfinite(khat)) and np.all(khat > 0), (
        "khat must contain one finite positive recovered rate constant per temperature."
    )
