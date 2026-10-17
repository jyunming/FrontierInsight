import numpy as np


def test_summarize_cell_reports_k_relative_bias_under_the_planned_name():
    try:
        import experiment
    except Exception as exc:
        assert False, f"experiment could not be imported: {exc}"

    metrics = {
        "Ea_hat_J_mol": {"trials": [0, 1], "values": [50000.0, 50000.0]},
        "lnA_hat": {"values": [np.log(1.0e8), np.log(1.0e8)]},
        "Ea_se_J_mol": {"values": [1.0, 1.0]},
        "lnA_se": {"values": [1.0, 1.0]},
        "Ea_covered": {"values": [1, 1]},
        "lnA_covered": {"values": [1, 1]},
        "Ea_lnA_reported_correlation": {"values": [0.0, 0.0]},
    }
    for temperature_index in range(5):
        metrics[f"k_relative_error_{temperature_index}"] = {
            "values": [float(temperature_index), float(temperature_index + 2)]
        }
        metrics[f"k_covered_{temperature_index}"] = {"values": [1, 1]}

    cell_record = {
        "cell": {"additive_noise_sd": 0.1, "n_time_points": 3},
        "metrics": metrics,
        "planned": 2,
        "failed": 0,
    }

    try:
        summary = experiment.summarize_cell(cell_record, np.random.default_rng(1))
    except Exception as exc:
        assert False, f"summarize_cell raised unexpectedly: {exc}"

    assert "k_relative_bias" in summary, (
        "The planned k_relative_bias result is missing; only "
        "k_relative_bias_by_temp is reported."
    )

    reported = summary["k_relative_bias"]
    assert len(reported) == 5, "k_relative_bias must report one mean relative bias per temperature."
    for temperature_index, item in enumerate(reported):
        estimate = item["estimate"] if isinstance(item, dict) else item
        expected = float(temperature_index + 1)
        assert estimate == expected, (
            "k_relative_bias must equal the mean of successful trials' "
            "((k_hat - k_true) / k_true) values at each temperature."
        )
