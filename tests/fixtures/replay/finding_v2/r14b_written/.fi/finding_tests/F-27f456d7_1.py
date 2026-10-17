import experiment


class _Axis:
    def __init__(self):
        self.plots = []

    def plot(self, *args, **kwargs):
        self.plots.append(tuple(args[1]))
        return []

    def axhline(self, *args, **kwargs):
        return None

    def set(self, **kwargs):
        return None

    def legend(self, **kwargs):
        return None


class _Figure:
    def __init__(self, axes, saved):
        self.axes = axes
        self._saved = saved

    def savefig(self, path, **kwargs):
        self._saved[path] = self


class _PlotRecorder:
    def __init__(self):
        self.saved = {}

    def subplots(self, rows, columns, **kwargs):
        axes = [_Axis() for _ in range(columns)]
        return _Figure(axes, self.saved), axes

    def close(self, figure):
        return None


def test_error_and_bias_figure_includes_both_parameters_and_all_timepoint_designs():
    recorder = _PlotRecorder()
    original_plt = experiment.plt
    time_point_counts = (5, 7, 9, 13, 17)
    noise_levels = (0.01, 0.02)

    cells = []
    for n_time_points in time_point_counts:
        for noise_index, noise in enumerate(noise_levels):
            methods = {}
            for method_index, method in enumerate(
                ("weighted_centered", "unweighted_uncentered")
            ):
                offset = 1000 * method_index + 100 * n_time_points + noise_index
                methods[method] = {
                    "Ea": {
                        "coverage_t": 0.95,
                        "bias": offset + 1,
                        "sd": offset + 2,
                        "rmse": offset + 3,
                    },
                    "ln_A": {
                        "coverage_t": 0.95,
                        "bias": offset + 11,
                        "sd": offset + 12,
                        "rmse": offset + 13,
                    },
                    "k_hat_by_temperature": [1, 1, 1, 1, 1],
                    "Ea_ln_A_correlation": 0.0,
                }
            cells.append(
                {
                    "noise_sd_fraction_c0": noise,
                    "n_time_points": n_time_points,
                    "by_arrhenius_method": methods,
                }
            )

    result = {
        "cells": cells,
        "study": {"true_values": {"k_s_inv": [1, 1, 1, 1, 1]}},
        "theory": {"Ea_ln_A_correlation": 0.0},
    }

    experiment.plt = recorder
    try:
        try:
            experiment._figures(result)
        except Exception as exc:
            assert False, f"_figures raised unexpectedly: {exc}"
    finally:
        experiment.plt = original_plt

    figure = recorder.saved["figures/error_and_bias_vs_noise.png"]
    plotted_series = {series for axis in figure.axes for series in axis.plots}

    for n_time_points in time_point_counts:
        ea_bias = tuple(100 * n_time_points + i + 1 for i in range(len(noise_levels)))
        ln_a_bias = tuple(100 * n_time_points + i + 11 for i in range(len(noise_levels)))
        ea_spread = tuple(100 * n_time_points + i + 2 for i in range(len(noise_levels)))
        ln_a_spread = tuple(100 * n_time_points + i + 12 for i in range(len(noise_levels)))

        assert ea_bias in plotted_series, (
            "error_and_bias_vs_noise.png omits Ea bias for a requested "
            f"{n_time_points}-point design."
        )
        assert ln_a_bias in plotted_series, (
            "error_and_bias_vs_noise.png must plot ln A bias as well as Ea bias."
        )
        assert ea_spread in plotted_series, (
            "error_and_bias_vs_noise.png must plot an Ea spread/error measure, "
            "not bias alone."
        )
        assert ln_a_spread in plotted_series, (
            "error_and_bias_vs_noise.png must plot an ln A spread/error measure, "
            "not bias alone."
        )
