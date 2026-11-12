"""Spectra: the signal of two samples across wavelength, with 95% intervals, a band for sample B and the plan's baseline."""


def figure_spectra(fig, table):
    waves = sorted({table.setting(k)["wavelength"] for k in table.keys()})
    for sample in ("a", "b"):
        keys = [table.at(wavelength=w, sample=sample)[0] for w in waves]
        means = [table.mean("signal", k) for k in keys]
        halves = [(table.interval("signal", k)[1] - table.interval("signal", k)[0]) / 2 for k in keys]
        fig.line(0, sample, waves, means, "signal", yerr=halves, yerr_meaning="ci95")
        if sample == "b":
            fig.band(0, "b", waves, [m - h for m, h in zip(means, halves)], [m + h for m, h in zip(means, halves)],
                     "signal", "ci95")
    fig.ref(0, "baseline")
    fig.note(0, 480, 7.0, "mean signal of sample A at the peak: {signal}", key="wavelength=500,sample=a")
