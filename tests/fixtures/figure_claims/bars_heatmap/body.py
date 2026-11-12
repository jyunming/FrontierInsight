"""Bars and a heat map: growth per method for two arms (panel 1, spread as one standard deviation) and a method-by-dose map (panel 2)."""


def figure_bars_heatmap(fig, table):
    methods = ["m1", "m2", "m3"]
    fig.panel(0, dose=2)  # the bars are of dose 2
    for arm in ("treated", "control"):
        keys = [table.at(method=m, arm=arm, dose=2)[0] for m in methods]
        means = [table.mean("growth", k) for k in keys]
        sds = [table.sd("growth", k) for k in keys]
        fig.bars(0, arm, ("method", methods), means, "growth", yerr=sds, yerr_meaning="sd")
    doses = [1, 2, 3]
    z = [[table.mean("growth", table.at(method=m, arm="treated", dose=d)[0]) for m in methods] for d in doses]
    fig.image(1, z, x=("method", methods), y=("dose", doses), what="growth")
