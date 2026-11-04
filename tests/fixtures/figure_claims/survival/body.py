"""Survival: a step curve of the share surviving for two groups; group B carries a 95% band."""


def figure_survival(fig, table):
    times = sorted({table.setting(k)["time"] for k in table.keys()})
    for group in ("a", "b"):
        keys = [table.at(time=t, group=group)[0] for t in times]
        means = [table.mean("survival", k) for k in keys]
        fig.line(0, "group_" + group, times, means, "survival", step=True)
        if group == "b":
            ci = [table.interval("survival", k) for k in keys]
            fig.band(0, "group_b", times, [lo for lo, _hi in ci], [hi for _lo, hi in ci], "survival", "ci95")
