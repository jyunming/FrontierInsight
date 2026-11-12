"""Map: a load on a column-by-row grid (panel 1) and two rows of it as lines (panel 2)."""


def figure_map(fig, table):
    cols = sorted({table.setting(k)["col"] for k in table.keys()})
    rows = sorted({table.setting(k)["row"] for k in table.keys()})
    z = [[table.mean("load", table.at(col=c, row=r)[0]) for c in cols] for r in rows]
    fig.image(0, z, x=("col", cols), y=("row", rows), what="load")
    for r in (0, 2):
        fig.line(1, r, cols, [table.mean("load", table.at(col=c, row=r)[0]) for c in cols], "load")
