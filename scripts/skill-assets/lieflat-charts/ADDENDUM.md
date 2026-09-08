## FI addendum (not part of the upstream skill)

**Static export for FI's paper/poster/slides pipeline.** FI's outputs
(`paper_pdf`, `poster`, `slides`) are static/print-oriented and cannot embed
a live Chart.js/ECharts canvas or run JavaScript. Pure-SVG templates (Lupi
Editorial, Basics) already are static vector graphics and can be embedded
directly. For anything that renders via Chart.js/ECharts (Glance, the
`big-*` interactive templates, most report templates), rasterize the
finished HTML to a PNG first:

```
node scripts/render_static.mjs <chart.html> <out.png> [css-selector] [width] [height]
```

Requires Playwright's Chromium, installed once, globally (not a project
dependency — matches this skill's own dev-script pattern):

```
npm install -g playwright && npx playwright install chromium
```
