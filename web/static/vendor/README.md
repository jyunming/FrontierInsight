# Vendored web assets

The web UI loads nothing from another host, so it looks and works the same on a machine with no internet access.

| File | What | Source | License |
|---|---|---|---|
| `tailwindcss-3.4.17-forms-container-queries.js` | Tailwind CSS play script with the `forms` and `container-queries` plugins | `https://cdn.tailwindcss.com/3.4.17?plugins=forms@0.5.10,container-queries@0.1.1` | MIT |
| `fonts/Fraunces-*`, `Geist-*`, `JetBrainsMono-*`, `IBMPlex*` | latin and latin-ext subsets of the text fonts | Google Fonts | SIL Open Font License 1.1 |
| `fonts/MaterialSymbolsOutlined-*` | Material Symbols Outlined icon font, default instance (every icon) | Google Fonts | Apache License 2.0 |
| `fonts.css` | `@font-face` rules for the files above, and the `.material-symbols-outlined` class | generated | — |

Refresh the fonts with `python scripts/vendor_web_fonts.py web/static/vendor` (needs network). To move to another
Tailwind version, download the URL above with the new version number and update the `<script src>` of every page in
`web/static/*.html`.
