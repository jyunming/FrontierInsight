/* Frontier Insight — Obsidian Frontier shared Tailwind config.
 *
 * This file MUST load BEFORE the Tailwind CDN script. Tailwind CDN
 * reads `window.tailwind.config` synchronously when it boots. If you
 * `defer` this or load it after the CDN, your custom tokens disappear.
 *
 * Why a shared file at all: every page (dashboard, quest detail,
 * settings, trash, compare, interview, tools) needs the exact same
 * color + typography + spacing scale. Duplicating ~120 lines of
 * `tailwind.config = {...}` per page drifts and bloats. Loading from
 * /static/theme.js means changing a token rolls out everywhere.
 *
 * Design system: "Obsidian Frontier" — dark mode primary, deep obsidian
 * background (#0a0a0a), indigo→cyan gradient accents (#4f46e5 → #06b6d4),
 * electric-lime tertiary (#E4F222) for live-status signals.
 */

// `window.tailwind` may not exist yet when this file runs (it's
// created by the CDN script). Defining `tailwind` here makes the
// CDN script pick up our config when it initializes.
if (!window.tailwind) { window.tailwind = {}; }

window.tailwind.config = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        "surface-obsidian":         "#0a0a0a",
        "surface-charcoal":         "#171717",
        "surface-dim":              "#121212",
        "surface-container-lowest": "#0e0e0e",
        "surface-container-low":    "#1c1b1b",
        "surface-container":        "#201f1f",
        "surface-container-high":   "#2a2a2a",
        "surface-container-highest":"#353534",
        "on-surface":               "#e5e2e1",
        "on-surface-variant":       "#c7c4d8",
        "outline":                  "#918fa1",
        "outline-variant":          "#464555",
        "border-subtle":            "rgba(255, 255, 255, 0.08)",
        "primary":                  "#c3c0ff",
        "primary-container":        "#4f46e5",
        "secondary":                "#4cd7f6",
        "tertiary":                 "#E4F222",
        "indigo-accent":            "#4f46e5",
        "cyan-accent":              "#06b6d4",
        "electric-lime":            "#E4F222",
        "error":                    "#ffb4ab",
      },
      fontFamily: {
        display: ["Inter", "system-ui", "sans-serif"],
        body:    ["Geist", "Inter", "system-ui", "sans-serif"],
        mono:    ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        "display-lg":        ["56px", { lineHeight: "1.05", letterSpacing: "-0.035em", fontWeight: "900" }],
        "display-lg-mobile": ["36px", { lineHeight: "1.1",  letterSpacing: "-0.03em",  fontWeight: "900" }],
        "headline-lg":       ["32px", { lineHeight: "1.2",  letterSpacing: "-0.02em",  fontWeight: "800" }],
        "headline-md":       ["22px", { lineHeight: "1.3",  letterSpacing: "-0.01em",  fontWeight: "700" }],
        "body-lg":           ["18px", { lineHeight: "1.6",  fontWeight: "400" }],
        "body-md":           ["15px", { lineHeight: "1.6",  fontWeight: "400" }],
        "body-sm":           ["13px", { lineHeight: "1.5",  fontWeight: "400" }],
        "label-mono":        ["11px", { lineHeight: "1",    letterSpacing: "0.12em",   fontWeight: "600" }],
        "code-sm":           ["13px", { lineHeight: "1.5",  fontWeight: "400" }],
      },
    },
  },
};
