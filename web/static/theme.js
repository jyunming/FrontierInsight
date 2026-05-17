/* Frontier Insight — Obsidian Frontier shared Tailwind config.
 *
 * Tailwind v3 CDN reads `tailwind.config` AFTER it boots, so the
 * recommended pattern is:
 *
 *   <script src="/static/theme.js"></script>     // sets window._fiConfig
 *   <script src="https://cdn.tailwindcss.com?..."></script>
 *   <script>tailwind.config = window._fiConfig;</script>
 *
 * That order: theme.js loads → tailwind script loads (creates
 * window.tailwind) → inline script assigns config from the
 * pre-loaded blob. Centralizing the config object here means every
 * page (dashboard, quest, settings, trash, compare, tools, interview)
 * gets the same tokens without ~120 lines of duplication.
 *
 * Design system: "Obsidian Frontier" — dark-mode-first developer-tool
 * aesthetic, deep obsidian background (#0a0a0a), indigo→cyan gradient
 * accents (#4f46e5 → #06b6d4), electric-lime tertiary (#E4F222).
 */

window._fiConfig = {
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
