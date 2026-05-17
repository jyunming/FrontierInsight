/* Frontier Insight — shared Tailwind config, lifted verbatim from
 * Stitch's UI Framework generation (all 6 web app screens were generated
 * in one Stitch project, share the same tailwind.config — extracted here).
 *
 * Tailwind v3 CDN creates window.tailwind itself when it boots, so the
 * proven pattern is:
 *
 *   <script src="/static/theme.js"></script>     // sets window._fiConfig
 *   <script src="https://cdn.tailwindcss.com?..."></script>
 *   <script>tailwind.config = window._fiConfig;</script>
 *
 * Loading theme.js first stages the config blob; the inline script
 * after the CDN copies it onto `tailwind.config` so the CDN sees it
 * before scanning the DOM for utility classes.
 *
 * Don't drift from these tokens — every page reuses the same class
 * names from the Stitch markup (e.g. `text-headline-xl`,
 * `bg-surface-mid`, `font-label-sm`, `text-status-success`).
 */

window._fiConfig = {
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Surface (background + tonal layers)
        "background":               "#131313",
        "surface":                  "#131313",
        "surface-dim":              "#131313",
        "surface-bright":           "#3a3939",
        "surface-low":              "#0d1c2d",
        "surface-mid":              "#122131",
        "surface-high":             "#1c2b3c",
        "surface-variant":          "#353534",
        "surface-tint":             "#c3c0ff",
        "surface-container-lowest": "#0e0e0e",
        "surface-container-low":    "#1c1b1b",
        "surface-container":        "#201f1f",
        "surface-container-high":   "#2a2a2a",
        "surface-container-highest":"#353534",

        // On-surface tones
        "on-surface":               "#e5e2e1",
        "on-surface-variant":       "#c7c4d8",
        "on-background":            "#e5e2e1",
        "inverse-surface":          "#e5e2e1",
        "inverse-on-surface":       "#313030",
        "outline":                  "#918fa1",
        "outline-variant":          "#464555",

        // Primary (indigo → blurple)
        "primary":                  "#c3c0ff",
        "on-primary":               "#1d00a5",
        "primary-container":        "#4f46e5",
        "on-primary-container":     "#dad7ff",
        "inverse-primary":          "#4d44e3",
        "primary-fixed":            "#e2dfff",
        "primary-fixed-dim":        "#c3c0ff",
        "on-primary-fixed":         "#0f0069",
        "on-primary-fixed-variant": "#3323cc",

        // Secondary (cyan)
        "secondary":                "#4cd7f6",
        "on-secondary":             "#003640",
        "secondary-container":      "#03b5d4",
        "on-secondary-container":   "#00424e",
        "secondary-fixed":          "#acedff",
        "secondary-fixed-dim":      "#4cd7f6",
        "on-secondary-fixed":       "#001f26",
        "on-secondary-fixed-variant":"#004e5c",

        // Tertiary (lime green)
        "tertiary":                 "#a4d64c",
        "on-tertiary":              "#233600",
        "tertiary-container":       "#496a00",
        "on-tertiary-container":    "#b9ec5f",
        "tertiary-fixed":           "#bff366",
        "tertiary-fixed-dim":       "#a4d64c",
        "on-tertiary-fixed":        "#131f00",
        "on-tertiary-fixed-variant":"#354e00",

        // Error
        "error":                    "#ffb4ab",
        "on-error":                 "#690005",
        "error-container":          "#93000a",
        "on-error-container":       "#ffdad6",

        // Custom semantic
        "status-success":           "#a4d64c",
        "status-warning":           "#E4F222",
        "accent-blurple":           "#DFD1FF",
      },
      borderRadius: {
        DEFAULT: "0.125rem",
        lg:      "0.25rem",
        xl:      "0.5rem",
        full:    "0.75rem",
      },
      spacing: {
        "component-padding-md": "16px",
        "margin-desktop":       "64px",
        "max-width":            "1440px",
        "section-gap":          "80px",
        "margin-mobile":        "20px",
        "component-padding-xs": "4px",
        "gutter":               "24px",
        "component-padding-sm": "8px",
      },
      fontFamily: {
        // Body — Geist (modern grotesque sans)
        "body-lg":            ["Geist", "Inter", "sans-serif"],
        "body-md":            ["Geist", "Inter", "sans-serif"],
        // Display — Fraunces (variable serif, opsz 9..144 + wght 400..900)
        // Editorial gravitas for a research tool; falls back to Inter if loading.
        "headline-xl":        ["Fraunces", "Inter", "serif"],
        "headline-lg":        ["Fraunces", "Inter", "serif"],
        "headline-lg-mobile": ["Fraunces", "Inter", "serif"],
        // Mono — JetBrains Mono for code; labels in same to keep terminal vibe.
        "code-md":            ["JetBrains Mono", "ui-monospace", "monospace"],
        "label-sm":           ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      fontSize: {
        "body-lg":            ["18px", { lineHeight: "1.6", fontWeight: "400" }],
        "body-md":            ["16px", { lineHeight: "1.6", fontWeight: "400" }],
        // Fraunces is heaviest at 900; tighten tracking for display sizes.
        "headline-xl":        ["72px", { lineHeight: "1.0", letterSpacing: "-0.04em", fontWeight: "700" }],
        "headline-lg":        ["44px", { lineHeight: "1.1", letterSpacing: "-0.03em", fontWeight: "700" }],
        "headline-lg-mobile": ["34px", { lineHeight: "1.1", letterSpacing: "-0.02em", fontWeight: "700" }],
        "code-md":            ["14px", { lineHeight: "1.5", fontWeight: "500" }],
        "label-sm":           ["12px", { lineHeight: "1",   letterSpacing: "0.08em", fontWeight: "600" }],
      },
    },
  },
};
