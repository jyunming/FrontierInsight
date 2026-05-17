# marketing/

Standalone landing page for Frontier Insight, intended to be
deployed **separately** from the FastAPI `--serve` server (GitHub
Pages, Netlify, Cloudflare Pages, S3 + CloudFront, etc.).

## Contents

- `index.html` — the entire landing page. Single file. Inline SVG
  logo. Tailwind v3 + Google Fonts (Fraunces / Geist / JetBrains
  Mono) are pulled from public CDNs at view time — no build step,
  but the deploy target does need outbound HTTPS to load assets.

## Deploy

### GitHub Pages (this repo)
`.github/workflows/pages.yml` deploys `marketing/` to GitHub Pages
on every push to `main` that touches `marketing/**`. The page lands at
`https://<owner>.github.io/FrontierInsight/`.

Setup is one-time:
1. Repo → Settings → Pages.
2. **Source: GitHub Actions** (NOT "Deploy from a branch").
3. The workflow then runs automatically on the next push.

### Netlify / Cloudflare Pages / Vercel
Point the publish directory at `marketing/`. No build command needed
— `index.html` is the entire site.

### S3 + CloudFront / nginx / Apache
`cp marketing/index.html <document-root>/`. Done.

## Why this lives separately from `web/static/`

The FastAPI server at `python launch.py --serve` ships a different
UI — the operational dashboard for managing quests (read-only paper
viewer, file browser, cost chart, etc.). The landing page is a
marketing surface for people who haven't installed FI yet; it
shouldn't share routing with the operational UI.

The single source of truth for "what FI does" is `README.md` at the
repo root + `docs/capabilities.md`. This page is a friendlier,
designed presentation of the same information.

## Updating

Edit `index.html` directly. Push to `main` and `pages.yml` auto-deploys.
A unit test (`tests/test_web_static_pages.py::test_marketing_landing_page_is_self_contained`)
forbids `/static/*` and `/api/*` references so the page stays
single-file-deployable; Tailwind + Google Fonts via CDN are
intentional and allowed.
