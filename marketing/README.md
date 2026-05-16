# marketing/

Standalone landing page for Frontier Insight, intended to be
deployed **separately** from the FastAPI `--serve` server (GitHub
Pages, Netlify, Cloudflare Pages, S3 + CloudFront, etc.).

## Contents

- `index.html` — the entire landing page. Self-contained: inline
  CSS, inline SVG logo, no external assets. Single file deploy.

## Deploy

### GitHub Pages
1. Settings → Pages → Source: deploy from a branch.
2. Pick a branch (`main` is fine) and the `/marketing` folder.
3. The page goes live at `https://<owner>.github.io/<repo>/`.

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

Edit `index.html` directly. Keep it dependency-free: no CDN links,
no external fonts, no JavaScript. One file, deployable anywhere.
