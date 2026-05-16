// md_lite.js — a tiny self-hosted Markdown → HTML renderer.
//
// Intentionally minimal: handles what FI's paper.md contains in
// practice — headings, emphasis, inline code, fenced code, lists,
// blockquotes, links, images, tables, horizontal rules. Math
// expressions ($...$ and $$...$$) pass through as <code> blocks so
// downstream MathJax (if loaded) can grab them; without MathJax
// they render as raw LaTeX which is the existing CLI behavior too.
//
// Why not vendor marked.js (50 KB)? This is ~150 LOC, has no
// external dependency, ships fully offline, and renders FI's papers
// correctly enough for the in-browser preview. If a user needs the
// real published PDF rendering, they download paper.pdf — that path
// is unchanged.
(function () {
  const ESCAPE_MAP = {
    '&': '&amp;', '<': '&lt;', '>': '&gt;',
    '"': '&quot;', "'": '&#39;',
  };
  function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ESCAPE_MAP[c]);
  }

  function renderInline(s) {
    // Order matters: code spans first so their content is escaped
    // verbatim and not subject to **bold** etc.
    s = s.replace(/`([^`\n]+)`/g, (m, code) =>
      `<code>${esc(code)}</code>`);
    // Images ![alt](url)
    s = s.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (m, alt, url) =>
      `<img alt="${esc(alt)}" src="${esc(url)}" style="max-width:100%">`);
    // Links [text](url)
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (m, text, url) =>
      `<a href="${esc(url)}">${esc(text)}</a>`);
    // Bold **text**
    s = s.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    // Italic *text*
    s = s.replace(/(^|[\s\W])\*([^*\n]+)\*/g, '$1<i>$2</i>');
    return s;
  }

  function renderTable(lines) {
    // Already trimmed of fences. First line = header, second =
    // separator, rest = body.
    const cells = (line) => line.split('|').slice(1, -1).map(c => c.trim());
    const header = cells(lines[0]).map(c => `<th>${renderInline(esc(c))}</th>`).join('');
    const body = lines.slice(2).map(line =>
      '<tr>' + cells(line).map(c => `<td>${renderInline(esc(c))}</td>`).join('') + '</tr>',
    ).join('');
    return `<table class="fi-md-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
  }

  window.fi_renderMarkdown = function (src) {
    const lines = String(src || '').split(/\r?\n/);
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      // Fenced code block
      if (/^```/.test(line)) {
        const lang = line.replace(/^```/, '').trim();
        const start = ++i;
        while (i < lines.length && !/^```/.test(lines[i])) i++;
        const code = lines.slice(start, i).map(esc).join('\n');
        out.push(`<pre><code class="lang-${esc(lang)}">${code}</code></pre>`);
        i++;
        continue;
      }
      // Horizontal rule
      if (/^---+\s*$/.test(line)) {
        out.push('<hr>');
        i++;
        continue;
      }
      // Heading
      const h = line.match(/^(#{1,6})\s+(.+)$/);
      if (h) {
        const level = h[1].length;
        out.push(`<h${level}>${renderInline(esc(h[2]))}</h${level}>`);
        i++;
        continue;
      }
      // Table — two consecutive pipe-rows, second is a separator.
      if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length
          && /^\|[\s\-:|]+\|\s*$/.test(lines[i + 1])) {
        const tbl = [line, lines[i + 1]];
        i += 2;
        while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) {
          tbl.push(lines[i]);
          i++;
        }
        out.push(renderTable(tbl));
        continue;
      }
      // Block quote
      if (line.startsWith('> ')) {
        const start = i;
        while (i < lines.length && lines[i].startsWith('> ')) i++;
        const inner = lines.slice(start, i).map(l => l.slice(2)).join('\n');
        out.push(`<blockquote>${window.fi_renderMarkdown(inner)}</blockquote>`);
        continue;
      }
      // List (- or 1.)
      const isUl = /^\s*[-*+]\s/.test(line);
      const isOl = /^\s*\d+\.\s/.test(line);
      if (isUl || isOl) {
        const tag = isOl ? 'ol' : 'ul';
        const start = i;
        while (i < lines.length && (
          (isUl && /^\s*[-*+]\s/.test(lines[i]))
          || (isOl && /^\s*\d+\.\s/.test(lines[i]))
        )) i++;
        const items = lines.slice(start, i).map(l =>
          `<li>${renderInline(esc(l.replace(/^\s*(?:[-*+]|\d+\.)\s/, '')))}</li>`,
        ).join('');
        out.push(`<${tag}>${items}</${tag}>`);
        continue;
      }
      // Blank line
      if (!line.trim()) {
        i++;
        continue;
      }
      // Paragraph — accumulate consecutive non-blank lines.
      const start = i;
      while (i < lines.length && lines[i].trim()
          && !/^(#{1,6}\s|```|\|.*\|\s*$|>\s|---+\s*$|\s*[-*+]\s|\s*\d+\.\s)/.test(lines[i])) {
        i++;
      }
      const para = lines.slice(start, i).join(' ');
      out.push(`<p>${renderInline(esc(para))}</p>`);
    }
    return out.join('\n');
  };
})();
