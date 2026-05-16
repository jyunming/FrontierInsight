// Frontier Insight — shared header bar.
//
// Both /interview and / and /quest/<id> embed <div id="fi-header-root">.
// This script injects the same brand bar (logo + breadcrumb + nav)
// into all of them so the FI identity is consistent. Vanilla JS;
// no framework dependency.
(function () {
  function renderHeader() {
    const root = document.getElementById('fi-header-root');
    if (!root) return;
    const breadcrumb = root.dataset.breadcrumb || '';
    root.outerHTML = `
      <header class="fi-header">
        <div class="fi-header-inner">
          <a class="fi-logo" href="/" title="Frontier Insight">
            <img src="/static/logo.svg" alt="FI" />
            <span>Frontier Insight</span>
          </a>
          ${breadcrumb ? `<span class="fi-breadcrumb">${escapeHtml(breadcrumb)}</span>` : ''}
          <nav class="fi-nav">
            <a href="/">Dashboard</a>
            <a href="/interview">+ New Quest</a>
          </nav>
        </div>
      </header>
    `;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderHeader);
  } else {
    renderHeader();
  }
})();
