// Frontier Insight — shared header bar.
//
// Every page embeds <div id="fi-header-root">. This script injects
// the brand bar (logo + breadcrumb + nav with Tools dropdown +
// theme toggle) into all of them. Vanilla JS, no framework dep.
(function () {
  // Tools menu — keep in sync with TOOL_SPECS in web/tools_routes.py
  // (the canonical source). Adding a tool here is one entry.
  const TOOLS = [
    { name: 'proposal',  label: 'Proposal'  },
    { name: 'critique',  label: 'Critique'  },
    { name: 'digest',    label: 'Digest'    },
    { name: 'portfolio', label: 'Portfolio' },
    { name: 'summarize', label: 'Summarize' },
    { name: 'analyze',   label: 'Analyze'   },
    { name: 'fleet',     label: 'Fleet'     },
    { name: 'ingest',    label: 'Ingest'    },
  ];

  function renderHeader() {
    const root = document.getElementById('fi-header-root');
    if (!root) return;
    const breadcrumb = root.dataset.breadcrumb || '';
    const toolsItems = TOOLS.map(t =>
      `<a href="/tools/${t.name}" role="menuitem">${escapeHtml(t.label)}</a>`
    ).join('');
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
            <div class="fi-dropdown" id="fi-tools-dropdown">
              <button class="fi-dropdown-toggle" type="button"
                      onclick="window._fiToggleTools(event)">
                Tools ▾
              </button>
              <div class="fi-dropdown-menu" role="menu">
                ${toolsItems}
                <hr/>
                <a href="/compare" role="menuitem">Compare</a>
                <a href="/trash" role="menuitem">Trash</a>
                <a href="/settings" role="menuitem">Settings</a>
              </div>
            </div>
            <button class="fi-theme-toggle" type="button"
                    onclick="window._fiToggleTheme()"
                    title="Toggle light/dark"
                    aria-label="Toggle theme">☾</button>
          </nav>
        </div>
      </header>
    `;
    applyStoredTheme();
  }

  // Theme toggle — overrides the prefers-color-scheme media query
  // when the user has explicitly picked. localStorage key:
  // 'fi_theme' = 'light' | 'dark' | unset (= auto).
  window._fiToggleTheme = function () {
    const current = localStorage.getItem('fi_theme');
    const next = current === 'dark' ? 'light' : (current === 'light' ? '' : 'dark');
    if (next) {
      localStorage.setItem('fi_theme', next);
    } else {
      localStorage.removeItem('fi_theme');
    }
    applyStoredTheme();
  };

  function applyStoredTheme() {
    const theme = localStorage.getItem('fi_theme');
    const root = document.documentElement;
    if (theme === 'light' || theme === 'dark') {
      root.setAttribute('data-theme', theme);
    } else {
      root.removeAttribute('data-theme');
    }
    // Reflect the current effective theme on the toggle button.
    const btn = document.querySelector('.fi-theme-toggle');
    if (btn) {
      const effective = theme || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      btn.textContent = effective === 'dark' ? '☀' : '☾';
    }
  }

  // Tools dropdown toggle. Closes on outside click or Esc.
  window._fiToggleTools = function (ev) {
    ev?.stopPropagation();
    const drop = document.getElementById('fi-tools-dropdown');
    if (!drop) return;
    drop.classList.toggle('open');
  };

  document.addEventListener('click', (ev) => {
    const drop = document.getElementById('fi-tools-dropdown');
    if (!drop) return;
    if (!drop.contains(ev.target)) drop.classList.remove('open');
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      document.getElementById('fi-tools-dropdown')?.classList.remove('open');
    }
  });

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderHeader);
  } else {
    renderHeader();
  }
})();
