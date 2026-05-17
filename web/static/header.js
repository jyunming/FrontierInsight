// Frontier Insight — shared glass-blur top nav (Obsidian Frontier theme).
//
// Every page embeds <div id="fi-header-root" data-breadcrumb="..."> and
// loads this script with `defer`. We replace that node with the sticky
// nav: FI logo wordmark + beta tag + breadcrumb + Tools dropdown +
// Compare/Trash/Settings/GitHub icons.
//
// Tools dropdown items are fetched from /api/tools/schema so the menu
// stays in sync with web/tools_routes.py without re-touching this file.

(function () {
  // Fallback list if /api/tools/schema is unreachable. Matches the
  // names in web/tools_routes.py::TOOL_SPECS — keep them aligned.
  const TOOLS_FALLBACK = [
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
    root.outerHTML = `
      <nav class="fixed top-0 left-0 right-0 z-50 glass-panel border-b border-border-subtle">
        <div class="max-w-[1440px] mx-auto px-6 md:px-10 h-16 flex items-center justify-between">
          <div class="flex items-center gap-5">
            <a href="/" class="flex items-center gap-2 group" title="Frontier Insight dashboard">
              <svg viewBox="0 0 40 40" class="w-8 h-8 rounded-lg" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <defs>
                  <linearGradient id="hdrG" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                    <stop offset="0" stop-color="#4f46e5"/>
                    <stop offset="1" stop-color="#06b6d4"/>
                  </linearGradient>
                </defs>
                <rect width="40" height="40" rx="8" fill="url(#hdrG)"/>
                <path d="M12 10H28M12 20H24M12 30H20" stroke="white" stroke-width="3" stroke-linecap="round"/>
              </svg>
              <span class="font-display font-extrabold text-headline-md tracking-tight group-hover:text-primary transition-colors">Frontier Insight</span>
              <span class="font-mono text-label-mono px-2 py-0.5 rounded bg-surface-container-highest text-on-surface-variant uppercase border border-border-subtle">beta</span>
            </a>
            <div class="hidden md:flex items-center gap-4 text-on-surface-variant">
              <span class="opacity-30 font-mono">/</span>
              <span class="text-primary font-mono text-label-mono uppercase border-b border-primary pb-1">${escapeHtml(breadcrumb || 'Quests')}</span>
              <div class="fi-dropdown" id="fi-tools-dropdown">
                <button class="fi-dropdown-toggle" type="button" onclick="window._fiToggleTools(event)">
                  Tools <span class="material-symbols-outlined text-[14px] -mr-1">expand_more</span>
                </button>
                <div class="fi-dropdown-menu" role="menu" id="fi-tools-menu">
                  ${renderToolsItems(TOOLS_FALLBACK)}
                </div>
              </div>
              <a href="/compare" class="font-mono text-label-mono uppercase hover:text-primary transition-colors">Compare</a>
            </div>
          </div>
          <div class="flex items-center gap-1.5">
            <a href="/interview" class="hidden sm:inline-flex btn-gradient text-white font-semibold text-body-sm px-4 py-2 rounded-lg items-center gap-1.5">
              <span class="material-symbols-outlined text-[18px]">add</span>
              <span class="hidden md:inline">New Quest</span>
            </a>
            <a href="/trash" class="p-2 rounded-lg hover:bg-surface-container-high transition-colors text-on-surface-variant hover:text-primary" title="Trash">
              <span class="material-symbols-outlined text-[20px]">delete</span>
            </a>
            <a href="/settings" class="p-2 rounded-lg hover:bg-surface-container-high transition-colors text-on-surface-variant hover:text-primary" title="Settings">
              <span class="material-symbols-outlined text-[20px]">settings</span>
            </a>
            <a href="https://github.com/jyunming/FrontierInsight" target="_blank" rel="noopener" class="p-2 rounded-lg hover:bg-surface-container-high transition-colors text-on-surface-variant hover:text-primary" title="GitHub">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5a11.5 11.5 0 0 0-3.63 22.41c.58.1.79-.25.79-.56v-2c-3.22.7-3.9-1.39-3.9-1.39-.53-1.35-1.3-1.71-1.3-1.71-1.06-.72.08-.7.08-.7 1.17.08 1.79 1.2 1.79 1.2 1.04 1.79 2.73 1.27 3.4.97.1-.75.41-1.27.74-1.56-2.57-.29-5.27-1.29-5.27-5.74 0-1.27.46-2.31 1.2-3.12-.12-.3-.52-1.49.11-3.1 0 0 .98-.31 3.2 1.19a11.04 11.04 0 0 1 5.83 0c2.22-1.5 3.2-1.19 3.2-1.19.63 1.61.23 2.8.11 3.1.75.81 1.2 1.85 1.2 3.12 0 4.46-2.71 5.45-5.29 5.74.42.36.79 1.06.79 2.15v3.19c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>
            </a>
          </div>
        </div>
      </nav>
    `;
    // Refresh the Tools menu from the live API so it stays in sync
    // with web/tools_routes.py without redeploying the JS bundle.
    refreshToolsMenu();
  }

  function renderToolsItems(tools) {
    return tools.map(t =>
      `<a href="/tools/${encodeURIComponent(t.name)}" role="menuitem">${escapeHtml(t.label || t.name)}</a>`
    ).join('');
  }

  async function refreshToolsMenu() {
    try {
      const res = await fetch('/api/tools/schema');
      if (!res.ok) return;
      const data = await res.json();
      const tools = (data.tools || []).map(t => ({ name: t.name, label: t.label || t.name }));
      const menu = document.getElementById('fi-tools-menu');
      if (menu && tools.length) menu.innerHTML = renderToolsItems(tools);
    } catch (_) { /* keep fallback */ }
  }

  window._fiToggleTools = function (ev) {
    ev?.stopPropagation();
    document.getElementById('fi-tools-dropdown')?.classList.toggle('open');
  };

  document.addEventListener('click', (ev) => {
    const drop = document.getElementById('fi-tools-dropdown');
    if (drop && !drop.contains(ev.target)) drop.classList.remove('open');
  });
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape') {
      document.getElementById('fi-tools-dropdown')?.classList.remove('open');
    }
  });

  function escapeHtml(s) {
    return String(s ?? '')
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', renderHeader);
  } else {
    renderHeader();
  }
})();
