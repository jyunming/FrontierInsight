// Frontier Insight — shared top nav, ported from Stitch's UI Framework
// markup verbatim. Every web app page embeds <div id="fi-header-root">
// and loads this script with `defer`; we replace that node with:
//
//   <nav class="docked full-width top-0 sticky bg-surface-low/60 backdrop-blur-[20px] border-b border-white/10 z-50">
//     [FI gradient wordmark]    [Tools dropdown] [Compare]    [+ New Quest gradient CTA] [trash/settings/github icons]
//
// Tools dropdown items are fetched from /api/tools/schema so the menu
// stays in sync with web/tools_routes.py without re-touching this file.

(function () {
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
    root.outerHTML = `
      <nav class="full-width top-0 sticky bg-surface-low/60 backdrop-blur-[20px] border-b border-white/10 z-50">
        <div class="flex justify-between items-center w-full px-margin-mobile md:px-margin-desktop h-16 max-w-max-width mx-auto">
          <div class="flex items-center gap-8">
            <a href="/" class="flex items-center gap-3 group" title="Frontier Insight dashboard">
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
              <span class="font-headline-lg-mobile md:font-headline-lg text-headline-lg-mobile md:text-headline-lg text-gradient">Frontier Insight</span>
            </a>
            <div class="hidden md:flex items-center gap-6">
              <div class="fi-dropdown" id="fi-tools-dropdown">
                <button class="fi-dropdown-toggle text-on-surface-variant font-medium hover:text-primary transition-colors duration-200" type="button" onclick="window._fiToggleTools(event)">
                  <span>Tools</span>
                  <span class="material-symbols-outlined text-[18px]">expand_more</span>
                </button>
                <div class="fi-dropdown-menu" role="menu" id="fi-tools-menu">
                  ${renderToolsItems(TOOLS_FALLBACK)}
                </div>
              </div>
              <a class="text-on-surface-variant font-medium hover:text-primary transition-colors duration-200" href="/compare">Compare</a>
            </div>
          </div>
          <div class="flex items-center gap-4">
            <a href="/interview" class="hidden md:inline-flex items-center gap-1 bg-gradient-to-r from-primary-container to-secondary-container text-on-primary-container font-label-sm text-label-sm px-4 py-2 rounded uppercase tracking-wider hover:opacity-90 transition-opacity">
              <span class="material-symbols-outlined text-[16px]">add</span>
              <span>New Quest</span>
            </a>
            <div class="flex items-center gap-3 text-on-surface-variant">
              <a href="/trash" class="hover:text-primary transition-colors duration-200" title="Trash">
                <span class="material-symbols-outlined">delete</span>
              </a>
              <a href="/settings" class="hover:text-primary transition-colors duration-200" title="Settings">
                <span class="material-symbols-outlined">settings</span>
              </a>
              <a href="https://github.com/jyunming/FrontierInsight" target="_blank" rel="noopener" class="hover:text-primary transition-colors duration-200" title="GitHub">
                <span class="material-symbols-outlined">terminal</span>
              </a>
            </div>
          </div>
        </div>
      </nav>
    `;
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
