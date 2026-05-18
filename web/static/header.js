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
              <svg viewBox="0 0 512 512" class="w-8 h-8" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="none">
                <rect width="512" height="512" rx="96" fill="#13121b"/>
                <g stroke="url(#hdrG)" stroke-width="40" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M256 100V412" stroke-width="48"/>
                  <path d="M256 160L320 160L350 130L400 180"/>
                  <path d="M256 260H360"/>
                  <path d="M140 180C180 200 220 240 256 260" opacity="0.3" stroke-width="32"/>
                  <path d="M140 340C180 320 220 280 256 260" opacity="0.3" stroke-width="32"/>
                  <circle cx="256" cy="260" r="12" fill="white" stroke="none"/>
                </g>
                <defs>
                  <linearGradient id="hdrG" x1="256" y1="100" x2="256" y2="412" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#4f46e5"/>
                    <stop offset="1" stop-color="#06b6d4"/>
                  </linearGradient>
                </defs>
              </svg>
              <span class="font-headline-lg text-[22px] md:text-[28px] leading-none tracking-tight text-gradient">Frontier Insight</span>
            </a>
            <div class="hidden md:flex items-center gap-6">
              <div class="fi-dropdown" id="fi-tools-dropdown">
                <button class="fi-dropdown-toggle text-on-surface-variant font-medium hover:text-primary transition-colors duration-200"
                        type="button"
                        id="fi-tools-toggle"
                        aria-haspopup="menu"
                        aria-expanded="false"
                        aria-controls="fi-tools-menu"
                        onclick="window._fiToggleTools(event)">
                  <span>Tools</span>
                  <span class="material-symbols-outlined text-[18px]" aria-hidden="true">expand_more</span>
                </button>
                <div class="fi-dropdown-menu" role="menu" id="fi-tools-menu" aria-labelledby="fi-tools-toggle">
                  ${renderToolsItems(TOOLS_FALLBACK)}
                </div>
              </div>
              <a class="text-on-surface-variant font-medium hover:text-primary transition-colors duration-200" href="/jobs">Jobs</a>
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

  // ARIA-aware open/close. ``aria-expanded`` on the toggle button is
  // the source of truth screen readers consult; we keep the ``.open``
  // class on the wrapper in sync so CSS still drives visibility.
  function setToolsOpen(open) {
    const drop = document.getElementById('fi-tools-dropdown');
    const toggle = document.getElementById('fi-tools-toggle');
    if (!drop || !toggle) return;
    drop.classList.toggle('open', open);
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
      // Move keyboard focus into the menu so arrow-key / Enter
      // interaction works without a mouse. First menuitem only —
      // arrow keys are an accessibility nice-to-have that vanilla
      // browser focus order doesn't supply.
      const firstItem = document.querySelector('#fi-tools-menu [role="menuitem"]');
      if (firstItem) firstItem.focus();
    }
  }

  window._fiToggleTools = function (ev) {
    ev?.stopPropagation();
    const drop = document.getElementById('fi-tools-dropdown');
    const willOpen = drop && !drop.classList.contains('open');
    setToolsOpen(Boolean(willOpen));
  };

  document.addEventListener('click', (ev) => {
    const drop = document.getElementById('fi-tools-dropdown');
    if (drop && drop.classList.contains('open') && !drop.contains(ev.target)) {
      setToolsOpen(false);
    }
  });
  document.addEventListener('keydown', (ev) => {
    const drop = document.getElementById('fi-tools-dropdown');
    if (!drop || !drop.classList.contains('open')) return;
    if (ev.key === 'Escape') {
      setToolsOpen(false);
      // Return focus to the toggle so the user knows where they are.
      document.getElementById('fi-tools-toggle')?.focus();
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
