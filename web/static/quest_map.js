/* The quest map: every step of a quest in a few big blocks, one shared script for the web quest page and the VS Code
 * "Quest map" panel (the extension copies this file and quest_map.css next to its webview).
 *
 *   FIQuestMap.mount(root, {
 *     questId,                         // shown in the command line
 *     command(node) -> string,         // the same restart as a command line
 *     restart(node) -> Promise<string> // starts it; resolves with a sentence to show (rejects with an Error)
 *   }) -> { update(data), setBusy(bool, why), setStatus(text) }
 *
 * `data` is what GET /api/quests/<id>/rerun-steps returns: { blocks, nodes, finished }. The map draws it and decides
 * nothing: which steps can be restarted from, what a restart redoes and the status of each step all come from
 * core/rerun_from.py (node_map). Text is always set with textContent, never as HTML. */
(function () {
  'use strict';

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }

  var STATUS_TEXT = { done: 'Finished', now: 'Where it stopped', todo: 'Not reached yet', off: 'Not on this quest’s path' };

  function mount(root, opts) {
    opts = opts || {};
    var data = { blocks: [], nodes: [], finished: false };
    var sel = '';
    var open = {};
    var busy = false;
    var busyWhy = '';
    var status = '';
    var armed = '';
    var initialised = false;
    var theme = '';
    try { theme = localStorage.getItem('fi-qmap-theme') || ''; } catch (e) { theme = ''; }
    if (theme) root.dataset.theme = theme;
    root.classList.add('fi-qmap');

    var title = el('h1', 'qm-title', 'Quest map');
    var sub = el('p', 'qm-sub', 'Every step of this quest, grouped into a few big blocks. Open a block to see its small ' +
      'steps. Click a step to see what running the quest again from there would keep and what it would redo.');
    var bar = el('div', 'qm-bar');
    var openAll = el('button', 'qm-btn', 'Open all');
    var closeAll = el('button', 'qm-btn', 'Close all');
    var themeBtn = el('button', 'qm-btn', 'Light / dark');
    themeBtn.setAttribute('aria-label', 'Toggle light or dark');
    bar.append(openAll, closeAll, el('span', 'qm-grow'), themeBtn);
    var legend = el('div', 'qm-legend');
    [['done', 'finished'], ['now', 'where it stopped'], ['todo', 'not reached'], ['redo', 'would be redone']].forEach(function (p) {
      var s = el('span');
      var i = el('i');
      i.style.background = 'var(--' + p[0] + ')';
      s.append(i, p[1]);
      legend.appendChild(s);
    });
    legend.appendChild(el('span', '', 'dashed block = not on this quest’s path'));
    var grid = el('div', 'qm-grid');
    var map = el('div', 'qm-map');
    var panel = el('aside', 'qm-panel');
    panel.setAttribute('aria-live', 'polite');
    grid.append(map, panel);
    root.textContent = '';
    root.append(title, sub, bar, legend, grid);

    function nodeByName(name) {
      for (var i = 0; i < data.nodes.length; i++) if (data.nodes[i].node === name) return data.nodes[i];
      return null;
    }
    function blockNodes(b) { return data.nodes.filter(function (n) { return n.block === b.id; }); }
    function redoSet(n) {
      var out = {};
      if (!n || !n.clickable) return out;
      var hit = false;
      data.nodes.forEach(function (x) {
        if (x.step === n.step) hit = true;
        if (hit && (x.status === 'done' || x.status === 'now')) out[x.node] = true;
      });
      return out;
    }
    function blockStat(b) {
      if (b.off) return 'not used';
      var ns = blockNodes(b);
      if (ns.some(function (n) { return n.status === 'now'; })) return 'stopped here';
      var d = ns.filter(function (n) { return n.status === 'done'; }).length;
      return d === ns.length ? 'finished' : d === 0 ? 'not reached' : d + '/' + ns.length;
    }
    function defaultSelection() {
      var now = data.nodes.filter(function (n) { return n.status === 'now'; })[0];
      if (now) return now.node;
      var done = data.nodes.filter(function (n) { return n.status === 'done'; });
      return done.length ? done[done.length - 1].node : (data.nodes[0] ? data.nodes[0].node : '');
    }

    function render() {
      var redo = redoSet(nodeByName(sel));
      map.textContent = '';
      data.blocks.forEach(function (b, bi) {
        var prev = data.blocks[bi - 1];
        map.appendChild(el('div', bi && !b.off && prev && !prev.off ? 'qm-link' : 'qm-gap'));
        var isOpen = !!open[b.id];
        var sec = el('section', 'qm-block' + (b.off ? ' qm-off' : ''));
        sec.dataset.open = String(isOpen);
        var head = el('button', 'qm-bhead');
        head.type = 'button';
        head.dataset.block = b.id;
        head.setAttribute('aria-expanded', String(isOpen));
        var text = el('span');
        text.append(el('div', 'qm-bname', b.name), el('div', 'qm-bdesc', b.desc));
        head.append(el('span', 'qm-chev', '▸'), text, el('span', 'qm-bstat', blockStat(b)));
        head.onclick = function () { open[b.id] = !open[b.id]; render(); };
        var nodes = el('div', 'qm-nodes');
        blockNodes(b).forEach(function (n, i) {
          if (i) nodes.appendChild(el('span', 'qm-arrow', '→')).setAttribute('aria-hidden', 'true');
          var btn = el('button', 'qm-node qm-s-' + n.status + (n.node === sel ? ' qm-sel' : '') +
            (redo[n.node] && n.node !== sel ? ' qm-redo' : ''));
          btn.type = 'button';
          btn.dataset.node = n.node;
          var t = el('span', 'qm-t');
          t.append(el('span', 'qm-dot'), n.title || n.node);
          btn.append(t, el('span', 'qm-k', n.node));
          btn.onclick = function () { sel = n.node; armed = ''; open[n.block] = true; render(); };
          nodes.appendChild(btn);
        });
        blockNodes(b).filter(function (n) { return n.loop; }).forEach(function (n) {
          nodes.appendChild(el('div', 'qm-loop', '↺ ' + n.loop));
        });
        sec.append(head, nodes);
        map.appendChild(sec);
      });
      renderPanel(redo);
    }

    function row(label, value) {
      var r = el('div', 'qm-row');
      r.append(el('b', '', label), el('span', '', value));
      return r;
    }
    function impactBox(n, redo) {
      var box = el('div', 'qm-impact');
      var names = data.nodes.filter(function (x) { return redo[x.node]; });
      if (n.status === 'off') {
        box.textContent = 'This step is on the other path (real data instead of a simulation, or the reverse), which ' +
          'this quest does not take, so it cannot be restarted from here.';
      } else if (!n.step) {
        box.textContent = 'This step is a pause, a repair loop or your own decision, not a place to restart from. ' +
          'Pick a step just before or after it.';
      } else if (n.status === 'todo') {
        box.className = 'qm-impact qm-ok';
        box.textContent = 'Nothing to redo: this step has not run yet, so there is nothing to restart from.';
      } else {
        var last = names[names.length - 1];
        var redone = el('b', '', 'Redone: ');
        box.append(redone, names.length + ' step' + (names.length === 1 ? '' : 's') + ', from “' + n.title +
          '” to “' + (last ? last.title : n.title) + '”. Their outputs move to .fi/previous/, they are not deleted.');
        box.append(el('br'), el('b', '', 'Kept: '), 'everything before it.');
        if (n.redoes) { box.append(el('br'), el('b', '', 'Restarting here: '), n.redoes + '.'); }
      }
      return box;
    }
    function renderPanel(redo) {
      panel.textContent = '';
      var n = nodeByName(sel);
      if (!n) { panel.textContent = 'No steps yet.'; return; }
      var b = data.blocks.filter(function (x) { return x.id === n.block; })[0];
      panel.appendChild(el('h2', '', n.title || n.node));
      panel.appendChild(el('div', 'qm-pkey', n.node + ' · in “' + (b ? b.name : n.block) + '” · ' +
        (STATUS_TEXT[n.status] || '')));
      panel.append(row('Does', n.sentence), row('Reads', n.reads), row('Writes', n.writes));
      if (n.needs_approval) {
        panel.appendChild(row('Careful', 'Restarting at or before Design changes the frozen protocol (the experiment ' +
          'plan the results are judged by), so it has to be approved again with your name. Use the command below; ' +
          'the button cannot do it.'));
      }
      if (n.hint) panel.appendChild(row('Tune', n.hint));
      panel.appendChild(impactBox(n, redo));
      var acts = el('div', 'qm-acts');
      var go = el('button', 'qm-btn qm-primary', armed === n.node ? 'Yes, restart now' : 'Restart from here');
      go.type = 'button';
      var reason = '';
      if (!n.clickable) reason = 'This step cannot be restarted from.';
      else if (n.needs_approval) reason = 'Needs your name: use the command line below.';
      else if (busy) reason = busyWhy || 'Busy.';
      go.disabled = !!reason || !opts.restart;
      if (reason) go.title = reason;
      go.onclick = function () {
        if (armed !== n.node) {
          armed = n.node;
          status = 'This moves what “' + n.title + '” and the later steps made to .fi/previous/ and runs them again. Press the button again to go ahead.';
          renderPanel(redo);
          return;
        }
        armed = '';
        status = 'Starting…';
        renderStatus();
        Promise.resolve().then(function () { return opts.restart(n); }).then(function (msg) {
          status = msg || 'Started.';
          renderStatus();
        }, function (err) {
          status = 'Could not start: ' + (err && err.message ? err.message : String(err));
          renderStatus();
        });
      };
      acts.appendChild(go);
      var cmdText = opts.command ? opts.command(n) : '';
      if (cmdText && n.clickable) {
        var copy = el('button', 'qm-btn', 'Copy command');
        copy.type = 'button';
        copy.onclick = function () {
          var done = function () { status = 'Command copied.'; renderStatus(); };
          try { navigator.clipboard.writeText(cmdText).then(done, function () { status = 'Select the command and copy it.'; renderStatus(); }); }
          catch (e) { status = 'Select the command and copy it.'; renderStatus(); }
        };
        acts.appendChild(copy);
      }
      panel.appendChild(acts);
      if (cmdText && n.clickable) panel.appendChild(el('div', 'qm-cmd', cmdText));
      var st = el('div', 'qm-status', status);
      st.id = 'qm-status';
      panel.appendChild(st);
    }
    function renderStatus() {
      var s = panel.querySelector('#qm-status');
      if (s) s.textContent = status;
    }

    openAll.onclick = function () { data.blocks.forEach(function (b) { open[b.id] = true; }); render(); };
    closeAll.onclick = function () { data.blocks.forEach(function (b) { open[b.id] = false; }); render(); };
    themeBtn.onclick = function () {
      var dark = root.dataset.theme === 'dark' ||
        (!root.dataset.theme && window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches);
      root.dataset.theme = dark ? 'light' : 'dark';
      try { localStorage.setItem('fi-qmap-theme', root.dataset.theme); } catch (e) { /* not kept */ }
    };

    return {
      update: function (next) {
        data = { blocks: next.blocks || [], nodes: next.nodes || [], finished: !!next.finished };
        if (!nodeByName(sel)) sel = defaultSelection();
        if (!initialised) {
          initialised = true;
          var start = nodeByName(sel);
          data.blocks.forEach(function (b) { open[b.id] = false; });
          if (start) open[start.block] = true;
        }
        render();
      },
      setBusy: function (b, why) { busy = !!b; busyWhy = why || ''; render(); },
      setStatus: function (text) { status = text || ''; renderStatus(); },
      setTheme: function (t) { if (t === 'light' || t === 'dark') root.dataset.theme = t; },
    };
  }

  window.FIQuestMap = { mount: mount };
})();
