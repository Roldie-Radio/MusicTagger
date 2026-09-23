/* =========================================================================
   MusicTagger UI
   Vanilla JS, no build step. Long operations run as server-side jobs which
   this file polls, so the window never blocks on a big library.
   ========================================================================= */

'use strict';

const PAGE_SIZE = 100;

/* Every column the grid can show. `tag: true` means it maps to a writable tag
   field, which is what makes a cell sortable, diffable and editable. */
const COLUMNS = [
  { key: 'title', label: 'Title', width: 'minmax(130px, 1.5fr)', tag: true },
  { key: 'artist', label: 'Artist', width: 'minmax(100px, 1fr)', tag: true },
  { key: 'album', label: 'Album', width: 'minmax(100px, 1fr)', tag: true },
  {
    key: 'album_artist', label: 'Album artist', width: 'minmax(100px, 1fr)', tag: true,
    hint: 'Plex groups albums by this. When it is missing or inconsistent, '
        + 'one album shows up as several.',
  },
  { key: 'track_no', label: '#', width: '44px', tag: true, align: 'right', numeric: true },
  { key: 'track_total', label: 'of', pickerLabel: 'Track total', width: '42px',
    tag: true, align: 'right', numeric: true },
  { key: 'disc_no', label: 'Disc', width: '48px', tag: true, align: 'right', numeric: true },
  { key: 'disc_total', label: 'of', pickerLabel: 'Disc total', width: '42px',
    tag: true, align: 'right', numeric: true },
  { key: 'year', label: 'Year', width: '56px', tag: true, align: 'right', numeric: true },
  { key: 'date', label: 'Date', width: '92px', tag: true },
  { key: 'genre', label: 'Genre', width: 'minmax(80px, .7fr)', tag: true },
  { key: 'composer', label: 'Composer', width: 'minmax(90px, .7fr)', tag: true },
  { key: 'isrc', label: 'ISRC', width: '108px', tag: true },
  { key: 'filename', label: 'File', width: 'minmax(120px, 1fr)' },
  { key: 'folder', label: 'Folder', width: 'minmax(110px, 1fr)' },
  { key: 'format', label: 'Format', width: '64px' },
  { key: 'bitrate', label: 'Bitrate', width: '74px', align: 'right' },
  { key: 'duration', label: 'Length', width: '60px', align: 'right' },
  { key: 'quality', label: 'Quality', width: '104px' },
];

const DEFAULT_COLUMNS = ['title', 'artist', 'album', 'album_artist',
  'track_no', 'disc_no', 'year', 'genre', 'format', 'quality'];

const column = (key) => COLUMNS.find((c) => c.key === key);

function loadStored(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (_) { return fallback; }
}

function store(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* private mode */ }
}

const state = {
  tracks: [],
  total: 0,
  offset: 0,
  selected: new Set(),
  openPath: null,
  filters: { q: '', bucket: '', issues: '', sort: 'path', desc: false },
  view: loadStored('musictagger-view', 'review'),
  columns: loadStored('musictagger-columns', DEFAULT_COLUMNS),
  valuesMode: loadStored('musictagger-values', 'proposed'),
  config: null,
  status: null,
  update: null,
  //: The desktop shell's updater state, when running inside it.
  desktopUpdate: null,
  //: jobId -> poll timer. One per job: Identify and Quality may run side by
  //: side, and a single shared timer stopped following the first when the
  //: second started - its result was never shown and Cancel lost track of it.
  polls: new Map(),
  activeJobId: null,
  picker: { path: '', chosen: '', target: 'library' },
  exportPlan: null,
  exportResolutions: {},
};

// Guard against a stored list from an older version naming columns that no
// longer exist, which would otherwise render an empty grid.
state.columns = state.columns.filter(column);
if (!state.columns.length) state.columns = [...DEFAULT_COLUMNS];

/* ---------------------------------------------------------------- utils */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status})`);
  return data;
}

function toast(message, kind = '') {
  const node = el('div', `toast ${kind ? 'is-' + kind : ''}`, message);
  $('#toasts').appendChild(node);
  setTimeout(() => {
    node.style.opacity = '0';
    node.style.transition = 'opacity .3s';
    setTimeout(() => node.remove(), 320);
  }, kind === 'error' ? 7000 : 4000);
}

function bucketLabel(bucket) {
  return {
    high: 'Confident',
    review: 'Review',
    low: 'Uncertain',
    unidentified: 'Not identified',
  }[bucket] || '';
}

function confClass(value) {
  if (value == null) return 'c-none';
  const t = state.status?.stats?.thresholds || { auto_apply: 92, review: 70 };
  if (value >= t.auto_apply) return 'c-high';
  if (value >= t.review) return 'c-review';
  return 'c-low';
}

function fmtDuration(seconds) {
  if (!seconds) return '';
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function trackDesc(tags) {
  const bits = [tags.artist, tags.album].filter(Boolean);
  return bits.join(' — ');
}

/* ------------------------------------------------------------- rendering */

function renderStats() {
  const wrap = $('#stats');
  const stats = state.status?.stats;
  wrap.innerHTML = '';
  if (!stats || !stats.total) return;

  const cards = [
    { label: 'Tracks', value: stats.total, cls: '' },
    { label: 'Confident', value: stats.buckets.high, cls: 'is-high' },
    { label: 'Needs review', value: stats.buckets.review, cls: 'is-review' },
    { label: 'Uncertain', value: stats.buckets.low, cls: 'is-low' },
    { label: 'Not identified', value: stats.buckets.unidentified, cls: '' },
  ];
  if (stats.analysed) {
    cards.push({ label: 'Avg quality', value: stats.average_quality ?? '—', cls: '' });
    const flagged = stats.issues.high + stats.issues.medium;
    cards.push({ label: 'Quality flags', value: flagged, cls: flagged ? 'is-review' : '' });
  }
  if (stats.applied) cards.push({ label: 'Applied', value: stats.applied, cls: '' });

  for (const card of cards) {
    const node = el('div', `stat ${card.cls}`);
    node.appendChild(el('div', 'stat-value', String(card.value)));
    node.appendChild(el('div', 'stat-label', card.label));
    wrap.appendChild(node);
  }
}

function renderCapabilityNotice() {
  const box = $('#capabilityNotice');
  const caps = state.status?.capabilities;
  if (!caps) { box.hidden = true; return; }

  const missing = [];
  if (!caps.fingerprinting) {
    missing.push('Acoustic fingerprinting is off, so files with poor tags or '
      + 'meaningless names will match with low confidence.');
  }
  if (!caps.quality_analysis) {
    missing.push('ffmpeg was not found, so audio quality analysis is unavailable.');
  }
  if (!missing.length) { box.hidden = true; return; }

  box.innerHTML = '';
  box.appendChild(document.createTextNode(missing.join(' ')));
  box.appendChild(document.createTextNode(' '));
  const link = el('button', 'link', 'Open settings');
  link.addEventListener('click', () => openSettings());
  box.appendChild(link);
  box.hidden = false;
}

function renderUpdateNotice() {
  const box = $('#updateNotice');
  const info = state.update;
  const desk = state.desktopUpdate;
  // In the desktop app the shell knows more than the backend check: whether
  // the new version is downloading, or already downloaded and waiting.
  if (desk && desk.status === 'ready') {
    box.innerHTML = '';
    box.appendChild(document.createTextNode(
      `MusicTagger ${desk.available} is ready to install. `));
    const restart = el('button', 'link', 'Restart now');
    restart.addEventListener('click', installDesktopUpdate);
    box.appendChild(restart);
    box.hidden = false;
    return;
  }
  if (desk && desk.status === 'downloading') {
    box.innerHTML = '';
    box.textContent = `Downloading MusicTagger ${desk.available || 'update'}\u2026 ${desk.percent || 0}%`;
    box.hidden = false;
    return;
  }
  // Only ever speak up for an update that actually exists. Being up to date,
  // an unreachable GitHub and a switched-off check are all silence: a banner
  // that appears when there is nothing to do is a banner people learn to
  // ignore, including on the release where it matters.
  if (!info || !info.available || !info.latest) { box.hidden = true; return; }
  if (loadStored('musictagger-dismissed-update', '') === info.latest) { box.hidden = true; return; }

  box.innerHTML = '';
  // In the installed desktop app the shell downloads the update itself and
  // asks to restart once it is ready, so say that rather than send people off
  // to download an installer they do not need.
  box.appendChild(document.createTextNode(info.auto_install
    ? `MusicTagger ${info.latest} is available and is downloading in the background. `
      + `You will be asked to restart when it is ready. `
    : `MusicTagger ${info.latest} is available. You have ${info.current}. `));

  const link = el('a', '', info.auto_install ? 'What\u2019s new' : 'View the release');
  link.href = info.url;
  link.target = '_blank';
  link.rel = 'noopener noreferrer';
  // The release notes are the one bit of context worth having without leaving
  // the app, but they are arbitrary markdown - a hover preview, not layout.
  if (info.notes) link.title = info.notes.slice(0, 400);
  box.appendChild(link);

  box.appendChild(document.createTextNode(' · '));
  const dismiss = el('button', 'link', 'Dismiss');
  dismiss.addEventListener('click', () => {
    // Per version, so dismissing this one does not also hide the next one.
    store('musictagger-dismissed-update', info.latest);
    box.hidden = true;
  });
  box.appendChild(dismiss);
  box.hidden = false;
}

async function checkForUpdate({ force = false } = {}) {
  try {
    state.update = await api(`/api/update${force ? '?force=true' : ''}`);
  } catch (_) {
    // A check that cannot run is not worth a toast - the app is entirely
    // usable without it, and the user did not ask for this request.
    state.update = null;
  }
  renderUpdateNotice();
  return state.update;
}

function updateStatusText(info) {
  if (!info) return 'Could not reach GitHub just now.';
  if (!info.enabled) return 'Update checking is switched off.';
  if (info.error) return 'Could not reach GitHub just now.';
  if (!info.latest) return 'No releases have been published yet.';
  if (info.available) {
    return info.auto_install
      ? `${info.latest} is available and will be installed automatically - you have ${info.current}.`
      : `${info.latest} is available - you have ${info.current}.`;
  }
  return `Up to date (${info.current}).`;
}

/* ----------------------------------------------------- desktop updates */

function desktopUpdates() {
  return window.musictaggerDesktop?.updates || null;
}

function renderDesktopUpdate() {
  const desk = state.desktopUpdate;
  const usable = desk && desk.status !== 'unsupported';
  // Inside the installed app "Update now" replaces the plain "Check now",
  // because the shell can actually fetch and install what it finds.
  $('#btnUpdateNow').hidden = !usable;
  $('#btnCheckUpdate').hidden = !!usable;
  if (usable) {
    const btn = $('#btnUpdateNow');
    const hint = $('#updateStatusHint');
    btn.disabled = desk.status === 'checking' || desk.status === 'downloading';
    btn.textContent = {
      checking: 'Checking\u2026',
      downloading: `Downloading ${desk.percent || 0}%`,
      ready: 'Restart and install',
    }[desk.status] || 'Update now';
    hint.textContent = {
      'up-to-date': `Up to date (${desk.current}).`,
      downloading: `Downloading MusicTagger ${desk.available || ''}\u2026`,
      ready: `MusicTagger ${desk.available} is downloaded and ready.`,
      error: `Could not update: ${desk.error || 'unknown error'}`,
    }[desk.status] || '';
  }
  renderUpdateNotice();
}

async function installDesktopUpdate() {
  const updates = desktopUpdates();
  if (!updates) return;
  const result = await updates.install();
  if (result && !result.ok) toast(result.error || 'Could not install the update.', 'error');
}

async function initDesktopUpdates() {
  const updates = desktopUpdates();
  if (!updates) return;
  updates.onState((desk) => { state.desktopUpdate = desk; renderDesktopUpdate(); });
  try {
    state.desktopUpdate = await updates.getState();
  } catch (_) {
    state.desktopUpdate = null;
  }
  renderDesktopUpdate();
}

function ring(value) {
  const cls = confClass(value);
  const node = el('div', `ring ${cls}`);
  node.style.setProperty('--pct', value == null ? 0 : Math.max(0, Math.min(100, value)));
  node.appendChild(el('span', 'ring-text', value == null ? '–' : String(Math.round(value))));
  return node;
}

/* --------------------------------------------------- quality vocabulary */

/* The server owns the scale so the badge, the help text and the score can
   never disagree. These are only the fallbacks for a failed /api/status. */
const FALLBACK_SCALE = {
  max_score: 100,
  severities: [
    { key: 'high', label: 'Serious', penalty: 30 },
    { key: 'medium', label: 'Moderate', penalty: 15 },
    { key: 'low', label: 'Minor', penalty: 6 },
    { key: 'info', label: 'Note', penalty: 0 },
  ],
};

function qualityScale() {
  return state.status?.quality_scale || FALLBACK_SCALE;
}

function severityInfo(key) {
  return qualityScale().severities.find((s) => s.key === key)
    || { key, label: key, penalty: 0, meaning: '' };
}

/** The worst severity present, which is what the badge word reports. */
function worstSeverity(issues) {
  for (const key of ['high', 'medium', 'low']) {
    if (issues.some((i) => i.severity === key)) return key;
  }
  return null;
}

/** One sentence telling the user what to actually do about this file. */
function qualityVerdict(quality) {
  const worst = worstSeverity(quality.issues);
  if (!worst) {
    return quality.issues.length
      ? 'Nothing wrong with this file. The notes below are observations, not faults.'
      : 'Nothing wrong found with this file.';
  }
  return {
    high: 'Something is likely wrong with this file. Worth replacing if you can '
        + 'find a better copy.',
    medium: 'This file is noticeably below par. Worth listening to before you '
          + 'decide whether it bothers you.',
    low: 'Only small imperfections. Most people would never hear these.',
  }[worst];
}

function qualityBadge(quality) {
  if (!quality || !quality.analysed) {
    if (quality && quality.error) {
      const node = el('span', 'qbadge q-high', 'error');
      node.title = `This file could not be analysed:\n${quality.error}`;
      return node;
    }
    const node = el('span', 'q-none', 'not analysed');
    node.title = 'Run "Check Quality" to check this file for problems.';
    return node;
  }

  const worst = worstSeverity(quality.issues);
  const cls = { high: 'q-high', medium: 'q-medium', low: 'q-low' }[worst] || 'q-ok';
  const label = worst ? severityInfo(worst).label : 'Clean';
  const max = qualityScale().max_score;

  const node = el('span', `qbadge ${cls}`);
  node.appendChild(el('span', null, String(quality.score)));
  node.appendChild(el('span', 'qbadge-sep', '·'));
  node.appendChild(el('span', null, label));

  // Spell out both halves of the badge, because "94 Minor" explains nothing
  // on its own: one number is the total, the other word is the worst fault.
  const lines = [
    `Score ${quality.score} of ${max} — worst finding: ${label}`,
    '',
    `Every file starts at ${max}; each finding deducts points.`,
  ];
  if (quality.issues.length) {
    lines.push('');
    for (const issue of quality.issues) {
      const info = severityInfo(issue.severity);
      const cost = info.penalty ? ` −${info.penalty}` : '';
      lines.push(`• ${info.label}${cost}: ${issue.title}`);
    }
  } else {
    lines.push('', 'No problems found.');
  }
  lines.push('', 'Open the row for details, or click the ? in this column header.');
  node.title = lines.join('\n');
  node.setAttribute('aria-label',
    `Quality score ${quality.score} of ${max}, worst finding ${label}`);
  return node;
}

/* ------------------------------------------------------------ the grid */

function folderOf(path) {
  const parts = path.split(/[\\/]/);
  return parts.length > 1 ? parts[parts.length - 2] : '';
}

/** The current and proposed value of one column for one track. */
function cellValues(track, col) {
  if (col.tag) {
    const current = track.current[col.key];
    const proposed = track.match && track.match.proposed
      ? track.match.proposed[col.key] : null;
    return { current: current ?? null, proposed: proposed ?? null };
  }
  const p = track.props;
  const derived = {
    filename: track.filename,
    folder: folderOf(track.path),
    format: (p.container || '').toUpperCase(),
    bitrate: p.bitrate_kbps ? `${p.bitrate_kbps}` : null,
    duration: fmtDuration(p.duration_s),
  }[col.key];
  return { current: derived ?? null, proposed: null };
}

function gridTemplate() {
  const widths = state.columns.map((key) => column(key).width).join(' ');
  return `36px 66px ${widths} 34px`;
}

function sortableKeyFor(col) {
  // Every column maps onto something the server knows how to sort by.
  return col.key;
}

function renderTableHead() {
  const head = $('#tableHead');
  head.innerHTML = '';
  head.style.gridTemplateColumns = state.view === 'grid' ? gridTemplate() : '';
  head.classList.toggle('is-grid', state.view === 'grid');

  const check = el('label', 'cell-check');
  const box = el('input');
  box.type = 'checkbox';
  box.id = 'selectAll';
  box.setAttribute('aria-label', 'Select all visible');
  box.checked = state.tracks.length > 0
    && state.tracks.every((t) => state.selected.has(t.path));
  box.addEventListener('change', () => {
    if (box.checked) state.tracks.forEach((t) => state.selected.add(t.path));
    else state.tracks.forEach((t) => state.selected.delete(t.path));
    renderTracks();
  });
  check.appendChild(box);
  head.appendChild(check);

  const sortHeader = (label, key, extra) => {
    const node = el('div', `th sortable ${extra || ''}`);
    node.appendChild(el('span', null, label));
    if (state.filters.sort === key) {
      node.classList.add('is-sorted');
      node.appendChild(el('span', 'sort-arrow', state.filters.desc ? '▼' : '▲'));
    }
    node.addEventListener('click', () => {
      if (state.filters.sort === key) state.filters.desc = !state.filters.desc;
      else { state.filters.sort = key; state.filters.desc = false; }
      state.offset = 0;
      syncSortSelect();
      refreshTracks();
    });
    return node;
  };

  // The grid's confidence column is narrow; the full word does not fit.
  head.appendChild(sortHeader(
    state.view === 'grid' ? 'Conf.' : 'Confidence', 'confidence', 'cell-conf'));

  if (state.view === 'grid') {
    for (const key of state.columns) {
      const col = column(key);
      const node = sortHeader(col.label, sortableKeyFor(col),
        col.align === 'right' ? 'th-right' : '');
      if (col.hint) node.title = col.hint;
      if (col.key === 'quality') node.appendChild(qualityHelpDot());
      head.appendChild(node);
    }
  } else {
    head.appendChild(sortHeader('Filename', 'filename', 'cell-filename'));
    head.appendChild(sortHeader('Current tags', 'title', 'cell-track'));
    head.appendChild(el('div', 'cell-arrow'));
    head.appendChild(sortHeader('Proposed', 'title', 'cell-proposed'));
    const quality = sortHeader('Quality', 'quality', 'cell-quality');
    quality.appendChild(qualityHelpDot());
    head.appendChild(quality);
  }

  head.appendChild(el('div', 'cell-expand'));
}

function qualityHelpDot() {
  const dot = el('button', 'help-dot', '?');
  dot.title = 'What do these scores mean?';
  dot.setAttribute('aria-label', 'What do these quality scores mean?');
  dot.addEventListener('click', (e) => { e.stopPropagation(); openQualityHelp(); });
  return dot;
}

function renderGridRow(track) {
  const row = el('div', 'track-row is-grid');
  row.dataset.path = track.path;
  row.style.gridTemplateColumns = gridTemplate();
  if (state.selected.has(track.path)) row.classList.add('is-selected');
  if (state.openPath === track.path) row.classList.add('is-open');

  row.appendChild(selectCell(track, row));

  const conf = el('div', 'cell-conf');
  conf.appendChild(ring(track.match ? track.match.confidence : null));
  row.appendChild(conf);

  for (const key of state.columns) {
    const col = column(key);
    row.appendChild(col.key === 'quality'
      ? gridQualityCell(track)
      : gridCell(track, col));
  }

  row.appendChild(expandCell(track));
  openOnRowClick(row, track);
  return row;
}

function gridQualityCell(track) {
  const cell = el('div', 'grid-cell');
  cell.appendChild(qualityBadge(track.quality));
  return cell;
}

function gridCell(track, col) {
  const { current, proposed } = cellValues(track, col);
  const cell = el('div', `grid-cell ${col.align === 'right' ? 'is-right' : ''}`);
  cell.dataset.field = col.key;

  const showCurrent = state.valuesMode === 'current' || !col.tag;
  const shown = showCurrent ? current : (proposed ?? current);
  const changed = col.tag && proposed != null && proposed !== ''
    && String(proposed) !== String(current ?? '');

  if (shown == null || shown === '') {
    cell.appendChild(el('span', 'cell-blank', '—'));
    cell.title = col.tag ? 'Empty. Double-click to fill it in.' : 'Not available';
  } else {
    const text = el('span', 'cell-text', String(shown));
    if (changed && !showCurrent) text.classList.add('changed');
    cell.appendChild(text);
    cell.title = changed && !showCurrent
      ? `Proposed: ${proposed}\nCurrently: ${current || '(empty)'}`
      : String(shown);
  }

  if (col.tag) {
    cell.classList.add('is-editable');
    cell.addEventListener('dblclick', () => editCell(cell, track, col));
  }
  return cell;
}

/* Double-click a tag cell to correct it in place - the fastest path when you
   can see the mistake in the column. */
function editCell(cell, track, col) {
  if (cell.querySelector('input')) return;
  const { current, proposed } = cellValues(track, col);
  const startValue = (state.valuesMode === 'current' ? current : (proposed ?? current)) ?? '';

  const input = el('input', 'cell-input');
  input.type = col.numeric ? 'number' : 'text';
  input.value = startValue;
  cell.innerHTML = '';
  cell.appendChild(input);
  input.focus();
  input.select();

  let finished = false;
  const cancel = () => { if (!finished) { finished = true; renderTracks(); } };
  const commit = async () => {
    if (finished) return;
    finished = true;
    const raw = input.value.trim();
    if (String(raw) === String(startValue ?? '')) { renderTracks(); return; }
    // Number("3/12") is NaN, which JSON sends as null - silently clearing
    // the field instead of saving what was typed.
    if (col.numeric && raw !== '' && !Number.isFinite(Number(raw))) {
      toast(`"${raw}" is not a number.`, 'error');
      renderTracks();
      return;
    }
    const value = raw === '' ? null : (col.numeric ? Number(raw) : raw);
    try {
      await api('/api/track/edit', {
        method: 'POST', body: { path: track.path, tags: { [col.key]: value } },
      });
      await refreshTracks();
    } catch (err) { toast(err.message, 'error'); renderTracks(); }
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    if (e.key === 'Escape') { e.preventDefault(); cancel(); }
  });
  input.addEventListener('blur', commit);
}

function selectCell(track, row) {
  const check = el('label', 'cell-check');
  const box = el('input');
  box.type = 'checkbox';
  box.checked = state.selected.has(track.path);
  box.addEventListener('change', () => {
    if (box.checked) state.selected.add(track.path); else state.selected.delete(track.path);
    row.classList.toggle('is-selected', box.checked);
    updateActionBar();
  });
  check.appendChild(box);
  return check;
}

function toggleDetail(path) {
  state.openPath = state.openPath === path ? null : path;
  renderTracks();
}

function expandCell(track) {
  const cell = el('div', 'cell-expand');
  const expand = el('button', 'expand-btn', '›');
  expand.title = 'Show details';
  expand.addEventListener('click', () => toggleDetail(track.path));
  cell.appendChild(expand);
  return cell;
}

/* A click anywhere on a row opens its details, not just on the small arrow.
   Controls inside the row keep their own clicks, and so does selecting text.
   In the grid, double-click edits a cell; toggling on the first click would
   redraw the row and swallow the second, so the toggle waits a moment and a
   double-click cancels it. */
const ROW_CONTROLS = 'button, a, input, select, textarea, label, .cell-check';

function openOnRowClick(row, track) {
  let timer = null;
  row.addEventListener('click', (e) => {
    if (e.target.closest(ROW_CONTROLS)) return;
    // Cancel first: a double-click also selects the word under it, so the
    // selection check below would otherwise leave the first click's toggle
    // pending and it would close the editor the double-click just opened.
    clearTimeout(timer);
    if (e.detail > 1) return;
    if (window.getSelection && String(window.getSelection())) return;
    timer = setTimeout(() => toggleDetail(track.path), state.view === 'grid' ? 250 : 0);
  });
}

function renderTracks() {
  const list = $('#trackList');
  list.innerHTML = '';
  renderTableHead();
  $('#emptyState').classList.toggle('is-visible', state.tracks.length === 0);

  for (const track of state.tracks) {
    list.appendChild(state.view === 'grid' ? renderGridRow(track) : renderRow(track));
    if (state.openPath === track.path) list.appendChild(renderDetail(track));
  }

  const pages = Math.ceil(state.total / PAGE_SIZE);
  $('#pager').hidden = pages <= 1;
  $('#pagerLabel').textContent =
    `${state.offset + 1}–${Math.min(state.offset + PAGE_SIZE, state.total)} of ${state.total}`;
  $('#btnPrev').disabled = state.offset === 0;
  $('#btnNext').disabled = state.offset + PAGE_SIZE >= state.total;

  updateActionBar();
}

function renderRow(track) {
  const row = el('div', 'track-row');
  row.dataset.path = track.path;
  if (state.selected.has(track.path)) row.classList.add('is-selected');
  if (state.openPath === track.path) row.classList.add('is-open');

  row.appendChild(selectCell(track, row));

  // confidence
  const conf = el('div', 'cell-conf');
  const confInner = el('div', 'conf');
  confInner.appendChild(ring(track.match ? track.match.confidence : null));
  const label = el('div', 'conf-label');
  label.appendChild(el('div', null, bucketLabel(track.bucket)));
  if (track.status === 'applied') label.appendChild(el('div', 'changed', 'applied'));
  confInner.appendChild(label);
  conf.appendChild(confInner);
  row.appendChild(conf);

  // filename - its own column, deliberately separate from what the *tags*
  // claim (the next column over). Folding the two together, showing the
  // filename only as a fallback when the title tag was blank, made a
  // scrambled-tag file look at a glance like the file the tag lied about
  // rather than the file it actually is.
  const filenameCell = el('div', 'cell-filename');
  const filenameCellLine = el('div', 'filename-line', track.filename);
  filenameCellLine.title = track.path;
  filenameCell.appendChild(filenameCellLine);
  row.appendChild(filenameCell);

  // current
  const current = el('div', 'cell-track');
  // Repeats the filename inline, but only actually visible once the narrow-
  // window layout hides the dedicated column above (see styles.css) - there
  // has to be somewhere to see it once that column is gone.
  const filenameLine = el('div', 'filename-line', track.filename);
  filenameLine.title = track.path;
  current.appendChild(filenameLine);

  const titleText = track.current.title;
  const titleEl = el('div', 'title-line', titleText || 'no title tag');
  if (!titleText) titleEl.classList.add('empty-value');
  current.appendChild(titleEl);

  const sub = trackDesc(track.current);
  current.appendChild(el('div', 'sub-line', sub || '— no artist or album tag —'));
  const meta = [track.props.container?.toUpperCase(),
    track.props.bitrate_kbps ? `${track.props.bitrate_kbps} kbps` : null,
    fmtDuration(track.props.duration_s)].filter(Boolean).join(' · ');
  current.appendChild(el('div', 'path-line', meta));
  row.appendChild(current);

  row.appendChild(el('div', 'cell-arrow', '→'));

  // proposed
  const proposed = el('div', 'cell-proposed');
  if (track.match && track.match.proposed && track.match.proposed.title) {
    const p = track.match.proposed;
    const t = el('div', 'title-line', p.title);
    if (p.title !== track.current.title) t.classList.add('changed');
    proposed.appendChild(t);

    const s = el('div', 'sub-line', trackDesc(p));
    if (trackDesc(p) !== sub) s.classList.add('changed');
    proposed.appendChild(s);

    const extra = [];
    if (p.track_no) extra.push(`track ${p.track_no}${p.track_total ? '/' + p.track_total : ''}`);
    if (p.disc_total > 1) extra.push(`disc ${p.disc_no}/${p.disc_total}`);
    if (p.year) extra.push(String(p.year));
    proposed.appendChild(el('div', 'path-line', extra.join(' · ')));
  } else {
    proposed.appendChild(el('div', 'empty-value', 'not identified'));
  }
  row.appendChild(proposed);

  // quality
  const quality = el('div', 'cell-quality');
  quality.appendChild(qualityBadge(track.quality));
  row.appendChild(quality);

  row.appendChild(expandCell(track));
  openOnRowClick(row, track);
  return row;
}

/* ---------------------------------------------- current / proposed tags */

/* Every field the detail panel shows for a track, in display order. `pair`
   combines a number with its total ("3 of 12"); `fallback` reads a second
   field when the first is blank (no release date, but we do have a year). */
const DETAIL_FIELDS = [
  { key: 'title', label: 'Title' },
  { key: 'artist', label: 'Artist' },
  { key: 'album', label: 'Album' },
  { key: 'album_artist', label: 'Album artist' },
  { key: 'track_no', label: 'Track', pair: 'track_total' },
  { key: 'disc_no', label: 'Disc', pair: 'disc_total' },
  { key: 'date', label: 'Date', fallback: 'year' },
  { key: 'genre', label: 'Genre' },
  { key: 'composer', label: 'Composer' },
  { key: 'isrc', label: 'ISRC' },
  { key: 'compilation', label: 'Compilation', boolLabel: true },
];

/** This field's display value out of a TrackTags-shaped object, or null. */
function fieldValue(tags, field) {
  if (!tags) return null;
  if (field.boolLabel) return tags[field.key] ? 'Yes' : null;
  if (field.pair) {
    const no = tags[field.key];
    const total = tags[field.pair];
    if (no == null && total == null) return null;
    return total ? `${no ?? '?'} of ${total}` : String(no);
  }
  let value = tags[field.key];
  if ((value == null || value === '') && field.fallback) value = tags[field.fallback];
  return (value == null || value === '') ? null : value;
}

/** Everything the file already has, regardless of whether it has been
   identified. Shown for every track, so "what's on this file right now"
   is always one click away. */
function renderCurrentMetadata(track) {
  const box = el('div');
  const head = el('div', 'detail-section-head');
  head.appendChild(el('h4', null, 'Current metadata'));
  if (track.current.has_art) {
    const img = el('img', 'art-thumb');
    img.src = `/api/art?path=${encodeURIComponent(track.path)}`;
    img.alt = 'Embedded cover art';
    img.title = 'Embedded cover art';
    img.addEventListener('error', () => img.remove());
    head.appendChild(img);
  }
  box.appendChild(head);

  const rows = DETAIL_FIELDS
    .map((field) => ({ field, value: fieldValue(track.current, field) }))
    .filter((r) => r.value != null);

  if (!rows.length) {
    box.appendChild(el('div', 'note', 'This file has no metadata tags at all.'));
  } else {
    const grid = el('dl', 'fieldgrid two-col');
    for (const { field, value } of rows) {
      grid.appendChild(el('dt', null, field.label));
      grid.appendChild(el('dd', null, String(value)));
    }
    box.appendChild(grid);
  }
  if (!track.current.has_art) {
    box.appendChild(el('div', 'note-inline', 'No embedded cover art.'));
  }
  return box;
}

/** Before → after for every field that has a current or proposed value.
   Always rendered, even before Tag has run, so a file can be tagged
   entirely by hand from this panel. */
function renderProposedChanges(track) {
  const match = track.match;
  const box = el('div');
  const head = el('div', 'detail-section-head');
  head.appendChild(el('h4', null, 'Proposed changes'));
  box.appendChild(head);

  const editActions = () => {
    const actions = el('div', 'detail-actions');
    const editBtn = el('button', 'btn btn-sm', 'Edit by hand');
    editBtn.addEventListener('click', () => openEditor(track, box));
    actions.appendChild(editBtn);
    return actions;
  };

  if (!match) {
    box.appendChild(el('div', 'note',
      'Nothing proposed yet. Run Tag, or use Edit by hand below.'));
    box.appendChild(editActions());
    return box;
  }

  const proposed = match.proposed;
  const rows = DETAIL_FIELDS
    .map((field) => {
      const before = fieldValue(track.current, field);
      const rawAfter = fieldValue(proposed, field);
      // A field the match does not supply is left exactly as it is on disk -
      // every tag writer skips a null field rather than clearing it (verified
      // against actual writes, not just read from the source). Showing it as
      // "becomes empty" would be a straightforward lie about what Apply does,
      // so the effective value is the current one whenever nothing was
      // actually proposed for this field.
      const touched = rawAfter != null;
      return { field, before, after: touched ? rawAfter : before, touched };
    })
    .filter((r) => r.before != null || r.after != null);

  if (!rows.length) {
    box.appendChild(el('div', 'note', 'Nothing proposed for this file yet.'));
  } else {
    const changed = rows.filter(
      (r) => r.touched && String(r.before ?? '') !== String(r.after ?? ''));
    box.appendChild(el('div', 'help', changed.length
      ? `${changed.length} of ${rows.length} field${rows.length > 1 ? 's' : ''} `
        + 'would change if applied.'
      : 'Every field already matches — applying would just confirm the current tags.'));

    const grid = el('div', 'diffgrid');
    for (const label of ['Field', 'Current', '', 'Proposed', 'Conf.']) {
      grid.appendChild(el('div', 'diffgrid-head', label));
    }
    for (const { field, before, after, touched } of rows) {
      const isChanged = touched && String(before ?? '') !== String(after ?? '');
      grid.appendChild(el('div', 'diff-label', field.label));
      grid.appendChild(el('div', `diff-value ${before == null ? 'is-empty' : ''}`,
        before == null ? '(empty)' : String(before)));
      grid.appendChild(el('div', 'diff-arrow', isChanged ? '→' : '='));
      grid.appendChild(el('div',
        `diff-value ${isChanged ? 'changed' : ''} ${after == null ? 'is-empty' : ''}`,
        after == null ? '(empty)' : String(after)));
      const conf = match.field_confidence[field.key];
      grid.appendChild(el('span', `field-conf ${confClass(conf)}`,
        conf == null ? '—' : `${Math.round(conf)}%`));
    }
    box.appendChild(grid);
  }

  const actions = editActions();
  if (rows.some((r) => r.touched)) {
    const applyBtn = el('button', 'btn btn-sm btn-primary', 'Apply this file');
    applyBtn.addEventListener('click', () => applyTracks([track.path], false, false));
    actions.appendChild(applyBtn);
  }
  box.appendChild(actions);

  return box;
}

function renderDetail(track) {
  const wrap = el('div', 'detail');
  const match = track.match;

  // --- why this score ---------------------------------------------------
  const why = el('div');
  why.appendChild(el('h4', null, 'Why this confidence'));
  if (match && match.candidates.length) {
    const chosen = match.candidates[match.chosen_index] || match.candidates[0];
    for (const signal of chosen.signals) {
      const line = el('div', 'signal');
      const bar = el('div', 'signal-bar');
      const fill = el('span');
      fill.style.width = `${Math.round(signal.score * 100)}%`;
      bar.appendChild(fill);
      line.appendChild(bar);
      line.appendChild(el('span', 'signal-text', signal.detail));
      line.appendChild(el('span', 'signal-weight', `weight ${signal.weight}`));
      why.appendChild(line);
    }
    if (match.method) {
      why.appendChild(el('div', 'note', `Matched via: ${match.method}`));
    }
  } else {
    why.appendChild(el('div', 'note', 'No candidates were found for this file.'));
  }
  for (const note of (match?.notes || [])) why.appendChild(el('div', 'note', note));
  wrap.appendChild(why);

  // --- alternatives -----------------------------------------------------
  if (match && match.candidates.length) {
    const alts = el('div');
    alts.appendChild(el('h4', null, `Candidates (${match.candidates.length})`));
    match.candidates.forEach((cand, index) => {
      const node = el('div', `cand ${index === match.chosen_index ? 'is-chosen' : ''}`);
      const score = el('div', `cand-score ${confClass(cand.confidence)}`,
        `${Math.round(cand.confidence)}%`);
      score.style.color = `var(--${confClass(cand.confidence).slice(2)})`;
      node.appendChild(score);
      const body = el('div', 'cand-body');
      body.appendChild(el('div', 'cand-title', cand.tags.title || '(untitled)'));
      body.appendChild(el('div', 'cand-sub',
        cand.release_summary || trackDesc(cand.tags) || '—'));
      const details = [];
      if (cand.tags.track_no) {
        details.push(`Track ${cand.tags.track_no}${cand.tags.track_total ? `/${cand.tags.track_total}` : ''}`);
      }
      if (cand.length_s) details.push(fmtDuration(cand.length_s));
      if (cand.tags.isrc) details.push(cand.tags.isrc);
      if (details.length) body.appendChild(el('div', 'cand-detail', details.join('  ·  ')));
      node.appendChild(body);
      node.addEventListener('click', () => chooseCandidate(track.path, index));
      alts.appendChild(node);
    });
    wrap.appendChild(alts);
  }

  // --- current metadata and proposed changes -----------------------------
  // Current metadata is shown for every track, identified or not, so "what's
  // on this file right now" never requires running Identify first.
  wrap.appendChild(renderCurrentMetadata(track));
  // Proposed changes is likewise always shown: it doubles as the hand-editing
  // panel for a file Identify never matched, or that a candidate got wrong.
  wrap.appendChild(renderProposedChanges(track));

  // --- quality ----------------------------------------------------------
  const q = track.quality;
  if (q && (q.analysed || q.error)) {
    const box = el('div', 'detail-full');
    box.appendChild(el('h4', null, 'Audio quality'));

    if (q.error) box.appendChild(el('div', 'note', q.error));

    if (q.analysed) {
      const max = qualityScale().max_score;
      const worst = worstSeverity(q.issues);
      const label = worst ? severityInfo(worst).label : 'Clean';

      // Headline: the score, the word, and what they each mean.
      const head = el('div', 'quality-head');
      const badge = el('span',
        `qbadge ${{ high: 'q-high', medium: 'q-medium', low: 'q-low' }[worst] || 'q-ok'}`);
      badge.appendChild(el('span', null, String(q.score)));
      badge.appendChild(el('span', 'qbadge-sep', '·'));
      badge.appendChild(el('span', null, label));
      head.appendChild(badge);

      const summary = el('div', 'quality-summary');
      summary.appendChild(el('div', 'quality-verdict', qualityVerdict(q)));

      // Show the arithmetic rather than asserting the number.
      const counts = {};
      for (const issue of q.issues) {
        const info = severityInfo(issue.severity);
        if (!info.penalty) continue;
        const bucket = counts[info.label] || (counts[info.label] = { n: 0, points: 0 });
        bucket.n += 1;
        bucket.points += info.penalty;
      }
      const workings = Object.entries(counts)
        .sort((a, b) => b[1].points - a[1].points)        // worst first
        .map(([label, c]) =>
          `${c.points} for ${c.n} ${label.toLowerCase()} finding${c.n > 1 ? 's' : ''}`)
        .join(', ');
      summary.appendChild(el('div', 'quality-maths', workings
        ? `Score ${q.score} of ${max}. Deducted ${workings}.`
        : `Score ${q.score} of ${max}. Nothing deducted.`));

      const helpLink = el('button', 'link', 'What do these mean?');
      helpLink.addEventListener('click', openQualityHelp);
      summary.appendChild(helpLink);

      head.appendChild(summary);
      box.appendChild(head);
    }

    // Worst first: the finding that matters most should not be buried under
    // notes about the track being short.
    const order = qualityScale().severities.map((s) => s.key);
    const ranked = [...q.issues].sort(
      (a, b) => order.indexOf(a.severity) - order.indexOf(b.severity));

    for (const issue of ranked) {
      const info = severityInfo(issue.severity);
      const node = el('div', `issue sev-${issue.severity}`);
      node.appendChild(el('div', 'issue-sev'));
      const body = el('div');

      const heading = el('div', 'issue-heading');
      const chip = el('span', `sev-chip sev-chip-${issue.severity}`, info.label);
      if (info.penalty) chip.appendChild(el('span', 'sev-cost', `−${info.penalty}`));
      chip.title = info.meaning || '';
      heading.appendChild(chip);
      heading.appendChild(el('span', 'issue-title', issue.title));
      body.appendChild(heading);

      body.appendChild(el('div', 'issue-detail', issue.detail));
      node.appendChild(body);
      box.appendChild(node);
    }
    if (!q.issues.length && q.analysed) {
      box.appendChild(el('div', 'note', 'No problems found.'));
    }

    const interesting = [
      ['bitrate_kbps', 'Bitrate', (v) => `${v} kbps`],
      ['sample_rate', 'Sample rate', (v) => `${(v / 1000).toFixed(1)} kHz`],
      ['bit_depth', 'Bit depth', (v) => `${v}-bit`],
      ['spectral_cutoff_khz', 'Roll-off', (v) => `${v} kHz`],
      ['expected_cutoff_khz', 'Expected', (v) => `${v} kHz`],
      ['peak_dbfs', 'Peak', (v) => `${v} dBFS`],
      ['crest_factor_db', 'Crest', (v) => `${v} dB`],
      ['clipped_samples', 'Clipped', (v) => v.toLocaleString()],
      ['clicks', 'Clicks', (v) => String(v)],
      ['end_level_dbfs', 'End level', (v) => `${v} dBFS`],
      ['duration_delta_s', 'vs reference', (v) => `${v > 0 ? '-' : '+'}${Math.abs(v)}s`],
    ];
    const metrics = el('div', 'metrics');
    let shown = 0;
    for (const [key, label, fmt] of interesting) {
      const value = q.metrics[key];
      if (value == null || value === 0 && key !== 'clipped_samples' && key !== 'clicks') continue;
      const node = el('div', 'metric');
      node.appendChild(el('div', 'metric-k', label));
      node.appendChild(el('div', 'metric-v', fmt(value)));
      metrics.appendChild(node);
      shown++;
    }
    if (shown) box.appendChild(metrics);
    wrap.appendChild(box);
  }

  // --- destination ------------------------------------------------------
  if (track.planned_path) {
    const box = el('div', 'detail-full');
    box.appendChild(el('h4', null, 'Would be filed as'));
    box.appendChild(el('div', 'plan-path', track.planned_path));
    wrap.appendChild(box);
  }

  const pathBox = el('div', 'detail-full');
  pathBox.appendChild(el('h4', null, 'File'));
  pathBox.appendChild(el('div', 'plan-path', track.path));
  if (track.error) pathBox.appendChild(el('div', 'note', track.error));
  wrap.appendChild(pathBox);

  return wrap;
}

/* ------------------------------------------------------------ hand edits */

function openEditor(track, container) {
  const existing = container.querySelector('.editor');
  if (existing) { existing.remove(); return; }

  const editor = el('div', 'editor');
  const fields = [
    ['title', 'Title'], ['artist', 'Artist'], ['album', 'Album'],
    ['album_artist', 'Album artist'], ['track_no', 'Track no'],
    ['disc_no', 'Disc no'], ['date', 'Date'], ['genre', 'Genre'],
  ];
  const proposed = track.match ? track.match.proposed : null;
  const inputs = {};
  for (const [key, label] of fields) {
    const field = el('label', 'field');
    field.appendChild(el('span', null, label));
    const input = el('input');
    input.type = (key === 'track_no' || key === 'disc_no') ? 'number' : 'text';
    // Start from whatever is already proposed; for a file Identify has never
    // touched, fall back to what is already on the file so editing means
    // correcting a value rather than retyping it from nothing.
    const seed = (proposed && proposed[key] != null) ? proposed[key] : track.current[key];
    input.value = seed ?? '';
    inputs[key] = input;
    field.appendChild(input);
    editor.appendChild(field);
  }
  const save = el('button', 'btn btn-sm btn-primary', 'Save these values');
  save.addEventListener('click', async () => {
    const tags = {};
    for (const [key, input] of Object.entries(inputs)) {
      const raw = input.value.trim();
      if (raw === '') { tags[key] = null; continue; }
      const numeric = key === 'track_no' || key === 'disc_no';
      if (numeric && !Number.isFinite(Number(raw))) {
        toast(`${key === 'track_no' ? 'Track' : 'Disc'} number must be a number.`, 'error');
        return;
      }
      tags[key] = numeric ? Number(raw) : raw;
    }
    try {
      await api('/api/track/edit', { method: 'POST', body: { path: track.path, tags } });
      toast('Saved. Confidence for edited fields set to 100%.', 'success');
      await refreshTracks();
    } catch (err) { toast(err.message, 'error'); }
  });
  editor.appendChild(save);
  container.appendChild(editor);
}

async function chooseCandidate(path, index) {
  try {
    await api('/api/track/choose', { method: 'POST', body: { path, candidate_index: index } });
    await refreshTracks();
  } catch (err) { toast(err.message, 'error'); }
}

/* ----------------------------------------------------------- action bar */

function updateActionBar() {
  const bar = $('#actionBar');
  const count = state.selected.size;
  bar.hidden = count === 0;
  $('#selCount').textContent = String(count);

  const chosen = state.tracks.filter((t) => state.selected.has(t.path));
  const identified = chosen.filter((t) => t.match && t.match.candidates.length).length;
  const lowConf = chosen.filter((t) => t.match && t.bucket === 'low').length;
  const hints = [];
  if (identified < count) hints.push(`${count - identified} not identified yet`);
  if (lowConf) hints.push(`${lowConf} below your review threshold`);
  $('#selHint').textContent = hints.length ? `· ${hints.join(', ')}` : '';
  $('#btnApply').disabled = identified === 0;
}

/* ---------------------------------------------------------------- jobs */

function showJob(job) {
  const bar = $('#jobBar');
  if (!job || ['done', 'error', 'cancelled'].includes(job.status)) {
    bar.hidden = true;
    return;
  }
  bar.hidden = false;
  $('#jobTitle').textContent = job.message || job.kind;
  $('#jobDetail').textContent = job.detail || '';
  $('#jobFill').style.width = `${job.percent}%`;
  $('#jobCount').textContent = job.total ? `${job.done} / ${job.total}` : '';
}

function stopPolling(jobId) {
  clearInterval(state.polls.get(jobId));
  state.polls.delete(jobId);
  if (state.activeJobId === jobId) {
    // Hand the progress bar to whichever job is still running, if any.
    state.activeJobId = [...state.polls.keys()].pop() || null;
    if (!state.activeJobId) showJob(null);
  }
}

function pollJob(jobId, onDone) {
  if (state.polls.has(jobId)) return;
  state.activeJobId = jobId;
  state.polls.set(jobId, setInterval(async () => {
    let job;
    try { job = await api(`/api/jobs/${jobId}`); }
    catch (err) { stopPolling(jobId); toast(err.message, 'error'); return; }

    if (state.activeJobId === jobId) showJob(job);
    if (['done', 'error', 'cancelled'].includes(job.status)) {
      stopPolling(jobId);
      if (job.status === 'error') toast(job.error || 'Job failed', 'error');
      else if (job.status === 'cancelled') toast('Cancelled');
      else if (job.message) toast(job.message, 'success');
      if (onDone) onDone(job);
      await refreshAll();
    }
  }, 700));
}

async function startJob(endpoint, body, onDone) {
  try {
    const job = await api(endpoint, { method: 'POST', body });
    showJob(job);
    pollJob(job.id, onDone);
  } catch (err) { toast(err.message, 'error'); }
}

/* -------------------------------------------------------------- loading */

async function refreshStatus() {
  try {
    state.status = await api('/api/status');
    $('#appVersion').textContent = `v${state.status.version}`;
    $('#currentVersion').textContent = state.status.version;
    renderStats();
    renderCapabilityNotice();
    $('#btnIdentify').disabled = state.status.stats.total === 0;
    $('#btnQuality').disabled = state.status.stats.total === 0
      || !state.status.capabilities.quality_analysis;
    $('#btnExport').disabled = state.status.stats.total === 0;
    if (state.config && state.config.library_paths.length) {
      $('#libraryLabel').textContent = state.config.library_paths.join('  ·  ');
    }
  } catch (err) { toast(err.message, 'error'); }
}

async function refreshTracks() {
  const params = new URLSearchParams({
    q: state.filters.q,
    bucket: state.filters.bucket,
    issues: state.filters.issues,
    sort: state.filters.sort,
    desc: String(state.filters.desc),
    offset: String(state.offset),
    limit: String(PAGE_SIZE),
  });
  try {
    const data = await api(`/api/tracks?${params}`);
    state.tracks = data.tracks;
    state.total = data.total;
    renderTracks();
  } catch (err) { toast(err.message, 'error'); }
}

async function refreshAll() {
  await refreshStatus();
  await refreshTracks();
}

/* ------------------------------------------------------------- actions */

function selectedPaths() {
  return Array.from(state.selected);
}

async function applyTracks(paths, organize, dryRun) {
  const body = { paths, organize, dry_run: dryRun, write_art: true };
  if (dryRun) {
    try {
      const job = await api('/api/apply', { method: 'POST', body });
      showJob(job);
      pollJob(job.id, (finished) => showPreview(finished.result));
    } catch (err) { toast(err.message, 'error'); }
    return;
  }
  const label = organize ? 'write tags and move files' : 'write tags';
  if (!confirm(`This will ${label} for ${paths.length} file(s).\n\n`
    + 'Every change is journalled and can be undone from Settings → History.')) return;
  startJob('/api/apply', body, () => { state.selected.clear(); });
}

function showPreview(result) {
  const body = $('#previewBody');
  body.innerHTML = '';
  if (!result) { body.appendChild(el('p', null, 'Nothing to preview.')); }
  else {
    const summary = el('div', 'preview-summary');
    summary.textContent = `${result.tagged} file(s) would have tags written`
      + (result.planned?.length ? `, ${result.planned.length} would be moved or copied.` : '.');
    body.appendChild(summary);
    for (const move of (result.planned || []).slice(0, 300)) {
      const row = el('div', 'preview-move');
      row.appendChild(el('span', null, move.from));
      row.appendChild(el('span', null, '→'));
      row.appendChild(el('span', null, move.to));
      body.appendChild(row);
    }
    if ((result.errors || []).length) {
      body.appendChild(el('h4', null, 'Problems'));
      for (const e of result.errors.slice(0, 50)) {
        body.appendChild(el('div', 'note', `${e.path}: ${e.error}`));
      }
    }
  }
  $('#previewModal').hidden = false;
}

/* ------------------------------------------------------------ export to Plex */

function fmtTags(tags) {
  const bits = [tags.artist, tags.album, tags.title].filter(Boolean);
  return bits.length ? bits.join(' — ') : '(no tags)';
}

async function openExportFlow() {
  let cfg;
  try { cfg = await api('/api/config'); } catch (err) { toast(err.message, 'error'); return; }
  if (!cfg.organize_root) {
    toast('Set a Plex Music folder in Settings first.', 'error');
    await openSettings();
    document.querySelector('#settingsTabs .tab[data-tab="plex"]')?.click();
    return;
  }

  const paths = selectedPaths();
  try {
    const job = await api('/api/export/plan', { method: 'POST', body: { paths } });
    showJob(job);
    pollJob(job.id, (finished) => {
      if (finished.status === 'done') openExportReview(finished.result);
    });
  } catch (err) { toast(err.message, 'error'); }
}

function openExportReview(plan) {
  state.exportPlan = plan;
  state.exportResolutions = {};
  for (const item of plan.items) {
    if (!item.duplicate) state.exportResolutions[item.path] = 'export';
  }
  renderExportModal();
  $('#exportModal').hidden = false;
}

function renderExportModal() {
  const plan = state.exportPlan;
  const dupes = plan.items.filter((i) => i.duplicate);
  const ready = plan.items.filter((i) => !i.duplicate);

  const summary = $('#exportSummary');
  summary.innerHTML = '';
  summary.appendChild(el('div', 'note', `Plex folder: ${plan.plex_root}`));
  const bits = [];
  if (ready.length) bits.push(`${ready.length} track(s) will move directly`);
  if (dupes.length) bits.push(`${dupes.length} possible duplicate(s) need a decision`);
  if (plan.pending_apply.length) {
    bits.push(`${plan.pending_apply.length} track(s) have unapplied tag changes - Apply them first, they were skipped`);
  }
  if (plan.ineligible) bits.push(`${plan.ineligible} track(s) have no usable tags - skipped`);
  summary.appendChild(el('div', 'help', bits.join('. ') + '.'));

  const dupeBox = $('#exportDupes');
  dupeBox.innerHTML = '';
  if (dupes.length) {
    const bulk = el('div', 'export-bulk');
    const skipAll = el('button', 'btn btn-ghost btn-sm', 'Skip all duplicates');
    skipAll.addEventListener('click', () => {
      for (const i of dupes) setResolution(i.path, 'skip');
    });
    const keepAll = el('button', 'btn btn-ghost btn-sm', 'Keep both, all');
    keepAll.addEventListener('click', () => {
      for (const i of dupes) setResolution(i.path, 'export');
    });
    bulk.appendChild(skipAll);
    bulk.appendChild(keepAll);
    dupeBox.appendChild(bulk);

    for (const item of dupes) dupeBox.appendChild(renderDupeRow(item));
  }

  const readyBox = $('#exportReady');
  readyBox.innerHTML = '';
  if (ready.length) {
    const details = el('details', 'export-ready-list');
    const summaryEl = el('summary', null, `${ready.length} track(s) ready, no duplicate found`);
    details.appendChild(summaryEl);
    for (const item of ready) {
      details.appendChild(el('div', 'export-ready-row', `${basename(item.path)} → ${item.dest}`));
    }
    readyBox.appendChild(details);
  }

  updateExportCommitState();
}

function basename(path) {
  return path.split(/[\\/]/).pop();
}

function renderDupeRow(item) {
  const row = el('div', 'export-dupe-row');
  row.dataset.path = item.path;

  const incoming = el('div', 'export-dupe-side');
  incoming.appendChild(el('div', 'export-dupe-label', 'Incoming (ingest folder)'));
  incoming.appendChild(el('div', 'export-dupe-name', basename(item.path)));

  const existing = el('div', 'export-dupe-side');
  const kind = item.duplicate.kind === 'mbid' ? 'Same MusicBrainz recording' : 'Looks like the same song';
  existing.appendChild(el('div', 'export-dupe-label', `Already in Plex — ${kind}`));
  existing.appendChild(el('div', 'export-dupe-name', fmtTags(item.duplicate.existing_tags)));
  existing.appendChild(el('div', 'export-dupe-sub', basename(item.duplicate.existing_path)));

  const actions = el('div', 'export-dupe-actions');
  const options = [
    ['skip', 'Skip (keep Plex copy)'],
    ['replace', 'Replace Plex copy'],
    ['export', 'Keep both'],
  ];
  for (const [value, label] of options) {
    const btn = el('button', 'btn btn-sm export-dupe-choice', label);
    btn.dataset.value = value;
    btn.addEventListener('click', () => setResolution(item.path, value));
    actions.appendChild(btn);
  }

  row.appendChild(incoming);
  row.appendChild(existing);
  row.appendChild(actions);
  refreshDupeRowChoice(row);
  return row;
}

function setResolution(path, action) {
  state.exportResolutions[path] = action;
  const row = document.querySelector(`.export-dupe-row[data-path="${CSS.escape(path)}"]`);
  if (row) refreshDupeRowChoice(row);
  updateExportCommitState();
}

function refreshDupeRowChoice(row) {
  const chosen = state.exportResolutions[row.dataset.path];
  row.querySelectorAll('.export-dupe-choice').forEach((btn) => {
    btn.classList.toggle('is-chosen', btn.dataset.value === chosen);
  });
}

function updateExportCommitState() {
  const plan = state.exportPlan;
  if (!plan) return;
  const unresolved = plan.items.filter((i) => !state.exportResolutions[i.path]).length;
  const total = plan.items.length;
  $('#btnExportCommit').disabled = total === 0 || unresolved > 0;
  $('#exportStatus').textContent = unresolved
    ? `${unresolved} duplicate(s) still need a decision`
    : (total ? `${total} track(s) ready to export` : 'Nothing to export');
}

async function commitExportPlan() {
  const plan = state.exportPlan;
  if (!plan) return;
  const items = plan.items.map((item) => ({
    path: item.path,
    dest: item.dest,
    action: state.exportResolutions[item.path] || 'export',
    existing_path: item.duplicate ? item.duplicate.existing_path : null,
  }));
  $('#exportModal').hidden = true;
  startJob('/api/export/commit', { items }, () => {
    state.selected.clear();
    state.exportPlan = null;
    state.exportResolutions = {};
  });
}

/* --------------------------------------------------------- folder picker */

function updatePlexFolderLabel() {
  const root = state.config?.organize_root;
  const btn = $('#btnPickPlexFolder');
  if (root) {
    const name = root.split(/[\\/]/).filter(Boolean).pop() || root;
    $('#plexFolderLabel').textContent = name;
    btn.title = `Plex Music folder: ${root}`;
  } else {
    $('#plexFolderLabel').textContent = 'Plex folder';
    btn.title = 'Set your Plex Music folder';
  }
}

const PICKER_TITLES = {
  'library': 'Choose your music folder',
  'plex-root': 'Choose your Plex Music folder',
  'plex-root-quick': 'Choose your Plex Music folder',
};

async function openPicker(path = '', target = 'library') {
  state.picker.target = target;
  $('#pickerTitle').textContent = PICKER_TITLES[target] || PICKER_TITLES.library;
  $('#pickerModal').hidden = false;
  await loadPicker(path);
}

async function loadPicker(path) {
  try {
    const data = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    state.picker.path = data.path;
    state.picker.chosen = data.path;
    $('#pickerPath').textContent = data.path || 'This computer';
    $('#btnPickConfirm').disabled = !data.path;
    $('#pickerCount').textContent = data.audio_files
      ? `${data.audio_files} audio file(s) directly in this folder`
      : (data.path ? 'No audio files directly here (subfolders still count)' : '');

    const list = $('#pickerList');
    list.innerHTML = '';
    if (data.parent != null && data.path) {
      const up = el('div', 'picker-item', '↑  ..');
      up.addEventListener('click', () => loadPicker(data.parent));
      list.appendChild(up);
    }
    for (const entry of data.entries) {
      const item = el('div', 'picker-item');
      item.appendChild(el('span', null, entry.kind === 'drive' ? '💾' : '📁'));
      item.appendChild(el('span', null, entry.name));
      item.addEventListener('click', () => loadPicker(entry.path));
      list.appendChild(item);
    }
  } catch (err) { toast(err.message, 'error'); }
}

/* ------------------------------------------------------------- settings */

const CONFIG_FIELDS = [
  'acoustid_api_key', 'fpcalc_path', 'musicbrainz_contact', 'musicbrainz_rate_limit',
  'auto_apply_threshold', 'review_threshold', 'preserve_existing_tags',
  'write_cover_art', 'write_cover_file', 'write_musicbrainz_ids', 'id3v2_version',
  'various_artists_name', 'organize_enabled', 'organize_mode', 'organize_root',
  'folder_template', 'file_template', 'ffmpeg_path', 'quality_workers',
  'quality_max_seconds', 'update_check_enabled',
];

async function openSettings() {
  try {
    state.config = await api('/api/config');
  } catch (err) { toast(err.message, 'error'); return; }

  for (const key of CONFIG_FIELDS) {
    const input = document.getElementById(`cfg_${key}`);
    if (!input) continue;
    const value = state.config[key];
    if (input.type === 'checkbox') input.checked = Boolean(value);
    else input.value = value == null ? '' : String(value);
  }

  const caps = state.config._capabilities || {};
  // A blank key is fine when the app carries its own - say so, rather than
  // leave an empty box that looks like something still needs doing.
  $('#acoustidHint').textContent = caps.acoustid_key_set
    ? 'Using your own key.'
    : (caps.acoustid_builtin_key
      ? 'Using the key built into MusicTagger. Leave blank, or paste your own to use it instead.'
      : 'No key yet: fingerprinting is off until you add one.');
  $('#fpcalcHint').textContent = caps.fpcalc
    ? `Found: ${caps.fpcalc}`
    : `Not found. The download comes from ${state.config._fpcalc_url || state.config._fpcalc_homepage}.`;

  updateTemplatePreview();
  updatePlexFolderLabel();
  await loadHistory();
  await refreshLookupCacheInfo();
  $('#settingsModal').hidden = false;
}

async function saveSettings() {
  const payload = {};
  for (const key of CONFIG_FIELDS) {
    const input = document.getElementById(`cfg_${key}`);
    if (!input) continue;
    payload[key] = input.type === 'checkbox' ? input.checked : input.value;
  }
  try {
    state.config = await api('/api/config', { method: 'POST', body: payload });
    $('#saveState').textContent = 'Saved';
    setTimeout(() => { $('#saveState').textContent = ''; }, 2500);
    updatePlexFolderLabel();
    await refreshStatus();
  } catch (err) { toast(err.message, 'error'); }
}

function updateTemplatePreview() {
  const folder = $('#cfg_folder_template').value || '{album_artist}/{album}';
  const file = $('#cfg_file_template').value || '{track:02d} - {title}';
  const sample = {
    album_artist: 'Portishead', artist: 'Portishead', album: 'Dummy',
    title: 'Glory Box', year: '1994', year_suffix: ' (1994)',
    track: '10', disc: '1', disc_prefix: '', genre: 'Trip Hop',
  };
  const fill = (tpl) => tpl.replace(/\{(\w+)(?::[^}]+)?\}/g,
    (_, key) => sample[key] ?? '');
  $('#templatePreview').textContent =
    `…/${fill(folder)}/${fill(file)}.flac`;
}

async function refreshLookupCacheInfo() {
  try {
    const data = await api('/api/lookup-cache');
    $('#lookupCacheInfo').textContent = `${data.entries.toLocaleString()} cached response(s)`;
  } catch (err) { $('#lookupCacheInfo').textContent = ''; }
}

async function loadHistory() {
  const list = $('#historyList');
  list.innerHTML = '';
  try {
    const data = await api('/api/history');
    if (!data.batches.length) {
      list.appendChild(el('p', 'help', 'Nothing has been applied yet.'));
      return;
    }
    for (const batch of data.batches) {
      const item = el('div', 'history-item');
      const body = el('div', 'grow');
      body.appendChild(el('div', null, batch.description || 'Batch'));
      const counts = batch.counts || {};
      const parts = [];
      if (counts.tagged) parts.push(`${counts.tagged} tagged`);
      if (counts.moved) parts.push(`${counts.moved} moved`);
      if (counts.copied) parts.push(`${counts.copied} copied`);
      body.appendChild(el('div', 'history-when',
        `${new Date(batch.started * 1000).toLocaleString()} · ${parts.join(', ') || batch.entry_count + ' entries'}`));
      item.appendChild(body);

      if (batch.undone) {
        // Undoing twice would move back files that were already put back.
        item.appendChild(el('span', 'history-undone', 'Undone'));
      } else {
        const undo = el('button', 'btn btn-sm', 'Undo');
        undo.addEventListener('click', async () => {
          if (!confirm('Restore the previous tags and move any files back?')) return;
          await startJob('/api/undo', { batch_id: batch.id }, () => loadHistory());
        });
        item.appendChild(undo);
      }
      list.appendChild(item);
    }
  } catch (err) { toast(err.message, 'error'); }
}

/* ------------------------------------------------------ view + columns */

function applyView() {
  const grid = state.view === 'grid';
  $$('#viewToggle .seg-btn').forEach((b) =>
    b.classList.toggle('is-active', (b.dataset.view === 'grid') === grid));
  $('#btnColumns').hidden = !grid;
  $('#valuesMode').hidden = !grid;
  // Header clicks are the sort control in grid view; the select would just
  // be a second, competing one.
  $('#sortBy').hidden = grid;
  $('.table-wrap').classList.toggle('is-grid', grid);
  store('musictagger-view', state.view);
}

function syncSortSelect() {
  const select = $('#sortBy');
  const value = state.filters.desc && state.filters.sort === 'confidence'
    ? 'confidence_desc' : state.filters.sort;
  select.value = [...select.options].some((o) => o.value === value) ? value : '';
}

function renderColumnPicker() {
  const wrap = $('#columnPicker');
  wrap.innerHTML = '';
  for (const col of COLUMNS) {
    const row = el('label', 'column-option');
    const box = el('input');
    box.type = 'checkbox';
    box.checked = state.columns.includes(col.key);
    box.addEventListener('change', () => {
      if (box.checked) {
        // Keep the canonical COLUMNS order rather than click order.
        state.columns = COLUMNS.filter(
          (c) => c.key === col.key || state.columns.includes(c.key)).map((c) => c.key);
      } else {
        state.columns = state.columns.filter((k) => k !== col.key);
      }
      if (!state.columns.length) {
        state.columns = ['title'];
        box.checked = col.key === 'title';
        toast('At least one column has to stay visible.');
      }
      store('musictagger-columns', state.columns);
      renderTracks();
    });
    row.appendChild(box);
    row.appendChild(el('span', null, col.pickerLabel || col.label));
    if (col.tag) row.appendChild(el('span', 'column-tag', 'tag'));
    if (col.hint) row.title = col.hint;
    wrap.appendChild(row);
  }
}

/* -------------------------------------------------------- quality help */

function openQualityHelp() {
  const table = $('#scaleTable');
  table.innerHTML = '';
  for (const severity of qualityScale().severities) {
    const row = el('div', 'scale-row');
    const chip = el('span', `sev-chip sev-chip-${severity.key}`, severity.label);
    row.appendChild(chip);
    row.appendChild(el('span', 'scale-cost',
      severity.penalty ? `−${severity.penalty}` : 'no cost'));
    row.appendChild(el('span', 'scale-meaning', severity.meaning || ''));
    table.appendChild(row);
  }
  $('#qualityHelpModal').hidden = false;
}

/* ---------------------------------------------------------------- theme */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('musictagger-theme', theme);
}

/* --------------------------------------------------------------- wiring */

function wire() {
  // theme
  applyTheme(localStorage.getItem('musictagger-theme') || 'system');
  $('#btnTheme').addEventListener('click', () => {
    const order = ['system', 'light', 'dark'];
    const next = order[(order.indexOf(document.documentElement.dataset.theme) + 1) % 3];
    applyTheme(next);
    toast(`Theme: ${next}`);
  });

  // folder picker
  $('#btnPickFolder').addEventListener('click', () => openPicker(state.picker.path, 'library'));
  $('#btnPickFolder2').addEventListener('click', () => openPicker(state.picker.path, 'library'));
  $('#btnPickPlexRoot').addEventListener('click', () => openPicker($('#cfg_organize_root').value, 'plex-root'));
  $('#btnPickPlexFolder').addEventListener('click', async () => {
    let cfg;
    try { cfg = await api('/api/config'); } catch (err) { toast(err.message, 'error'); return; }
    openPicker(cfg.organize_root || '', 'plex-root-quick');
  });
  $('#btnPickConfirm').addEventListener('click', async () => {
    const chosen = state.picker.chosen;
    const target = state.picker.target;
    $('#pickerModal').hidden = true;
    if (!chosen) return;

    if (target === 'plex-root') {
      // Settings is open behind this - just fill the field, Save persists it
      // along with anything else changed in the panel.
      $('#cfg_organize_root').value = chosen;
      updateTemplatePreview();
      return;
    }

    if (target === 'plex-root-quick') {
      // Picked straight from the header, with no Settings panel open to
      // "Save" from - persist it immediately.
      try {
        state.config = await api('/api/config', { method: 'POST', body: { organize_root: chosen } });
        updatePlexFolderLabel();
        toast(`Plex Music folder set to:\n${chosen}`, 'success');
      } catch (err) { toast(err.message, 'error'); }
      return;
    }

    try {
      // MusicTagger remembers one current library folder, not a growing
      // list of everywhere it has ever pointed - so picking a genuinely
      // different folder replaces it, and clears the tracks that came from
      // the old one (nothing on disk is touched; re-pointing it back would
      // just re-scan). Re-picking the *same* folder is just a normal
      // re-scan and should not wipe anything.
      const cfg = await api('/api/config');
      const previous = (cfg.library_paths || [])[0] || '';
      const isNewFolder = previous.toLowerCase() !== chosen.toLowerCase();

      let replace = false;
      if (isNewFolder) {
        const current = await api('/api/status');
        const total = current.stats.total;
        if (total > 0 && !confirm(
          `Switch the library folder to:\n${chosen}\n\n`
          + `MusicTagger tracks one folder at a time. Switching clears the ${total} `
          + 'track(s) currently tracked from the previous folder - nothing on your '
          + 'disk is touched, and you can point it back at any time.'
        )) {
          return;
        }
        replace = true;
      }

      await api('/api/scan', { method: 'POST', body: { paths: [chosen], recursive: true, replace } })
        .then((job) => { showJob(job); pollJob(job.id); });
      state.config = await api('/api/config');
      $('#libraryLabel').textContent = chosen;
    } catch (err) { toast(err.message, 'error'); }
  });

  // primary actions
  $('#btnIdentify').addEventListener('click', () => {
    const paths = selectedPaths();
    startJob('/api/identify', { paths, only_pending: paths.length === 0 });
  });
  $('#btnQuality').addEventListener('click', () => {
    const paths = selectedPaths();
    startJob('/api/quality', { paths, only_pending: paths.length === 0 });
  });
  $('#btnExport').addEventListener('click', openExportFlow);
  $('#btnExportCommit').addEventListener('click', commitExportPlan);
  $('#btnCancelJob').addEventListener('click', () => {
    if (state.activeJobId) api(`/api/jobs/${state.activeJobId}/cancel`, { method: 'POST' });
  });

  // convert
  $('#btnConvert').addEventListener('click', () => {
    const paths = selectedPaths();
    if (!paths.length) return;
    const select = $('#convertFormat');
    const label = select.options[select.selectedIndex].text;
    if (!confirm(`Convert ${paths.length} file(s) to ${label}?\n\n`
      + 'Each converted copy is saved next to its original, with the same tags and '
      + 'cover art. Originals are not changed or deleted. Files already in this '
      + 'format are skipped.\n\nConverting from one lossy format to another '
      + '(for example MP3 to M4A) loses a little quality; converting never adds '
      + 'quality that was not there.')) return;
    startJob('/api/convert', { paths, format: select.value }, (job) => {
      const result = job.result || {};
      if (result.failed && result.errors?.length) {
        toast(`${result.failed} file(s) could not be converted: ${result.errors[0].error}`, 'error');
      }
    });
  });

  // apply
  $('#btnApply').addEventListener('click', () =>
    applyTracks(selectedPaths(), $('#optOrganize').checked, false));
  $('#btnPreview').addEventListener('click', () =>
    applyTracks(selectedPaths(), $('#optOrganize').checked, true));
  $('#btnPreviewApply').addEventListener('click', () => {
    $('#previewModal').hidden = true;
    applyTracks(selectedPaths(), $('#optOrganize').checked, false);
  });

  // filters
  let searchTimer;
  $('#search').addEventListener('input', (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.filters.q = e.target.value;
      state.offset = 0;
      refreshTracks();
    }, 250);
  });
  $('#bucketChips').addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    $$('#bucketChips .chip').forEach((c) => c.classList.remove('is-active'));
    chip.classList.add('is-active');
    state.filters.bucket = chip.dataset.bucket;
    state.offset = 0;
    refreshTracks();
  });
  $('#issueFilter').addEventListener('change', (e) => {
    state.filters.issues = e.target.value;
    state.offset = 0;
    refreshTracks();
  });
  $('#sortBy').addEventListener('change', (e) => {
    const value = e.target.value;
    state.filters.desc = value.endsWith('_desc');
    state.filters.sort = value.replace(/_desc$/, '');
    state.offset = 0;
    refreshTracks();
  });

  // view mode and columns
  $('#viewToggle').addEventListener('click', (e) => {
    const btn = e.target.closest('.seg-btn');
    if (!btn || btn.dataset.view === state.view) return;
    state.view = btn.dataset.view;
    applyView();
    renderTracks();
  });
  $('#btnColumns').addEventListener('click', () => {
    renderColumnPicker();
    $('#columnsModal').hidden = false;
  });
  $('#btnResetColumns').addEventListener('click', () => {
    state.columns = [...DEFAULT_COLUMNS];
    store('musictagger-columns', state.columns);
    renderColumnPicker();
    renderTracks();
  });
  $('#valuesMode').addEventListener('change', (e) => {
    state.valuesMode = e.target.value;
    store('musictagger-values', state.valuesMode);
    renderTracks();
  });

  // pager
  $('#btnPrev').addEventListener('click', () => {
    state.offset = Math.max(0, state.offset - PAGE_SIZE);
    refreshTracks();
  });
  $('#btnNext').addEventListener('click', () => {
    state.offset += PAGE_SIZE;
    refreshTracks();
  });

  // The quality "?" lives in the table header, which is rebuilt on every
  // render, so it wires itself up in qualityHelpDot() rather than here.

  // settings
  $('#btnSettings').addEventListener('click', openSettings);
  $('#btnSaveSettings').addEventListener('click', saveSettings);
  $('#btnUpdateNow').addEventListener('click', async () => {
    const updates = desktopUpdates();
    if (!updates) return;
    if (state.desktopUpdate?.status === 'ready') { await installDesktopUpdate(); return; }
    state.desktopUpdate = await updates.check();
    renderDesktopUpdate();
  });
  $('#btnCheckUpdate').addEventListener('click', async () => {
    $('#updateStatusHint').textContent = 'Checking\u2026';
    const info = await checkForUpdate({ force: true });
    $('#updateStatusHint').textContent = updateStatusText(info);
  });
  $('#settingsTabs').addEventListener('click', (e) => {
    const tab = e.target.closest('.tab');
    if (!tab) return;
    $$('#settingsTabs .tab').forEach((t) => t.classList.remove('is-active'));
    tab.classList.add('is-active');
    $$('.tab-panel').forEach((p) =>
      p.classList.toggle('is-active', p.dataset.panel === tab.dataset.tab));
  });
  $('#cfg_folder_template').addEventListener('input', updateTemplatePreview);
  $('#cfg_file_template').addEventListener('input', updateTemplatePreview);
  $('#btnInstallFpcalc').addEventListener('click', async () => {
    const url = state.config?._fpcalc_url;
    if (!confirm(`Download fpcalc from:\n${url}\n\nProceed?`)) return;
    await startJob('/api/install-fpcalc', {}, () => openSettings());
  });
  $('#btnClearLookupCache').addEventListener('click', async () => {
    try {
      const result = await api('/api/lookup-cache/clear', { method: 'POST' });
      toast(`Cleared ${result.cleared.toLocaleString()} cached lookup(s). `
          + 'Run Tag again to look everything up fresh.', 'success');
      await refreshLookupCacheInfo();
    } catch (err) { toast(err.message, 'error'); }
  });

  // modal close buttons
  $$('[data-close]').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.getElementById(btn.dataset.close).hidden = true;
    });
  });
  $$('.modal').forEach((modal) => {
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.hidden = true; });
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') $$('.modal').forEach((m) => { m.hidden = true; });
  });
}

/* ----------------------------------------------------------------- boot */

(async function main() {
  wire();
  $('#valuesMode').value = state.valuesMode;
  applyView();
  syncSortSelect();
  try { state.config = await api('/api/config'); } catch (_) { /* shown below */ }
  updatePlexFolderLabel();
  await refreshAll();

  // If a job was already running when the page loaded, latch onto it.
  const active = state.status?.active_jobs || [];
  for (const job of active) pollJob(job.id);
  if (active.length) showJob(active[active.length - 1]);

  initDesktopUpdates();

  // Deliberately last and deliberately not awaited: this one can go out to
  // the network, and nothing on the page should wait on it to become usable.
  checkForUpdate();
})();
