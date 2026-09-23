// Automatic updates for the packaged desktop app.
//
// electron-updater reads the latest *published* GitHub release (drafts are
// invisible to it), downloads the new installer in the background, and runs it
// when the app quits. The backend's own update check (musictag/update.py)
// still decides what the banner says; this file is what actually installs.
//
// Automatic checks honour the "Check for new versions" setting: with it
// switched off, no request is made on the app's own initiative. The setting
// is read from the backend before every check rather than once at start-up,
// so turning it off in Settings takes effect without a restart. A check the
// user asks for with "Update now" (through preload.js) always runs.
//
// Nothing is installed from under a running job. A finished download waits
// until the backend reports no active jobs before offering a restart, and
// "Later" defers the install to the next time the app is closed.

const { app, dialog, ipcMain } = require('electron');
const http = require('http');
const { isBackendUrl } = require('./links');

// Same cadence as the backend check, and the first check waits until the
// window has had time to load so it never competes with start-up.
const FIRST_CHECK_DELAY_MS = 30 * 1000;
const CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;
const BUSY_RETRY_MS = 60 * 1000;

function getJson(port, path) {
  return new Promise((resolve, reject) => {
    const req = http.get({ host: '127.0.0.1', port, path, timeout: 5000 }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`${path} answered ${res.statusCode}`));
          return;
        }
        try { resolve(JSON.parse(body)); } catch (err) { reject(err); }
      });
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error(`${path} timed out`)); });
  });
}

// What the UI shows. `status` is one of: unsupported (running from source),
// idle, checking, up-to-date, downloading, ready, error.
const state = {
  status: app.isPackaged ? 'idle' : 'unsupported',
  current: app.getVersion(),
  available: null,
  percent: 0,
  error: '',
};

let ctx = null;          // { port, getWindow, beforeInstall } once started
let manualRequest = false;
let prompted = false;

function setState(patch) {
  Object.assign(state, patch);
  const win = ctx && ctx.getWindow();
  if (win && !win.isDestroyed()) win.webContents.send('updates:state', { ...state });
}

function autoUpdater() {
  // Loaded lazily: requiring it outside a packaged app logs noise about a
  // missing app-update.yml, and running from source never updates anyway.
  return require('electron-updater').autoUpdater;
}

async function backendBusy() {
  try {
    const status = await getJson(ctx.port, '/api/status');
    return Array.isArray(status.active_jobs) && status.active_jobs.length > 0;
  } catch (_) {
    return false;
  }
}

async function checkingAllowed() {
  try {
    const cfg = await getJson(ctx.port, '/api/config');
    return cfg.update_check_enabled !== false;
  } catch (err) {
    // If we cannot even read the setting, do not assume permission to go
    // out to the network on the user's behalf.
    console.warn('Update check skipped, could not read settings:', err.message);
    return false;
  }
}

async function check({ manual = false } = {}) {
  if (!ctx || state.status === 'unsupported') return { ...state };
  // Already fetching, or already have it: nothing more to do until restart.
  if (['checking', 'downloading', 'ready'].includes(state.status)) return { ...state };
  if (!manual && !(await checkingAllowed())) return { ...state };
  if (manual) manualRequest = true;

  setState({ status: 'checking', error: '' });
  try {
    await autoUpdater().checkForUpdates();
    // Every normal outcome fires an event that moves the status on; this
    // only catches a check that ended without one, so it never sticks.
    if (state.status === 'checking') setState({ status: 'idle' });
  } catch (err) {
    // An unreachable GitHub is not worth a dialog; the UI shows it when asked.
    console.warn('Update check failed:', err && err.message);
    setState({ status: 'error', error: (err && err.message) || 'Update check failed' });
  }
  return { ...state };
}

async function install() {
  if (state.status !== 'ready') {
    return { ok: false, error: 'No update has been downloaded yet.' };
  }
  if (await backendBusy()) {
    return { ok: false, error: 'Wait for the current job to finish (or cancel it) first.' };
  }
  // The backend lives inside the install folder; it has to be gone before
  // the installer tries to replace its files.
  if (ctx.beforeInstall) ctx.beforeInstall();
  // Silent, because this is an update of an install the user already set
  // up - it keeps their chosen folder - and relaunch once it is done.
  autoUpdater().quitAndInstall(true, true);
  return { ok: true };
}

async function offerRestart() {
  // A download the user started from Settings is finished from there too:
  // the button turns into "Restart and install", no dialog on top of it.
  if (prompted || manualRequest || state.status !== 'ready') return;
  if (await backendBusy()) {
    setTimeout(offerRestart, BUSY_RETRY_MS);
    return;
  }
  prompted = true;

  const win = ctx.getWindow();
  const options = {
    type: 'info',
    buttons: ['Restart now', 'Later'],
    defaultId: 0,
    cancelId: 1,
    title: 'Update ready',
    message: `MusicTagger ${state.available} is ready to install.`,
    detail: 'Restart now to finish updating, or it will be installed '
          + 'automatically the next time you close MusicTagger.',
  };
  const { response } = win && !win.isDestroyed()
    ? await dialog.showMessageBox(win, options)
    : await dialog.showMessageBox(options);
  if (response === 0) await install();
}

// Only the app's own page may drive updates - never some other page that
// ended up in the window. An exact origin match, not a prefix test:
// "http://127.0.0.1:8731@evil.example/" starts with "http://127.0.0.1:" but
// its host is evil.example (see links.js).
function fromOwnPage(event) {
  const url = (event.senderFrame && event.senderFrame.url) || '';
  if (!ctx) return false;
  return isBackendUrl(url, `http://127.0.0.1:${ctx.port}`);
}

// Registered in every build, so the UI always gets an answer - running from
// source it is simply "unsupported".
function registerUpdateIpc() {
  ipcMain.handle('updates:get-state', (event) =>
    (fromOwnPage(event) ? { ...state } : null));
  ipcMain.handle('updates:check', (event) =>
    (fromOwnPage(event) ? check({ manual: true }) : null));
  ipcMain.handle('updates:install', (event) =>
    (fromOwnPage(event) ? install() : { ok: false, error: 'Refused' }));
}

function startAutoUpdate({ port, getWindow, beforeInstall }) {
  // A second window (macOS "activate") must not start a second schedule.
  if (ctx) return;
  ctx = { port, getWindow, beforeInstall };

  const updater = autoUpdater();
  updater.autoDownload = true;
  updater.autoInstallOnAppQuit = true;
  // Only ever move forwards along published, non-prerelease versions.
  updater.allowPrerelease = false;
  updater.allowDowngrade = false;
  updater.logger = console;

  updater.on('update-available', (info) => {
    setState({ status: 'downloading', available: info && info.version, percent: 0 });
  });
  updater.on('update-not-available', () => {
    manualRequest = false;
    setState({ status: 'up-to-date', available: null });
  });
  updater.on('download-progress', (progress) => {
    setState({ status: 'downloading', percent: Math.round((progress && progress.percent) || 0) });
  });
  updater.on('update-downloaded', (info) => {
    setState({ status: 'ready', available: info && info.version, percent: 100 });
    offerRestart();
  });
  updater.on('error', (err) => {
    console.warn('Auto-update error:', err && err.message);
    // A failure after the download finished does not undo the download.
    if (state.status !== 'ready') {
      setState({ status: 'error', error: (err && err.message) || 'Update failed' });
    }
  });

  setTimeout(() => check(), FIRST_CHECK_DELAY_MS);
  setInterval(() => check(), CHECK_INTERVAL_MS);
}

module.exports = { registerUpdateIpc, startAutoUpdate };
