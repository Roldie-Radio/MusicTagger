// Automatic updates for the packaged desktop app.
//
// electron-updater reads the latest *published* GitHub release (drafts are
// invisible to it), downloads the new installer in the background, and runs it
// when the app quits. The backend's own update check (musictag/update.py)
// still decides what the banner says; this file is what actually installs.
//
// It honours the same "Check for new versions" setting as the banner: with it
// switched off, no request is made at all. The setting is read from the
// backend before every check rather than once at start-up, so turning it off
// in Settings takes effect without a restart.
//
// Nothing is installed from under a running job. A finished download waits
// until the backend reports no active jobs before offering a restart, and
// "Later" defers the install to the next time the app is closed.

const { dialog } = require('electron');
const http = require('http');
const { autoUpdater } = require('electron-updater');

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

let started = false;

function startAutoUpdate({ port, getWindow, beforeInstall }) {
  // A second window (macOS "activate") must not start a second schedule.
  if (started) return;
  started = true;

  let downloadedVersion = null;
  let prompted = false;

  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  // Only ever move forwards along published, non-prerelease versions.
  autoUpdater.allowPrerelease = false;
  autoUpdater.allowDowngrade = false;
  autoUpdater.logger = console;

  async function checkingAllowed() {
    try {
      const cfg = await getJson(port, '/api/config');
      return cfg.update_check_enabled !== false;
    } catch (err) {
      // If we cannot even read the setting, do not assume permission to go
      // out to the network on the user's behalf.
      console.warn('Update check skipped, could not read settings:', err.message);
      return false;
    }
  }

  async function check() {
    // Once an update is downloaded there is nothing more to fetch until the
    // app restarts into it.
    if (downloadedVersion) return;
    if (!(await checkingAllowed())) return;
    try {
      await autoUpdater.checkForUpdates();
    } catch (err) {
      // An unreachable GitHub is not worth interrupting anyone over.
      console.warn('Update check failed:', err && err.message);
    }
  }

  async function backendBusy() {
    try {
      const status = await getJson(port, '/api/status');
      return Array.isArray(status.active_jobs) && status.active_jobs.length > 0;
    } catch (_) {
      return false;
    }
  }

  async function offerRestart() {
    if (prompted || !downloadedVersion) return;
    if (await backendBusy()) {
      setTimeout(offerRestart, BUSY_RETRY_MS);
      return;
    }
    prompted = true;

    const win = getWindow();
    const options = {
      type: 'info',
      buttons: ['Restart now', 'Later'],
      defaultId: 0,
      cancelId: 1,
      title: 'Update ready',
      message: `MusicTagger ${downloadedVersion} is ready to install.`,
      detail: 'Restart now to finish updating, or it will be installed '
            + 'automatically the next time you close MusicTagger.',
    };
    const { response } = win
      ? await dialog.showMessageBox(win, options)
      : await dialog.showMessageBox(options);
    if (response !== 0) return;

    // The backend lives inside the install folder; it has to be gone before
    // the installer tries to replace its files.
    if (beforeInstall) beforeInstall();
    // Silent, because this is an update of an install the user already set
    // up - it keeps their chosen folder - and relaunch once it is done.
    autoUpdater.quitAndInstall(true, true);
  }

  autoUpdater.on('update-downloaded', (info) => {
    downloadedVersion = info && info.version;
    offerRestart();
  });
  autoUpdater.on('error', (err) => {
    console.warn('Auto-update error:', err && err.message);
  });

  setTimeout(check, FIRST_CHECK_DELAY_MS);
  setInterval(check, CHECK_INTERVAL_MS);
}

module.exports = { startAutoUpdate };
