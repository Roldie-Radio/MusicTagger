// Electron shell for MusicTagger.
//
// All the actual application logic - tagging, MusicBrainz matching, quality
// analysis - lives in the Python backend this file spawns as a child process.
// This file's only jobs are: find the backend and its bundled tools, start
// the backend, wait for it to be healthy, show a window pointed at it, and
// shut everything down cleanly when the window closes. Installing updates
// is updater.js's job.

const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn, spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { startAutoUpdate } = require('./updater');

// A second launch should focus the existing window, not start a second
// backend fighting over the same sqlite state on a different port.
if (!app.requestSingleInstanceLock()) {
  app.quit();
}

let mainWindow = null;
let backendProcess = null;
let shuttingDown = false;

// ---------------------------------------------------------------- paths

function backendExePath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'backend', 'musictagger-backend.exe')
    : path.join(__dirname, '..', 'dist', 'musictagger-backend', 'musictagger-backend.exe');
}

function bundledToolsDir() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'tools')
    : path.join(__dirname, '..', 'build-resources', 'tools');
}

// The Python backend already knows how to find ffmpeg/ffprobe/fpcalc in
// "~/.musictagger/tools" (see Config.resolve_tool in musictag/config.py) -
// dropping the bundled binaries there means zero backend code has to know
// or care that it is running inside a packaged Electron app.
function userToolsDir() {
  return path.join(app.getPath('home'), '.musictagger', 'tools');
}

function installBundledTools() {
  const src = bundledToolsDir();
  const dest = userToolsDir();
  if (!fs.existsSync(src)) return;
  fs.mkdirSync(dest, { recursive: true });
  for (const name of fs.readdirSync(src)) {
    const destPath = path.join(dest, name);
    // Never overwrite a tool the user already has - they may have a newer
    // ffmpeg installed themselves, and Config.resolve_tool prefers whatever
    // is already in this folder over anything else.
    if (fs.existsSync(destPath)) continue;
    fs.copyFileSync(path.join(src, name), destPath);
  }
}

// -------------------------------------------------------------- backend

function waitForHealthy(port, { timeoutMs = 20000, intervalMs = 200 } = {}) {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const attempt = () => {
      const req = http.get({ host: '127.0.0.1', port, path: '/api/status', timeout: 1500 }, (res) => {
        res.resume();
        if (res.statusCode === 200) resolve();
        else retry();
      });
      req.on('error', retry);
      req.on('timeout', () => { req.destroy(); retry(); });
    };
    const retry = () => {
      if (Date.now() > deadline) reject(new Error('Backend did not become healthy in time'));
      else setTimeout(attempt, intervalMs);
    };
    attempt();
  });
}

function startBackend() {
  return new Promise((resolve, reject) => {
    const exe = backendExePath();
    if (!fs.existsSync(exe)) {
      reject(new Error(`Backend executable not found at:\n${exe}`));
      return;
    }

    backendProcess = spawn(exe, [], {
      windowsHide: true,
      // Tells the backend that updates install themselves here, so the UI
      // describes what will actually happen instead of linking to a download.
      env: { ...process.env, MUSICTAGGER_AUTO_UPDATE: app.isPackaged ? '1' : '0' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let resolved = false;
    let stdoutBuffer = '';
    let stderrBuffer = '';

    backendProcess.stdout.on('data', (chunk) => {
      stdoutBuffer += chunk.toString();
      const match = stdoutBuffer.match(/MUSICTAGGER_PORT=(\d+)/);
      if (match && !resolved) {
        resolved = true;
        resolve(parseInt(match[1], 10));
      }
    });
    backendProcess.stderr.on('data', (chunk) => {
      stderrBuffer += chunk.toString();
    });

    backendProcess.on('error', (err) => {
      if (!resolved) reject(err);
    });
    backendProcess.on('exit', (code) => {
      backendProcess = null;
      if (!resolved) {
        reject(new Error(`Backend exited before starting (code ${code}).\n${stderrBuffer.slice(-2000)}`));
        return;
      }
      // An unexpected exit after the app is up is a real crash, not a
      // deliberate shutdown - tell the user rather than leaving a dead window.
      if (!shuttingDown) {
        dialog.showErrorBox(
          'MusicTagger stopped unexpectedly',
          `The background service exited (code ${code}).\n\n${stderrBuffer.slice(-2000)}`
        );
        app.quit();
      }
    });
  });
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed && !shuttingDown) {
    shuttingDown = true;
    // taskkill /T also takes down anything the backend itself may have
    // spawned (ffmpeg/ffprobe subprocess calls mid-analysis), which a plain
    // process.kill() on Windows does not reliably do for a process tree.
    // Synchronous, because an update installer runs straight after quit and
    // cannot replace backend files that are still in use.
    if (process.platform === 'win32') {
      spawnSync('taskkill', ['/pid', String(backendProcess.pid), '/T', '/F'], { windowsHide: true });
    } else {
      backendProcess.kill();
    }
  }
}

// --------------------------------------------------------------- window

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    icon: path.join(__dirname, '..', 'build-resources', 'icon.ico'),
    autoHideMenuBar: true,
    backgroundColor: '#0d0d11',
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  // The app only ever talks to its own local backend; anything that tries to
  // navigate elsewhere (a stray external link) opens in the real browser
  // instead of inside this window.
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!url.startsWith('http://127.0.0.1:')) {
      event.preventDefault();
      shell.openExternal(url);
    }
  });

  mainWindow.on('closed', () => { mainWindow = null; });

  try {
    installBundledTools();
    const port = await startBackend();
    await waitForHealthy(port);
    await mainWindow.loadURL(`http://127.0.0.1:${port}/`);
    // Only an installed build has anywhere to update to; running from source
    // there is no installer to replace.
    if (app.isPackaged) {
      startAutoUpdate({ port, getWindow: () => mainWindow, beforeInstall: stopBackend });
    }
  } catch (err) {
    dialog.showErrorBox('MusicTagger could not start', String(err && err.message || err));
    app.quit();
  }
}

// ----------------------------------------------------------------- app

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(() => {
  app.setAppUserModelId('com.musictagger.app');
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  // Windows/Linux convention: no window means the app is done.
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopBackend();
});
