// The one door between the web UI and the desktop shell.
//
// The window stays sandboxed with no Node access; this exposes exactly the
// update controls and nothing else. The UI checks for `musictaggerDesktop`
// and falls back to the browser behaviour when it is absent, so the same page
// works when the backend is run from source and opened in a normal browser.

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('musictaggerDesktop', {
  updates: {
    getState: () => ipcRenderer.invoke('updates:get-state'),
    // Check now and download if there is something newer - ignores the
    // automatic-check setting, because the user asked.
    check: () => ipcRenderer.invoke('updates:check'),
    // Quit, install the downloaded update and relaunch.
    install: () => ipcRenderer.invoke('updates:install'),
    onState: (callback) => {
      const listener = (_event, state) => callback(state);
      ipcRenderer.on('updates:state', listener);
      return () => ipcRenderer.removeListener('updates:state', listener);
    },
  },
});
