// Which URLs the window may load, and which may be handed to the OS.
//
// Kept free of any Electron import so it can be unit-tested with plain Node
// (links.test.js) - these two rules are the whole of the shell's defence
// against a page steering it somewhere it should not go.

// Only https leaves the app. shell.openExternal passes the URL to the OS,
// which will happily launch whatever handles file:, smb:, ms-settings: or any
// other registered scheme; every link the app itself produces (releases,
// MusicBrainz, AcoustID, ffmpeg downloads) is https.
function isSafeExternalUrl(url) {
  try {
    return new URL(url).protocol === 'https:';
  } catch {
    return false;
  }
}

// Navigation inside the window is allowed only to the backend's exact origin.
// A prefix test is not enough: "http://127.0.0.1:8731@evil.example/" starts
// with "http://127.0.0.1:" but its host is evil.example - everything before
// the "@" is a username.
function isBackendUrl(url, backendOrigin) {
  if (!backendOrigin) return false;
  try {
    return new URL(url).origin === backendOrigin;
  } catch {
    return false;
  }
}

module.exports = { isSafeExternalUrl, isBackendUrl };
