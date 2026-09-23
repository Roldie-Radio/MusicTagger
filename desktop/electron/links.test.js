// Run with: node --test desktop/electron/links.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { isSafeExternalUrl, isBackendUrl } = require('./links');

test('only https links are handed to the OS', () => {
  assert.ok(isSafeExternalUrl('https://github.com/Roldie-Radio/MusicTagger/releases'));
  for (const url of ['http://example.com', 'file:///C:/Windows/System32/calc.exe',
    'smb://host/share', 'ms-settings:privacy', 'javascript:alert(1)', 'not a url', '']) {
    assert.strictEqual(isSafeExternalUrl(url), false, url);
  }
});

test('the window may only navigate to the backend origin itself', () => {
  const origin = 'http://127.0.0.1:8731';
  assert.ok(isBackendUrl('http://127.0.0.1:8731/', origin));
  assert.ok(isBackendUrl('http://127.0.0.1:8731/static/app.js?x=1', origin));
  assert.strictEqual(isBackendUrl('http://127.0.0.1:8731@evil.example/', origin), false);
  assert.strictEqual(isBackendUrl('http://127.0.0.1:9999/', origin), false);
  assert.strictEqual(isBackendUrl('https://127.0.0.1:8731/', origin), false);
  assert.strictEqual(isBackendUrl('http://127.0.0.1:8731/', null), false);
});

test('every local module main.js loads ships in the installer', () => {
  // electron-builder packages only what "files" lists; a require()d file
  // missing from it works from source and crashes the installed app on start.
  const dir = __dirname;
  const pkg = JSON.parse(fs.readFileSync(path.join(dir, 'package.json'), 'utf8'));
  const shipped = new Set(pkg.build.files);
  const seen = new Set();
  const visit = (file) => {
    if (seen.has(file)) return;
    seen.add(file);
    const source = fs.readFileSync(path.join(dir, file), 'utf8');
    for (const [, spec] of source.matchAll(/require\(\s*['"](\.\/[^'"]+)['"]\s*\)/g)) {
      const name = spec.replace(/^\.\//, '').replace(/(\.js)?$/, '.js');
      assert.ok(shipped.has(name), `${file} requires ${spec}, which package.json "files" leaves out`);
      visit(name);
    }
  };
  visit(pkg.main || 'main.js');
});
