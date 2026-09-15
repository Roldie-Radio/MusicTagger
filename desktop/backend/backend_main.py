"""Entry point for the frozen backend PyInstaller packages for the Electron shell.

This is deliberately not ``musictag.cli`` - the CLI's job is to be a good
terminal citizen (argument parsing, pywebview fallback, browser-opening).
Here Electron already owns the window and the process lifecycle; all this
needs to do is bind a port, print it so the parent process can find it, and
serve the same FastAPI app the rest of the app already uses and tests.
"""

from __future__ import annotations

import socket
import sys


def _free_port(preferred: int = 8731) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return sock.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port available")


def main() -> int:
    import uvicorn
    from musictag.server import app

    port = _free_port()
    # Electron's main process reads this exact line from stdout to learn
    # which port to point the BrowserWindow at. Keep the format stable.
    print(f"MUSICTAGGER_PORT={port}", flush=True)

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
