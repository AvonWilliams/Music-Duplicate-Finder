# Music Duplicate Finder — server_client.py
# Stdlib-only HTTP client for the FastAPI inference server.
# No third-party dependencies (requests etc.) to keep Picard compatibility.
# V1.2: X-API-Key header support.

import json
import socket
import time
import urllib.error
import urllib.request
from typing import Any

from .diag import get_logger

_log = get_logger("server_client")


class ServerClient:
    """Thin wrapper around urllib for the CLAP/FAISS inference server."""

    # ── Header builder ─────────────────────────────────────────────────────

    @staticmethod
    def _headers(api_key: str, with_json: bool = False) -> dict[str, str]:
        h: dict[str, str] = {}
        if api_key:
            h["X-API-Key"] = api_key
        if with_json:
            h["Content-Type"] = "application/json"
        return h

    # ── Public class helpers ───────────────────────────────────────────────

    @staticmethod
    def ping(
        host: str,
        port: int,
        api_key: str = "",
        timeout: int = 5,
    ) -> tuple[bool, str]:
        """
        Return (ok, message) for a /health check.
        /health is typically unauthenticated on the server, but we include
        X-API-Key anyway so the same key can be validated end-to-end.
        """
        url = f"http://{host}:{port}/health"
        _log.info("PING → %s  (api_key=%s)", url, "set" if api_key else "empty")
        try:
            req = urllib.request.Request(
                url, headers=ServerClient._headers(api_key), method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                gpu   = data.get("gpu",   "unknown")
                model = data.get("model", "unknown")
                _log.info("PING ← HTTP %s  gpu=%s  model=%s", resp.status, gpu, model)
                return True, f"Connected — GPU: {gpu}  Model: {model}"
        except urllib.error.HTTPError as exc:
            _log.warning("PING ← HTTP %s: %s", exc.code, exc.reason)
            if exc.code in (401, 403):
                return False, f"Authentication failed (HTTP {exc.code}) — check API key"
            return False, f"HTTP {exc.code}: {exc.reason}"
        except urllib.error.URLError as exc:
            _log.warning("PING ← connection failed: %s", exc.reason)
            return False, str(exc.reason)
        except Exception as exc:  # noqa: BLE001
            _log.exception("PING ← unexpected error")
            return False, str(exc)

    @staticmethod
    def scan(
        host: str,
        port: int,
        lxc_paths: list[str],
        certain_threshold: float,
        likely_threshold: float,
        unsure_threshold: float,
        api_key: str = "",
        timeout: int = 300,
    ) -> dict[str, Any]:
        """
        POST /scan to the inference server.

        Response shape:
        {
            "groups": [
                {"confidence": "certain"|"likely"|"unsure",
                 "similarity": 0.98,
                 "files": ["/mnt/music/a.flac", "/mnt/music/b.mp3"]}
            ],
            "scanned": 412,
            "elapsed_seconds": 14.3
        }
        """
        url = f"http://{host}:{port}/scan"
        payload = json.dumps({
            "files":              lxc_paths,
            "certain_threshold":  certain_threshold,
            "likely_threshold":   likely_threshold,
            "unsure_threshold":   unsure_threshold,
        }).encode()

        _log.info(
            "SCAN → %s  files=%d  thresholds=(certain=%.3f likely=%.3f unsure=%.3f)  "
            "payload=%d bytes  api_key=%s",
            url, len(lxc_paths), certain_threshold, likely_threshold,
            unsure_threshold, len(payload), "set" if api_key else "empty",
        )
        for i, p in enumerate(lxc_paths[:3]):
            _log.info("  sending path [%d]: %r", i, p)
        if len(lxc_paths) > 3:
            _log.info("  … and %d more paths", len(lxc_paths) - 3)

        req = urllib.request.Request(
            url,
            data    = payload,
            headers = ServerClient._headers(api_key, with_json=True),
            method  = "POST",
        )
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode()
                data = json.loads(body)
                dt   = time.time() - t0
                n_groups = len(data.get("groups", []))
                scanned  = data.get("scanned", "?")
                srv_el   = data.get("elapsed_seconds", "?")
                _log.info(
                    "SCAN ← HTTP %s  groups=%d  scanned=%s  "
                    "server_elapsed=%ss  network_elapsed=%.2fs  body=%d bytes",
                    resp.status, n_groups, scanned, srv_el, dt, len(body),
                )
                # Sample the first group so we can see confidence/similarity
                for i, g in enumerate(data.get("groups", [])[:3]):
                    _log.info(
                        "  group[%d]: confidence=%s similarity=%.4f files=%d",
                        i, g.get("confidence"), float(g.get("similarity", 0)),
                        len(g.get("files", [])),
                    )
                    for j, fp in enumerate(g.get("files", [])[:3]):
                        _log.info("      file[%d]: %r", j, fp)
                return data
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            _log.error("SCAN ← HTTP %s: %s  body=%s", exc.code, exc.reason, body[:500])
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"Authentication failed (HTTP {exc.code}).  "
                    "Check your API key in Options → Plugins → "
                    "Music Duplicate Finder → Remote GPU Server."
                ) from exc
            raise RuntimeError(f"Server error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            _log.error("SCAN ← connection failed: %s", exc.reason)
            raise RuntimeError(f"Connection failed: {exc.reason}") from exc

    @staticmethod
    def _resolve_host(host: str) -> str:
        try:
            return socket.gethostbyname(host)
        except socket.gaierror:
            return host
