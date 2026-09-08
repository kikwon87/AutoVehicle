"""The test platform: a local web application.

Python simulates, the browser draws and edits.  The split is not cosmetic --
the simulation is numpy and belongs in Python, while sliders, a canvas and a
results table belong in a browser, and gluing them with a small JSON API costs
less than either a desktop toolkit or a headless CLI.

The server uses only the standard library, so the platform adds no dependency
to a package whose core is numpy alone.

The API, in the order a session uses it::

    GET  /api/bootstrap        parameter specs, presets, controllers, scoring
    POST /api/controller       validate a .py plug-in, report what it provides
    POST /api/run              start a run; returns a run id
    GET  /api/stream?id&from   frames since an index, plus status
    GET  /api/scene?id         static geometry for the canvas (sent once)
    POST /api/stop             stop a running simulation
    POST /api/batch            run a matrix of presets x controllers
    POST /api/rescore          re-score stored runs with new weights
    GET  /api/results          every completed run in this server session
    GET  /api/report           the controller contract, as markdown

``/api/rescore`` matters more than it looks.  Weights and thresholds are a
*reporting* choice, not a simulation input, so changing them re-scores stored
metrics instantly.  Re-running the simulation to answer "what if safety
mattered three times as much?" would make the question too expensive to ask,
and it is the question the platform exists for.
"""

from __future__ import annotations

import json
import math
import mimetypes
import threading
import traceback
import uuid
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..models.params import REFERENCE_VEHICLE
from .controller_api import CONTRACT_VERSION, ControllerLoadError, load_controller
from .controllers import BUILTIN_CONTROLLERS
from .parameters import (
    PARAMETER_GROUPS,
    build_vehicle_params,
    default_values,
    derived_summary,
    specs_json,
)
from .presets import PRESETS, preset_summaries
from .scoring import CATEGORY_LABELS, ScoreConfig, compare, score_run
from .session import RunConfig, RunSession

STATIC = Path(__file__).parent / "static"
DOCS = Path(__file__).resolve().parents[2] / "docs" / "controller_api.md"

BUILTIN_LABELS = {
    "mpc": "MPC — built-in stack (내장 MPC)",
    "pure_pursuit": "Pure pursuit + PI (기하 기반)",
    "linear_policy": "Linear policy — ML-shaped (학습형 인터페이스)",
}


class Run:
    """One simulation, executing on its own thread."""

    def __init__(self, config: RunConfig, score_config: ScoreConfig):
        self.id = uuid.uuid4().hex[:12]
        self.config = config
        self.session = RunSession(config, score_config)
        self.status = "ready"
        self.error = ""
        self.result = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.lock = threading.Lock()

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            while not self.session.finished and not self._stop.is_set():
                with self.lock:
                    self.session.step()
            self.result = self.session.result()
            self.status = "stopped" if self._stop.is_set() else "finished"
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not swallowed
            self.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
            self.status = "error"

    def stop(self) -> None:
        self._stop.set()

    def frames(self, since: int) -> list[dict]:
        with self.lock:
            frames = self.session.frames[since:]
        return [asdict(f) for f in frames]


class Platform:
    """Server-side state: the runs, the scoring policy, the loaded plug-ins."""

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.score_config = ScoreConfig()
        self.plugins: dict[str, str] = {}   # label -> path
        self.results: list[dict] = []

    # --- controllers ---------------------------------------------------------

    def controller_choices(self) -> list[dict]:
        out = [
            {"value": k, "label": BUILTIN_LABELS.get(k, k), "kind": "builtin"}
            for k in BUILTIN_CONTROLLERS
        ]
        out += [
            {"value": path, "label": f"{label}  ({Path(path).name})", "kind": "plugin"}
            for label, path in self.plugins.items()
        ]
        return out

    def register_plugin(self, path: str) -> dict:
        controller = load_controller(path)         # raises ControllerLoadError
        label = getattr(controller, "name", Path(path).stem)
        self.plugins[label] = str(Path(path).expanduser().resolve())
        return {
            "label": label,
            "path": self.plugins[label],
            "description": getattr(controller, "description", ""),
        }

    # --- runs ------------------------------------------------------------------

    def start_run(self, payload: dict) -> Run:
        config = RunConfig(
            preset=payload.get("preset", "grid_random"),
            controller=payload.get("controller", "mpc"),
            controller_options=payload.get("controller_options") or {},
            parameters=payload.get("parameters") or {},
            seed=int(payload.get("seed", 0)),
            n_vehicles=(
                int(payload["n_vehicles"]) if payload.get("n_vehicles") not in (None, "") else None
            ),
            duration=(
                float(payload["duration"]) if payload.get("duration") not in (None, "") else None
            ),
        )
        run = Run(config, self.score_config)
        self.runs[run.id] = run
        run.start()
        return run

    def record(self, run: Run) -> dict | None:
        if run.result is None:
            return None
        entry = run.result.to_dict()
        entry["id"] = run.id
        entry["label"] = f"{run.result.preset} / {run.result.controller}"
        if not any(r["id"] == run.id for r in self.results):
            self.results.append(entry)
        return entry

    def rescore(self, config: ScoreConfig) -> list[dict]:
        self.score_config = config
        for entry in self.results:
            entry["score"] = score_run(entry["metrics"], config).to_dict()
        return self.results


PLATFORM = Platform()


class Handler(BaseHTTPRequestHandler):
    server_version = "avsim-platform"

    # --- plumbing ---------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - quieter console
        pass

    def _json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(_finite(payload), default=_encode).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def _static(self, path: str) -> None:
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC / name).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            self.send_error(404)
            return
        data = target.read_bytes()
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- routes -------------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        url = urlparse(self.path)
        q = parse_qs(url.query)
        try:
            if url.path == "/api/bootstrap":
                return self._json(self._bootstrap())
            if url.path == "/api/scene":
                run = PLATFORM.runs[q["id"][0]]
                return self._json(run.session.static_scene())
            if url.path == "/api/stream":
                return self._json(self._stream(q))
            if url.path == "/api/results":
                return self._json({"results": PLATFORM.results,
                                   "ranking": _ranking(PLATFORM.results)})
            if url.path == "/api/report":
                text = DOCS.read_text(encoding="utf-8") if DOCS.is_file() else \
                    "docs/controller_api.md not found next to the package."
                return self._json({"markdown": text, "version": CONTRACT_VERSION})
            return self._static(url.path)
        except KeyError as exc:
            return self._json({"error": f"unknown id {exc}"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc), "trace": traceback.format_exc(limit=4)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/run":
                run = PLATFORM.start_run(body)
                return self._json({"id": run.id, "status": run.status})
            if url.path == "/api/stop":
                PLATFORM.runs[body["id"]].stop()
                return self._json({"ok": True})
            if url.path == "/api/controller":
                try:
                    return self._json(PLATFORM.register_plugin(body["path"]))
                except ControllerLoadError as exc:
                    return self._json({"error": str(exc)}, 400)
            if url.path == "/api/rescore":
                cfg = ScoreConfig.from_dict(body.get("score"))
                results = PLATFORM.rescore(cfg)
                return self._json({"results": results, "ranking": _ranking(results),
                                   "score": cfg.to_dict()})
            if url.path == "/api/derived":
                params = build_vehicle_params(body.get("parameters") or {})
                return self._json(derived_summary(params))
            if url.path == "/api/batch":
                return self._json(self._batch(body))
            return self._json({"error": f"no route {url.path}"}, 404)
        except Exception as exc:  # noqa: BLE001
            return self._json({"error": str(exc), "trace": traceback.format_exc(limit=4)}, 500)

    # --- handlers ---------------------------------------------------------------------

    def _bootstrap(self) -> dict:
        return {
            "parameters": specs_json(),
            "groups": [{"key": k, "label": v} for k, v in PARAMETER_GROUPS],
            "defaults": default_values(),
            "presets": preset_summaries(),
            "controllers": PLATFORM.controller_choices(),
            "score": PLATFORM.score_config.to_dict(),
            "categories": CATEGORY_LABELS,
            "derived": derived_summary(REFERENCE_VEHICLE),
            "contract_version": CONTRACT_VERSION,
        }

    def _stream(self, q: dict) -> dict:
        run = PLATFORM.runs[q["id"][0]]
        since = int(q.get("from", ["0"])[0])
        payload = {
            "status": run.status,
            "error": run.error,
            "frames": run.frames(since),
            "count": len(run.session.frames),
        }
        if run.status in ("finished", "stopped") and run.result is not None:
            payload["result"] = PLATFORM.record(run)
            payload["ranking"] = _ranking(PLATFORM.results)
        return payload

    def _batch(self, body: dict) -> dict:
        """Run every (preset, controller) pair requested, to completion.

        Synchronous on purpose: a batch is a measurement, and streaming its
        frames would only invite watching it instead of reading the table.
        """
        presets = body.get("presets") or ["grid_random"]
        controllers = body.get("controllers") or ["mpc"]
        seeds = body.get("seeds") or [int(body.get("seed", 0))]
        entries = []
        for preset in presets:
            for controller in controllers:
                for seed in seeds:
                    cfg = RunConfig(
                        preset=preset, controller=controller,
                        parameters=body.get("parameters") or {},
                        seed=int(seed),
                        n_vehicles=(int(body["n_vehicles"])
                                    if body.get("n_vehicles") not in (None, "") else None),
                        duration=(float(body["duration"])
                                  if body.get("duration") not in (None, "") else None),
                    )
                    try:
                        session = RunSession(cfg, PLATFORM.score_config)
                        result = session.run()
                        entry = result.to_dict()
                        entry["id"] = uuid.uuid4().hex[:12]
                        entry["label"] = f"{preset} / {result.controller} / seed {seed}"
                        PLATFORM.results.append(entry)
                        entries.append(entry)
                    except Exception as exc:  # noqa: BLE001 - one failure is a result
                        entries.append({
                            "id": uuid.uuid4().hex[:12],
                            "label": f"{preset} / {controller} / seed {seed}",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
        return {"results": entries, "all": PLATFORM.results, "ranking": _ranking(PLATFORM.results)}


def _ranking(results: list[dict]) -> list[dict]:
    scored = [
        (r.get("label", r.get("id", "?")), _Box(r["score"]))
        for r in results if "score" in r
    ]
    return compare(scored)


class _Box:
    """Adapter so :func:`compare` can rank plain dictionaries."""

    def __init__(self, d: dict):
        self.total = float(d.get("total", 0.0))
        self.categories = d.get("categories", {})
        self.collided = bool(d.get("collided", False))


def _finite(obj: Any):
    """Replace every non-finite float in a payload with ``null``.

    ``json.dumps`` writes ``Infinity`` for ``float('inf')``, which is valid
    Python and **invalid JSON**: ``JSON.parse`` rejects it and the whole page
    fails to start.  It is not a hypothetical -- an understeering car has an
    infinite critical speed, so the very first ``/api/bootstrap`` carries one.

    Infinity is meaningful here (no vehicle was ever near, the goal was never
    reached), so it is sent as ``null`` and the client decides what that means
    per field, rather than being clamped to a number that would read as data.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _finite(obj.tolist())
    return obj


def _encode(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if v == v and abs(v) != float("inf") else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if o == float("inf") or o == float("-inf"):
        return None
    return str(o)


def serve(host: str = "127.0.0.1", port: int = 8770, open_browser: bool = True) -> None:
    """Start the platform and block."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"avsim test platform on {url}")
    print("  Ctrl-C to stop")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # pragma: no cover - headless machines
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
