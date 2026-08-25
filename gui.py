"""Local GUI: enroll named portraits, then identify people in an uploaded photo."""

from __future__ import annotations

import base64
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask.json.provider import DefaultJSONProvider

from face_pipeline import (
    bgr_from_bytes,
    enroll_from_image,
    identify_frame,
    init_insightface,
    load_db,
)

ROOT = Path(__file__).resolve().parent


class NumpyJSONProvider(DefaultJSONProvider):
    """Flask 3 cannot encode NumPy scalars; coerce them to plain Python types."""

    def default(self, o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


web = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
web.json = NumpyJSONProvider(web)

_lock = threading.Lock()
_face_app = None
_init_error: str | None = None


def _boot_models() -> None:
    global _face_app, _init_error
    try:
        app = init_insightface()
        with _lock:
            _face_app = app
    except Exception as exc:
        with _lock:
            _init_error = str(exc)


def get_face_app():
    with _lock:
        if _init_error:
            raise RuntimeError(_init_error)
        if _face_app is None:
            raise RuntimeError("Models are still loading. Try again in a moment.")
        return _face_app


@web.get("/")
def index():
    return render_template("index.html")


@web.get("/favicon.ico")
def favicon():
    return send_from_directory(web.static_folder, "favicon.svg", mimetype="image/svg+xml")


@web.get("/api/status")
def status():
    db = load_db()
    with _lock:
        ready = _face_app is not None
        error = _init_error
    return jsonify(
        {
            "ready": ready,
            "error": error,
            "people": sorted(db.keys()),
        }
    )


@web.post("/api/enroll")
def enroll():
    name = (request.form.get("name") or "").strip()
    upload = request.files.get("image")
    if not name:
        return jsonify({"ok": False, "error": "Enter a name."}), 400
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Choose a front-face photo."}), 400
    try:
        frame = bgr_from_bytes(upload.read())
        ok, message = enroll_from_image(get_face_app(), name, frame)
        db = load_db()
        code = 200 if ok else 400
        return jsonify({"ok": ok, "message": message, "people": sorted(db.keys())}), code
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


@web.post("/api/identify")
def identify():
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Choose a photo to identify."}), 400
    try:
        frame = bgr_from_bytes(upload.read())
        db = load_db()
        if not db:
            return jsonify({"ok": False, "error": "Enroll someone first."}), 400
        t0 = time.perf_counter()
        annotated, labels = identify_frame(get_face_app(), frame, db)
        elapsed = time.perf_counter() - t0
        print(
            f"Identify: {len(labels)} faces in {elapsed:.1f}s | "
            f"named={[f['name'] for f in labels if f['known']]} | "
            f"best={max((f['score'] for f in labels), default=0):.2f}"
        )
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return jsonify({"ok": False, "error": "Could not encode the result."}), 500
        return jsonify(
            {
                "ok": True,
                "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
                "faces": labels,
                "seconds": round(elapsed, 1),
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500


def main() -> None:
    threading.Thread(target=_boot_models, daemon=True).start()
    print("THEBRAIN GUI → http://127.0.0.1:5050")
    web.run(host="127.0.0.1", port=5050, debug=False, threaded=True)


if __name__ == "__main__":
    main()
