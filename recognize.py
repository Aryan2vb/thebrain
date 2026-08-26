"""
Recognition script: live face recognition against embeddings in face_db.pkl.
Defaults to the RTSP camera feed; uses tiled high-res detection and tries
90/180/270 rotations when the camera is rolled.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from cctv_effects import apply_cctv_effects, draw_cctv_badge
from face_pipeline import (
    DB_PATH,
    LOW_RES_PX,
    OrientationTracker,
    face_short_side,
    flatten_db,
    init_insightface,
    load_db,
    match_embedding,
    open_capture,
)

DEFAULT_SOURCE = "rtsp://10.7.7.60:8554/mystream"


class LatestFrame:
    """Keep only the newest decoded frame so a slow detector cannot back up the stream."""

    def __init__(self, source: int | str) -> None:
        self.source = source
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = True
        self.cap = open_capture(source)
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> bool:
        if not self.cap.isOpened():
            return False
        self.thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        self.thread.join(timeout=2.0)
        self.cap.release()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _loop(self) -> None:
        misses = 0
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                misses += 1
                if misses >= 60:
                    print("Reconnecting to video source...")
                    self.cap.release()
                    self.cap = open_capture(self.source)
                    misses = 0
                time.sleep(0.01)
                continue
            misses = 0
            with self._lock:
                self._frame = frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live face recognition from an RTSP stream or camera.")
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="RTSP/HTTP URL or camera index (default: rtsp://10.7.7.60:8554/mystream).",
    )
    parser.add_argument(
        "--cctv",
        action="store_true",
        help="Simulate noisy CCTV video (blur, compression, low light, sensor noise).",
    )
    parser.add_argument(
        "--cctv-strength",
        type=float,
        default=1.0,
        help="CCTV degradation intensity (0=off, 1=default, 2=very harsh). Default: 1.0",
    )
    return parser.parse_args()


def _draw_labels(frame: np.ndarray, labels: list[dict]) -> None:
    for item in labels:
        x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(
            frame,
            item["text"],
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    if not os.path.exists(DB_PATH):
        print(
            f"Error: '{DB_PATH}' not found. "
            "Run enrollment.py or python gui.py first to register at least one person."
        )
        sys.exit(1)

    print(f"Loading face database from {DB_PATH}...")
    known_names, known_embeddings = flatten_db(load_db())
    if not known_names:
        print(f"Error: {DB_PATH} contains no embeddings. Enroll someone first.")
        sys.exit(1)
    print(f"Loaded {len(known_names)} embeddings for {len(set(known_names))} people.")

    print("Initializing InsightFace (buffalo_l)...")
    app = init_insightface()
    orientation = OrientationTracker()

    source = args.source if not str(args.source).isdigit() else int(args.source)
    print(f"Opening video source: {source}")
    stream = LatestFrame(source)
    if not stream.start():
        print(f"Error: could not open video source ({source}). Check the RTSP URL or camera.")
        sys.exit(1)

    first = None
    for _ in range(100):
        first = stream.latest()
        if first is not None:
            break
        time.sleep(0.05)
    if first is None:
        print(f"Error: no frames from ({source}). Check that the RTSP stream is publishing.")
        stream.stop()
        sys.exit(1)

    print(f"Stream resolution: {first.shape[1]}x{first.shape[0]}")
    print("Recognition running. Press 'q' to quit.")
    if args.cctv:
        print(f"CCTV simulation ON (strength={args.cctv_strength}).")

    labels_lock = threading.Lock()
    labels: list[dict] = []
    recognize_stop = threading.Event()

    def recognize_loop() -> None:
        nonlocal labels
        while not recognize_stop.is_set():
            frame = stream.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            if args.cctv:
                frame = apply_cctv_effects(frame, strength=args.cctv_strength)
            faces = orientation.detect_live(app, frame)
            found: list[dict] = []
            for face in faces:
                embedding = None
                if face.normed_embedding is not None:
                    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
                if embedding is None:
                    continue
                name, score, known = match_embedding(
                    embedding, known_names, known_embeddings
                )
                if not known:
                    continue
                px = face_short_side(face)
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                text = f"{name} ({score:.2f}) {px}px"
                if px < LOW_RES_PX:
                    text = f"{text} LOW-RES"
                found.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": text})
            with labels_lock:
                labels = found

    worker = threading.Thread(target=recognize_loop, daemon=True)
    worker.start()

    try:
        while True:
            frame = stream.latest()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Recognition stopped by user.")
                    break
                continue
            if args.cctv:
                frame = apply_cctv_effects(frame, strength=args.cctv_strength)
            with labels_lock:
                current = list(labels)
            _draw_labels(frame, current)
            if args.cctv:
                draw_cctv_badge(frame, strength=args.cctv_strength)
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recognition stopped by user.")
                break
    finally:
        recognize_stop.set()
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
