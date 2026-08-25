"""
Recognition script: live webcam face recognition against embeddings in face_db.pkl.
Uses tiled high-res detection and tries 90/180/270 rotations when the camera is rolled.
"""

import argparse
import os
import sys

import cv2

from cctv_effects import apply_cctv_effects, draw_cctv_badge
from face_pipeline import (
    DB_PATH,
    LOW_RES_PX,
    OrientationTracker,
    embed_face,
    face_short_side,
    flatten_db,
    init_insightface,
    load_db,
    match_embedding,
    open_capture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live face recognition from webcam.")
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

    cap = open_capture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (device 0). Check camera permissions.")
        sys.exit(1)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {got_w}x{got_h}")
    print("Recognition running. Press 'q' to quit.")
    if args.cctv:
        print(f"CCTV simulation ON (strength={args.cctv_strength}).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            if args.cctv:
                frame = apply_cctv_effects(frame, strength=args.cctv_strength)

            faces = orientation.detect(app, frame)

            for face in faces:
                embedding = embed_face(app, frame, face)
                px = face_short_side(face)
                x1, y1, x2, y2 = face.bbox.astype(int)

                if embedding is None:
                    label = f"Unknown {px}px"
                    color = (0, 0, 255)
                else:
                    name, score, known = match_embedding(
                        embedding, known_names, known_embeddings
                    )
                    if known:
                        label = f"{name} ({score:.2f}) {px}px"
                        color = (0, 255, 0)
                    else:
                        label = f"Unknown ({score:.2f}) {px}px"
                        color = (0, 0, 255)

                if px < LOW_RES_PX:
                    label = f"{label} LOW-RES"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
                cv2.putText(
                    frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if args.cctv:
                draw_cctv_badge(frame, strength=args.cctv_strength)

            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Recognition stopped by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
