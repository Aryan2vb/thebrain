"""
Enrollment script: capture 5 facial angles for a person and store embeddings
in face_db.pkl for later recognition.
"""

import argparse
import sys

import cv2
import numpy as np

from cctv_effects import apply_cctv_effects, draw_cctv_badge
from face_pipeline import DB_PATH, detect_faces, init_insightface, load_db, open_capture, save_db

POSES = ["Front", "Up", "Down", "Left", "Right"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll a person into the face database.")
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
    name = input("Enter the name of the person to enroll: ").strip()
    if not name:
        print("Error: name cannot be empty.")
        sys.exit(1)

    print("Initializing InsightFace (buffalo_l)...")
    app = init_insightface()

    cap = open_capture(0)
    if not cap.isOpened():
        print("Error: could not open webcam (device 0). Check camera permissions.")
        sys.exit(1)

    got_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    got_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {got_w}x{got_h}")

    embeddings: list[np.ndarray] = []
    pose_index = 0

    print("Enrollment started. Press 'c' to capture each pose, 'q' to quit.")
    print(f"Poses to capture: {', '.join(POSES)}")
    if args.cctv:
        print(f"CCTV simulation ON (strength={args.cctv_strength}).")

    try:
        while pose_index < len(POSES):
            ret, frame = cap.read()
            if not ret:
                print("Error: failed to read frame from webcam.")
                break

            if args.cctv:
                frame = apply_cctv_effects(frame, strength=args.cctv_strength)

            pose = POSES[pose_index]
            cv2.putText(
                frame,
                f"Look: {pose}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "Press 'c' to capture",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Progress: {pose_index}/{len(POSES)}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if args.cctv:
                draw_cctv_badge(frame, strength=args.cctv_strength)

            cv2.imshow("Enrollment", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Enrollment cancelled by user.")
                break

            if key == ord("c"):
                # Close-range enrollment; rotation-aware so a rolled webcam still works
                faces = detect_faces(app, frame)
                if len(faces) != 1:
                    print(
                        f"Warning: expected exactly 1 face, found {len(faces)}. "
                        "Please adjust and try again."
                    )
                    continue

                emb = faces[0].normed_embedding.copy()
                embeddings.append(emb)
                print(f"Captured '{pose}' for {name} ({pose_index + 1}/{len(POSES)}).")
                pose_index += 1

        if pose_index == len(POSES):
            db = load_db()
            db[name] = embeddings
            save_db(db)
            print(f"Successfully enrolled '{name}' with {len(embeddings)} embeddings.")
            print(f"Database saved to {DB_PATH}.")
        else:
            print("Incomplete enrollment; database was not updated.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
