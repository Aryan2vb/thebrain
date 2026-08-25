"""
Long-range face detection helpers: high-res InsightFace init, tiled inference, and
small-crop upscaling before ArcFace embedding.
"""

from __future__ import annotations

import io
import os
import pickle

import cv2
import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils.face_align import norm_crop
from PIL import Image, ImageDraw, ImageFont, ImageOps

DET_SIZE = (1280, 1280)
DET_THRESH = 0.3
NMS_IOU = 0.4
TILE_OVERLAP = 0.25
# ArcFace is trained on ~112px aligned faces; below this, upsample before embed.
MIN_EMBED_PX = 80
TARGET_SHORT_PX = 112
CROP_PAD = 0.20
# Overlay hint: identity is unreliable when the box is this small.
LOW_RES_PX = 40
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
DB_PATH = "face_db.pkl"
SIMILARITY_THRESHOLD = 0.45
# Group stills are smaller/noisier than enrollment portraits.
STILL_MATCH_THRESHOLD = 0.38
ORIENTATIONS = (0, 90, 180, 270)
# GUI stills: cap longest side so we run one detector pass, not 4 tiles × 4 rotations.
STILL_MAX_SIDE = 1280
# Skip ArcFace only when the ORIGINAL crop is tiny (crowd background).
MIN_IDENTIFY_PX = 40


def _onnx_providers() -> list[str]:
    # CoreML cannot run buffalo_l SCRFD (static vs inferred output rank mismatch).
    available = onnxruntime.get_available_providers()
    preferred = []
    for name in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if name in available:
            preferred.append(name)
    return preferred or ["CPUExecutionProvider"]


def init_insightface() -> FaceAnalysis:
    """buffalo_l detection + recognition only (skip gender/106-pt — they dominate group-photo time)."""
    providers = _onnx_providers()
    print(f"ONNX providers: {providers}")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    try:
        app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=DET_THRESH)
        print(f"InsightFace initialized (ctx_id=0), det_size={DET_SIZE}, det_thresh={DET_THRESH}.")
    except Exception as exc:
        print(f"GPU prepare note ({exc}); using CPU ctx.")
        app.prepare(ctx_id=-1, det_size=DET_SIZE, det_thresh=DET_THRESH)
        print(f"InsightFace initialized (ctx_id=-1), det_size={DET_SIZE}, det_thresh={DET_THRESH}.")
    return app


def open_capture(index: int = 0, width: int = CAPTURE_WIDTH, height: int = CAPTURE_HEIGHT) -> cv2.VideoCapture:
    """Open a camera and request 1080p so distant faces keep more pixels."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    return cap


def face_short_side(face: Face) -> int:
    """Return min(width, height) of the face bounding box in pixels."""
    x1, y1, x2, y2 = face.bbox.astype(float)
    return int(max(0, min(x2 - x1, y2 - y1)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _nms(faces: list[Face], iou_thresh: float = NMS_IOU) -> list[Face]:
    """Keep highest-score boxes; drop overlaps above iou_thresh."""
    ordered = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
    kept: list[Face] = []
    for face in ordered:
        if all(_iou(face.bbox, other.bbox) < iou_thresh for other in kept):
            kept.append(face)
    return kept


def _offset_face(face: Face, ox: int, oy: int) -> Face:
    """Translate bbox and landmarks from tile coordinates to full-frame coordinates."""
    bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    bbox[0] += ox
    bbox[1] += oy
    bbox[2] += ox
    bbox[3] += oy
    kps = None
    if face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32).copy()
        kps[:, 0] += ox
        kps[:, 1] += oy
    shifted = Face(bbox=bbox, kps=kps, det_score=face.det_score)
    if face.embedding is not None:
        shifted.embedding = np.asarray(face.embedding, dtype=np.float32).copy()
    return shifted


def _tile_origins(length: int, n: int = 2, overlap: float = TILE_OVERLAP) -> tuple[list[int], int]:
    """Origins and tile length so n windows with the given overlap cover [0, length)."""
    denom = n - (n - 1) * overlap
    tile = min(length, int(np.ceil(length / denom)))
    step = max(1, int(tile * (1.0 - overlap)))
    origins: list[int] = []
    for i in range(n):
        origin = min(i * step, max(0, length - tile))
        if origin not in origins:
            origins.append(origin)
    return origins, tile


def detect_faces_tiled(app: FaceAnalysis, frame: np.ndarray, overlap: float = TILE_OVERLAP) -> list[Face]:
    """
    Detect faces on the full frame, or on a 2x2 overlapping grid when the frame
    is larger than det_size so distant faces are not downsampled away.
    """
    h, w = frame.shape[:2]
    if max(h, w) <= DET_SIZE[0]:
        return app.get(frame)

    x_origins, tile_w = _tile_origins(w, n=2, overlap=overlap)
    y_origins, tile_h = _tile_origins(h, n=2, overlap=overlap)

    detected: list[Face] = []
    for oy in y_origins:
        for ox in x_origins:
            x2 = min(w, ox + tile_w)
            y2 = min(h, oy + tile_h)
            tile = frame[oy:y2, ox:x2]
            if tile.size == 0:
                continue
            for face in app.get(tile):
                detected.append(_offset_face(face, ox, oy))

    return _nms(detected)


def _rotate_bgr(frame: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return frame
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation angle: {angle}")


def _map_xy(x: float, y: float, orig_w: int, orig_h: int, angle: int) -> tuple[float, float]:
    """Map a point from a rotated image back into original (w, h) coordinates."""
    if angle == 0:
        return x, y
    if angle == 90:
        return y, orig_h - 1 - x
    if angle == 180:
        return orig_w - 1 - x, orig_h - 1 - y
    if angle == 270:
        return orig_w - 1 - y, x
    raise ValueError(f"Unsupported rotation angle: {angle}")


def _map_face_from_rotated(face: Face, orig_w: int, orig_h: int, angle: int) -> Face:
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    mapped = [_map_xy(px, py, orig_w, orig_h, angle) for px, py in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    bbox = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
    kps = None
    if face.kps is not None:
        kps = np.array(
            [_map_xy(float(px), float(py), orig_w, orig_h, angle) for px, py in face.kps],
            dtype=np.float32,
        )
    shifted = Face(bbox=bbox, kps=kps, det_score=face.det_score)
    if face.embedding is not None:
        shifted.embedding = np.asarray(face.embedding, dtype=np.float32).copy()
    return shifted


def detect_faces(
    app: FaceAnalysis,
    frame: np.ndarray,
    angles: tuple[int, ...] = ORIENTATIONS,
    stop_on_first: bool = True,
    use_tiles: bool = True,
) -> list[Face]:
    """
    Detect faces at one or more image rotations.

    SCRFD/ArcFace assume upright faces. A 90° or 180° camera/photo rotation
    yields no boxes even when the face is 200px+ — so we detect on rotated
    copies and map boxes back to the original frame.

    By default we stop at the first rotation that finds faces (typical photos
    are already EXIF-upright). Passing stop_on_first=False scans every angle.
    """
    orig_h, orig_w = frame.shape[:2]
    found: list[Face] = []
    for angle in angles:
        rotated = _rotate_bgr(frame, angle)
        raw = detect_faces_tiled(app, rotated) if use_tiles else app.get(rotated)
        mapped = [_map_face_from_rotated(face, orig_w, orig_h, angle) for face in raw]
        if stop_on_first and mapped:
            return _nms(mapped)
        found.extend(mapped)
    return _nms(found)


def resize_still(frame: np.ndarray, max_side: int = STILL_MAX_SIDE) -> tuple[np.ndarray, float]:
    """Shrink a photo so max(h, w) <= max_side. Returns (image, scale)."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame, 1.0
    scale = max_side / float(longest)
    small = cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return small, scale


def _scale_face(face: Face, factor: float) -> Face:
    bbox = np.asarray(face.bbox, dtype=np.float32) * factor
    kps = None
    if face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32) * factor
    out = Face(bbox=bbox, kps=kps, det_score=face.det_score)
    if face.embedding is not None:
        out.embedding = np.asarray(face.embedding, dtype=np.float32).copy()
    return out


def _candidate_det_sizes(frame: np.ndarray) -> list[tuple[int, int]]:
    """
    SCRFD at det_size=1280 often misses faces in small screenshots (~400px).
    Prefer 640 for small/medium stills; keep 1280 as a fallback for large photos.
    """
    side = max(frame.shape[:2])
    if side < 500:
        candidates = [(640, 640), (320, 320), (512, 512), (1280, 1280)]
    elif side < 1000:
        candidates = [(640, 640), (960, 960), (1280, 1280)]
    else:
        candidates = [(1280, 1280), (640, 640)]
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for size in candidates:
        if size not in seen:
            seen.add(size)
            out.append(size)
    return out


def detect_boxes(
    app: FaceAnalysis,
    frame: np.ndarray,
    input_size: tuple[int, int] | None = None,
) -> list[Face]:
    """SCRFD boxes + 5-point landmarks only — no ArcFace, gender, or 106-pt mesh."""
    sizes = [input_size] if input_size is not None else _candidate_det_sizes(frame)
    for size in sizes:
        bboxes, kpss = app.det_model.detect(frame, max_num=0, input_size=size)
        if bboxes is None or bboxes.shape[0] == 0:
            continue
        faces: list[Face] = []
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            faces.append(Face(bbox=bboxes[i, 0:4], kps=kps, det_score=float(bboxes[i, 4])))
        return faces
    return []


def embed_faces_batch(
    app: FaceAnalysis,
    frame: np.ndarray,
    faces: list[Face],
    min_px: int = 8,
) -> None:
    """ArcFace on this image using native landmarks (same crop InsightFace uses)."""
    rec = app.models.get("recognition")
    if rec is None:
        return
    for face in faces:
        if face.kps is None or face_short_side(face) < min_px:
            continue
        rec.get(frame, face)


def detect_faces_in_still(
    app: FaceAnalysis,
    frame: np.ndarray,
    min_embed_px: int = 8,
) -> list[Face]:
    """
    Detect on a downscaled copy, run ArcFace there (accurate 5-point kps),
    then map boxes back to the original photo. Scaling kps up to 4K and
    re-cropping was producing embeddings that no longer matched enrollments.
    """
    work, scale = resize_still(frame)
    work_h, work_w = work.shape[:2]
    for angle in ORIENTATIONS:
        rotated = _rotate_bgr(work, angle)
        boxes = detect_boxes(app, rotated)
        if not boxes:
            continue
        embed_faces_batch(app, rotated, boxes, min_px=max(1, min_embed_px))
        mapped = [_map_face_from_rotated(face, work_w, work_h, angle) for face in boxes]
        if scale != 1.0:
            sx = frame.shape[1] / float(work_w)
            sy = frame.shape[0] / float(work_h)
            mapped = [_scale_face_xy(face, sx, sy) for face in mapped]
        return _nms(mapped)
    return []


def _scale_face_xy(face: Face, sx: float, sy: float) -> Face:
    bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    bbox[0] *= sx
    bbox[2] *= sx
    bbox[1] *= sy
    bbox[3] *= sy
    kps = None
    if face.kps is not None:
        kps = np.asarray(face.kps, dtype=np.float32).copy()
        kps[:, 0] *= sx
        kps[:, 1] *= sy
    out = Face(bbox=bbox, kps=kps, det_score=face.det_score)
    if face.embedding is not None:
        out.embedding = np.asarray(face.embedding, dtype=np.float32).copy()
    return out


class OrientationTracker:
    """Reuse the last rotation that found faces (a rolled camera stays rolled)."""

    def __init__(self) -> None:
        self.angle = 0

    def detect(self, app: FaceAnalysis, frame: np.ndarray) -> list[Face]:
        ordered = (self.angle,) + tuple(a for a in ORIENTATIONS if a != self.angle)
        for angle in ordered:
            faces = detect_faces(app, frame, angles=(angle,))
            if faces:
                self.angle = angle
                return faces
        return []


def bgr_from_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded image, honoring EXIF orientation."""
    image = Image.open(io.BytesIO(data))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.asarray(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_db(path: str = DB_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(db: dict, path: str = DB_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump(db, f)


def flatten_db(db: dict) -> tuple[list[str], np.ndarray]:
    known_names: list[str] = []
    known_embeddings: list[np.ndarray] = []
    for name, emb_list in db.items():
        for emb in emb_list:
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            nrm = float(np.linalg.norm(vec))
            if nrm < 1e-6:
                continue
            known_names.append(name)
            known_embeddings.append(vec / nrm)
    if not known_embeddings:
        return [], np.zeros((0, 512), dtype=np.float32)
    return known_names, np.vstack(known_embeddings)


def match_embedding(
    embedding: np.ndarray,
    known_names: list[str],
    known_embeddings: np.ndarray,
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[str, float, bool]:
    """Return (label, score, is_known)."""
    if known_embeddings.size == 0:
        return "Unknown", 0.0, False
    scores = known_embeddings @ embedding
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score >= threshold:
        return known_names[best_idx], best_score, True
    return "Unknown", best_score, False


def _label_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate_identities(frame: np.ndarray, labels: list[dict]) -> np.ndarray:
    """Draw named boxes onto a BGR image for the GUI result view."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    font = _label_font(max(16, int(im.width / 55)))
    for item in labels:
        x1, y1, x2, y2 = (int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"]))
        color = (24, 140, 72) if item["known"] else (196, 48, 38)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, im.width // 400))
        text = item["text"]
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 6
        ty = y1 - th - pad * 2
        if ty < 0:
            ty = y1
        draw.rectangle([x1, ty, x1 + tw + pad * 2, ty + th + pad * 2], fill=color)
        draw.text((x1 + pad, ty + pad), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def identify_frame(app: FaceAnalysis, frame: np.ndarray, db: dict) -> tuple[np.ndarray, list[dict]]:
    """Detect (rotation-aware, still-optimized), match, and return annotated image + labels."""
    known_names, known_embeddings = flatten_db(db)
    faces = detect_faces_in_still(app, frame)
    labels: list[dict] = []
    for face in faces:
        embedding = None
        if face.normed_embedding is not None:
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
        px = int(face_short_side(face))
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        if embedding is None:
            name, score, known = "Unknown", 0.0, False
        else:
            name, score, known = match_embedding(
                embedding, known_names, known_embeddings, threshold=STILL_MATCH_THRESHOLD
            )
        text = f"{name} {float(score):.2f}" if score else name
        if px < LOW_RES_PX:
            text = f"{text} LOW-RES"
        labels.append(
            {
                "name": str(name),
                "score": float(score),
                "known": bool(known),
                "px": px,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "text": str(text),
            }
        )
    annotated = annotate_identities(frame, labels)
    return annotated, labels


def enroll_embedding(name: str, embedding: np.ndarray, path: str = DB_PATH) -> None:
    db = load_db(path)
    existing = list(db.get(name, []))
    existing.append(np.asarray(embedding, dtype=np.float32))
    db[name] = existing
    save_db(db, path)


def enroll_from_image(app: FaceAnalysis, name: str, frame: np.ndarray) -> tuple[bool, str]:
    """Enroll exactly one face from a still image (rotation-aware)."""
    faces = detect_faces_in_still(app, frame, min_embed_px=0)
    if len(faces) != 1:
        return False, f"Need exactly one face in the photo, found {len(faces)}."
    embedding = None
    if faces[0].normed_embedding is not None:
        embedding = np.asarray(faces[0].normed_embedding, dtype=np.float32)
    if embedding is None:
        embedding = embed_face(app, frame, faces[0])
    if embedding is None:
        return False, "Could not extract a face embedding from that photo."
    enroll_embedding(name, embedding)
    return True, f"Saved {name}."


def embed_face(app: FaceAnalysis, frame: np.ndarray, face: Face) -> np.ndarray | None:
    """
    Return an L2-normalized embedding.

    Faces already >= 80px use the detector embedding. Smaller crops are padded,
    cubic-upsampled so the short side is 112, then re-run through ArcFace.
    """
    short = face_short_side(face)
    if short >= MIN_EMBED_PX and face.normed_embedding is not None:
        return np.asarray(face.normed_embedding, dtype=np.float32)

    x1, y1, x2, y2 = face.bbox.astype(float)
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx1 = int(max(0, np.floor(x1 - bw * CROP_PAD)))
    cy1 = int(max(0, np.floor(y1 - bh * CROP_PAD)))
    cx2 = int(min(frame.shape[1], np.ceil(x2 + bw * CROP_PAD)))
    cy2 = int(min(frame.shape[0], np.ceil(y2 + bh * CROP_PAD)))
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        if face.normed_embedding is None:
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)

    ch, cw = crop.shape[:2]
    short_crop = min(ch, cw)
    if short_crop < 1:
        if face.normed_embedding is None:
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)

    scale = TARGET_SHORT_PX / float(short_crop)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    restored = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if face.kps is None or "recognition" not in app.models:
        if face.normed_embedding is None:
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)

    kps = (np.asarray(face.kps, dtype=np.float32) - np.array([cx1, cy1], dtype=np.float32)) * scale
    rec_face = Face(
        bbox=np.array([0.0, 0.0, float(new_w), float(new_h)], dtype=np.float32),
        kps=kps.astype(np.float32),
        det_score=face.det_score,
    )
    app.models["recognition"].get(restored, rec_face)
    if rec_face.normed_embedding is None:
        if face.normed_embedding is None:
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)
    return np.asarray(rec_face.normed_embedding, dtype=np.float32)
