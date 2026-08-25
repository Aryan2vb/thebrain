# THEBRAIN

Local face recognition with OpenCV + InsightFace (`buffalo_l`).

- **Webcam enroll** — capture 5 head poses into `face_db.pkl`
- **Live recognize** — label faces from the camera
- **Photo GUI** — enroll from a portrait, identify people in a group photo

## Requirements

- Python 3.10+ (3.11–3.13 recommended)
- Webcam (for live scripts)
- macOS / Linux / Windows

First InsightFace run downloads the `buffalo_l` model automatically.

## Setup

```bash
git clone https://github.com/Aryan2vb/thebrain.git
cd thebrain

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Quick start (photo GUI)

```bash
source .venv/bin/activate
python gui.py
```

Open [http://127.0.0.1:5050](http://127.0.0.1:5050)

1. **Enroll a face** — type a name, drop one front-face photo (JPG / PNG / WebP)
2. **Read a photo** — drop a group picture; named boxes appear on matches

Wait until the status says **Ready.** before uploading (model load can take a bit).

## Webcam enroll

```bash
python enrollment.py
```

1. Enter the person’s name in the terminal  
2. Look **Front → Up → Down → Left → Right**  
3. Press `c` to capture each pose (exactly one face in frame)  
4. Press `q` to quit  

Optional noisy CCTV simulation:

```bash
python enrollment.py --cctv
python enrollment.py --cctv --cctv-strength 1.5
```

## Live recognition

Enroll at least one person first.

```bash
python recognize.py
```

- Green box = known match (score ≥ threshold)  
- Red box = unknown  
- Press `q` to quit  

```bash
python recognize.py --cctv
```

## Data

| File | Purpose |
|------|---------|
| `face_db.pkl` | Local face embeddings (created on first enroll) |

`face_db.pkl` is gitignored — do not commit it (it stores biometric vectors).

## Tips

- Prefer a clear **front face** for GUI enroll (one person only).
- Small screenshots are OK; the pipeline tries several detector sizes.
- Group photos: expect a few seconds on CPU for many faces.
- Distance / low pixels: labels may show `LOW-RES` when the face box is tiny.
- Camera / photo rotated 90° or 180°: detection tries other orientations.

## Project layout

```
enrollment.py      # Webcam multi-pose enroll
recognize.py       # Live webcam recognition
gui.py             # Photo enroll + identify UI
face_pipeline.py   # InsightFace, tiling, stills, matching
cctv_effects.py    # Optional CCTV noise simulation
requirements.txt
```
