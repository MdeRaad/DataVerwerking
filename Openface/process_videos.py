import cv2
import csv
import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

# Alles wordt opgelost relatief aan de locatie van dit script.
# Verwachte mapstructuur:
#
#   process_videos.py          ← dit script
#   videos/                    ← input video's
#   output_csv/                ← wordt automatisch aangemaakt
#   weights/
#       Alignment_RetinaFace.pth
#       MTL_backbone.pth
#   OpenFace-3.0/              ← OpenFace broncode (voor interne weights)

SCRIPT_DIR = Path(__file__).resolve().parent
OPENFACE_DIR = SCRIPT_DIR / "OpenFace-3.0"
WEIGHTS_DIR = OPENFACE_DIR / "weights"

# OpenFace laadt soms intern via relatieve paden, dus chdir naar OpenFace-3.0
os.chdir(OPENFACE_DIR)

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

VIDEO_DIR = SCRIPT_DIR / "videos"
OUTPUT_DIR = SCRIPT_DIR / "output_csv"

FACE_MODEL = str(WEIGHTS_DIR / "Alignment_RetinaFace.pth")
MULTI_MODEL = str(WEIGHTS_DIR / "MTL_backbone.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_STEP  = 1

EMOTION_LABELS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# ── CSV-kolommen ──────────────────────────────────────────────────────────────

AU_LABELS = [
    "AU01_intensity",   # Inner Brow Raiser
    "AU02_intensity",   # Outer Brow Raiser
    "AU04_intensity",   # Brow Lowerer
    "AU06_intensity",   # Cheek Raiser
    "AU09_intensity",   # Nose Wrinkler
    "AU12_intensity",   # Lip Corner Puller
    "AU25_intensity",   # Lips Part
    "AU26_intensity",   # Jaw Drop
]

CSV_HEADER = (
    ["frame_index", "timestamp_sec", "face_detected",
     "face_confidence", "face_x1", "face_y1", "face_x2", "face_y2"]
    + [f"emotion_prob_{lbl}" for lbl in EMOTION_LABELS]  # softmax probabilities
    + ["emotion_predicted", "emotion_confidence"]
    + ["gaze_yaw", "gaze_pitch"]
    + AU_LABELS
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def row_no_face(frame_idx: int, timestamp: float) -> list:
    return [frame_idx, round(timestamp, 4), False] + [""] * (len(CSV_HEADER) - 3)


# BMP: geen compressie → snelst te schrijven, volledig lossless
_TMP_FRAME_PATH = os.path.join(tempfile.gettempdir(), "_openface_frame.bmp")


def process_video(
    video_path: Path,
    face_detector: FaceDetector,
    multitask_model: MultitaskPredictor,
    output_dir: Path,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Kan video niet openen: {video_path.name}")
        return

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    csv_path = output_dir / (video_path.stem + ".csv")

    print(f"  -> {video_path.name}  |  {total} frames  |  {fps:.1f} fps  |  device={DEVICE}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        frame_idx = 0
        written   = 0

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                timestamp = frame_idx / fps

                # get_face() verwacht een bestandspad → frame als BMP opslaan
                # BMP is lossless en heeft geen compressiestap, dus minimale overhead
                cv2.imwrite(_TMP_FRAME_PATH, bgr_frame)
                cropped_face, dets = face_detector.get_face(_TMP_FRAME_PATH)

                if cropped_face is None or dets is None:
                    writer.writerow(row_no_face(frame_idx, timestamp))
                else:
                    # Beste detectie (hoogste confidence) uit dets halen
                    best = dets[np.argmax(dets[:, 4])]
                    face_conf = round(float(best[4]), 4)
                    x1, y1, x2, y2 = (round(float(best[0])), round(float(best[1])),
                                      round(float(best[2])), round(float(best[3])))

                    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

                    # Emotie — softmax probabilities opslaan, niet logits
                    logits_np  = emotion_logits.cpu().numpy().flatten()
                    probs      = softmax(logits_np)
                    top_idx    = int(np.argmax(probs))
                    top_label  = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else str(top_idx)
                    confidence = round(float(probs[top_idx]), 4)

                    # Gaze
                    gaze = gaze_output.cpu().numpy().flatten() if hasattr(gaze_output, "cpu") else np.array(gaze_output).flatten()
                    gaze_yaw, gaze_pitch = (round(float(gaze[0]), 4), round(float(gaze[1]), 4)) if len(gaze) >= 2 else ("", "")

                    # Action Units
                    au = au_output.cpu().numpy().flatten() if hasattr(au_output, "cpu") else np.array(au_output).flatten()
                    au_values = [round(float(v), 4) for v in au]
                    au_values += [""] * max(0, len(AU_LABELS) - len(au_values))

                    row = (
                        [frame_idx, round(timestamp, 4), True,
                         face_conf, x1, y1, x2, y2]
                        + [round(float(v), 4) for v in probs]   # probabilities ipv logits
                        + [top_label, confidence]
                        + [gaze_yaw, gaze_pitch]
                        + au_values
                    )
                    writer.writerow(row)
                    written += 1

            frame_idx += 1

    cap.release()
    print(f"     Klaar - {written} frames met gezicht geschreven naar {csv_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    video_dir  = Path(VIDEO_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    videos = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in video_extensions])

    if not videos:
        print(f"Geen video's gevonden in '{video_dir}'.")
        sys.exit(1)

    print(f"Gevonden: {len(videos)} video('s)  |  output -> '{output_dir}'\n")

    face_detector   = FaceDetector(model_path=FACE_MODEL, device=DEVICE)
    multitask_model = MultitaskPredictor(model_path=MULTI_MODEL, device=DEVICE)

    for video_path in videos:
        process_video(video_path, face_detector, multitask_model, output_dir)

    if os.path.exists(_TMP_FRAME_PATH):
        os.remove(_TMP_FRAME_PATH)

    print("\nAlles klaar!")


if __name__ == "__main__":
    main()