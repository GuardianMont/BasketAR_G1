import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

def test_tracking_on_video(video_path: str, output_path: str = "output_tracked.mp4", conf_thresh: float = 0.20):
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERRORE] Video non trovato: {video_path}")
        return

    print(f"[INFO] Caricamento YOLOv8n standard (COCO)...")
    model = YOLO("yolov8n.pt")
    ball_class_id = 32  # 'sports ball' in COCO

    cap = cv2.VideoCapture(str(video_path))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Configura il VideoWriter per salvare il video di output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory = []  # Memorizza la cronologia dei punti per disegnare la scia/traiettoria

    print(f"[INFO] Elaborazione video '{video_path.name}'...")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Inferenza YOLO filtrando per classe 32
        results = model(frame, classes=[ball_class_id], verbose=False)[0]

        best_box = None
        max_conf = 0.0

        if len(results.boxes) > 0:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    # Coordinate assolute [x_min, y_min, x_max, y_max]
                    best_box = box.xyxy[0].cpu().numpy()

        # Se la palla viene rilevata con conf > soglia, disegna il tracciamento
        if best_box is not None and max_conf >= conf_thresh:
            x1, y1, x2, y2 = map(int, best_box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            trajectory.append((cx, cy))

            # 1. Bounding Box verde attorno alla palla
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Palla {max_conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Palla non rilevata", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 2. Disegna la scia rossa della traiettoria (ultimi 15 punti)
        for i in range(1, len(trajectory[-15:])):
            if trajectory[-15:][i - 1] is None or trajectory[-15:][i] is None:
                continue
            thickness = int(np.sqrt(15 / float(i + 1)) * 2.5)
            cv2.line(frame, trajectory[-15:][i - 1], trajectory[-15:][i], (0, 0, 255), thickness)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[COMPLETATO] Video annotato salvato in: '{output_path}'")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path del video da testare")
    parser.add_argument("--out", default="output_tracked.mp4", help="Path del video di output")
    args = parser.parse_args()

    test_tracking_on_video(args.video, args.out)