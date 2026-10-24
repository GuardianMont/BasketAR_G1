"""Disegna sulle immagini un puntino rosso dove le label dicono che c'è la palla
(e un puntino blu dove dicono che c'è il ferro), con un cerchio attorno per vederlo meglio.

Salva le immagini annotate in <out-dir>/review/overlay/<split>/ oppure le sfoglia a schermo.

Uso:
  python show_label_dots.py --out-dir annotation/test            # salva tutte le immagini
  python show_label_dots.py --out-dir annotation/test --show     # le sfoglia a schermo
Tasti con --show:  d = avanti   a = indietro   q (o Esc) = esci
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

RED, BLUE, WHITE = (0, 0, 255), (255, 80, 0), (255, 255, 255)  # BGR


def read_boxes(label_file: Path):
    if not label_file.exists():
        return None  # file mancante
    boxes = []
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def annotate(image, boxes, title: str):
    h, w = image.shape[:2]
    out = image.copy()
    n_ball = n_rim = 0
    for cls, xc, yc, bw, bh in boxes or []:
        cx, cy = int(xc * w), int(yc * h)
        color = RED if cls == 0 else BLUE
        radius = max(18, int(0.9 * max(bw * w, bh * h)))
        cv2.circle(out, (cx, cy), radius, WHITE, 3)      # alone bianco per il contrasto
        cv2.circle(out, (cx, cy), radius, color, 2)
        cv2.circle(out, (cx, cy), 5, color, -1)          # il puntino
        cv2.circle(out, (cx, cy), 5, WHITE, 1)
        n_ball += cls == 0
        n_rim += cls == 1
    status = "FILE LABEL MANCANTE" if boxes is None else f"palle={n_ball} ferri={n_rim}"
    cv2.rectangle(out, (0, 0), (min(w, 900), 34), (0, 0, 0), -1)
    cv2.putText(out, f"{title}  |  {status}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="cartella con images/ e labels/ (es. annotation/br)")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--show", action="store_true", help="sfoglia a schermo invece di salvare")
    p.add_argument("--scale", type=float, default=1.0, help="ridimensiona le immagini salvate (es. 0.5)")
    args = p.parse_args()

    root = Path(args.out_dir)
    items = []
    for split in args.splits:
        for image_path in sorted((root / "images" / split).glob("*.jpg")):
            items.append((split, image_path, root / "labels" / split / (image_path.stem + ".txt")))
    if not items:
        raise SystemExit(f"Nessuna immagine trovata in {root / 'images'}")

    def render(i):
        split, image_path, label_file = items[i]
        image = cv2.imread(str(image_path))
        return annotate(image, read_boxes(label_file), f"[{i+1}/{len(items)}] {split}/{image_path.name}")

    if args.show:
        i = 0
        while True:
            frame = render(i)
            scale = min(1.0, 1500 / frame.shape[1])
            cv2.imshow("label (d=avanti a=indietro q=esci)", cv2.resize(frame, None, fx=scale, fy=scale))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            i = (i + 1) % len(items) if key in (ord("d"), 32) else (i - 1) % len(items) if key == ord("a") else i
        cv2.destroyAllWindows()
        return

    for i, (split, image_path, _) in enumerate(items):
        out_dir = root / "review" / "overlay" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = render(i)
        if args.scale != 1.0:
            frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / image_path.name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{len(items)} immagini salvate in {root / 'review' / 'overlay'}")


if __name__ == "__main__":
    main()