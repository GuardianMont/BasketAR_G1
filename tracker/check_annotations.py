"""Controlla le annotazioni YOLO (ball=0, rim=1) prodotte con prepare_annotation_frames.py.

1) Validazione automatica: formato, classi, coordinate, duplicati, box sospetti,
   tiri senza palla/ferro, troppe palle in un frame.
2) Mosaici di ritagli (review/) per controllare a occhio tutti i box di palla e ferro.

Uso:  python check_annotations.py --out-dir annotation/br
Output: review/report.txt, review/flagged.csv, review/balls_<split>_NN.jpg, review/rims_<split>_NN.jpg
Nei mosaici ogni ritaglio ha un numero #k: review/tiles.csv dice a quale frame appartiene.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

TILE, COLS, PER_SHEET = 200, 6, 30


def parse_label(path: Path):
    """Restituisce (box, errori). box = [(classe, xc, yc, w, h)]."""
    boxes, errors = [], []
    if not path.exists():
        return boxes, ["file label mancante (conta come sfondo)"]
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            cls, vals = int(parts[0]), [float(v) for v in parts[1:]]
            assert len(parts) == 5
        except (ValueError, AssertionError):
            errors.append(f"riga {n}: formato non valido"); continue
        if cls not in (0, 1):
            errors.append(f"riga {n}: classe {cls} non valida")
        xc, yc, w, h = vals
        if not all(0 <= v <= 1 for v in vals) or w <= 0 or h <= 0:
            errors.append(f"riga {n}: coordinate fuori da [0,1] o dimensioni nulle")
        boxes.append((cls, xc, yc, w, h))
    return boxes, errors


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[1]-a[3]/2, a[2]-a[4]/2, a[1]+a[3]/2, a[2]+a[4]/2
    bx1, by1, bx2, by2 = b[1]-b[3]/2, b[2]-b[4]/2, b[1]+b[3]/2, b[2]+b[4]/2
    iw, ih = max(0, min(ax2, bx2)-max(ax1, bx1)), max(0, min(ay2, by2)-max(ay1, by1))
    inter = iw * ih
    union = a[3]*a[4] + b[3]*b[4] - inter
    return inter / union if union > 0 else 0.0


def crop_tile(img, box, caption, zoom_rim):
    h, w = img.shape[:2]
    _, xc, yc, bw, bh = box
    xc, yc, bw, bh = xc*w, yc*h, bw*w, bh*h
    half = int(max(90, bw * (1.6 if zoom_rim else 1.0) * (2 if zoom_rim else 1)))
    half = min(half, min(w, h) // 2)
    x0, y0 = int(min(max(xc-half, 0), w-2*half)), int(min(max(yc-half, 0), h-2*half))
    crop = img[y0:y0+2*half, x0:x0+2*half].copy()
    cv2.rectangle(crop, (int(xc-bw/2)-x0, int(yc-bh/2)-y0), (int(xc+bw/2)-x0, int(yc+bh/2)-y0), (0, 0, 255), 1)
    crop = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
    cv2.putText(crop, caption, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return crop


def write_sheets(tiles, out: Path, prefix: str):
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s:s+PER_SHEET]
        chunk += [np.zeros((TILE, TILE, 3), np.uint8)] * (-len(chunk) % COLS)
        sheet = np.vstack([np.hstack(chunk[i:i+COLS]) for i in range(0, len(chunk), COLS)])
        cv2.imwrite(str(out / f"{prefix}_{s//PER_SHEET+1:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    root = Path(args.out_dir)
    review = root / "review"; review.mkdir(exist_ok=True)
    with (root / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    flagged, tiles_csv, lines = [], [], []
    for split in ("train", "val"):
        ball_tiles, rim_tiles = [], []
        n_frames = n_ball = n_rim = 0
        for r in (x for x in rows if x["split"] == split):
            img = cv2.imread(str(root / "images" / r["image"]))
            if img is None:
                flagged.append((r["image"], "immagine non leggibile")); continue
            h, w = img.shape[:2]
            boxes, errors = parse_label(root / "labels" / split / (Path(r["image"]).stem + ".txt"))
            n_frames += 1
            balls = [b for b in boxes if b[0] == 0]; rims = [b for b in boxes if b[0] == 1]
            n_ball += len(balls); n_rim += len(rims)
            notes = list(errors)
            if len(balls) > 3: notes.append(f"{len(balls)} palle nel frame: sicuro?")
            if len(rims) > 2: notes.append(f"{len(rims)} ferri nel frame: sicuro?")
            if r["group"] == "shots" and not balls: notes.append("tiro senza palla annotata")
            if r["group"] == "shots" and not rims: notes.append("tiro senza ferro annotato")
            for i, a in enumerate(boxes):
                for b in boxes[i+1:]:
                    if a[0] == b[0] and iou(a, b) > 0.9: notes.append("box duplicato")
                if a[0] == 0:
                    pw, ph = a[3]*w, a[4]*h
                    if min(pw, ph) < 6: notes.append(f"palla molto piccola ({pw:.0f}x{ph:.0f}px)")
                    if not 0.5 <= pw/ph <= 2.0: notes.append(f"palla con forma anomala ({pw:.0f}x{ph:.0f}px)")
            for note in sorted(set(notes)):
                flagged.append((r["image"], note))
            for b in boxes:
                k = len(tiles_csv)
                tiles_csv.append((k, r["image"], b[0], f"{b[1]:.4f} {b[2]:.4f} {b[3]:.4f} {b[4]:.4f}"))
                cap = f"#{k} {r['group'][:6]} {b[3]*w:.0f}x{b[4]*h:.0f}"
                (ball_tiles if b[0] == 0 else rim_tiles).append(crop_tile(img, b, cap, zoom_rim=b[0] == 1))
        write_sheets(ball_tiles, review, f"balls_{split}")
        write_sheets(rim_tiles, review, f"rims_{split}")
        lines.append(f"{split}: {n_frames} frame, {n_ball} palle, {n_rim} ferri")

    with (review / "flagged.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([("image", "problema"), *flagged])
    with (review / "tiles.csv").open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([("tile", "image", "classe", "box"), *tiles_csv])
    lines.append(f"\nSegnalazioni: {len(flagged)} (vedi review/flagged.csv)")
    for image, note in flagged[:40]:
        lines.append(f"  {image}: {note}")
    (review / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines)); print(f"\nMosaici in {review}")


if __name__ == "__main__":
    main()