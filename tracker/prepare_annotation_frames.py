"""Seleziona e estrae frame da annotare (ball/rim) dalle clip del manifest, in formato YOLO.

Scelte di campionamento (per limitare i bias):
  * frame del TRAIN solo dai video dello split train; frame di VALIDATION del detector solo
    dal video dello split val; lo split test non viene mai toccato;
  * quota uguale per ogni video sorgente (il video più grande non domina);
  * dentro ogni video, quote per gruppo di azione (--quota) e clip distribuite lungo il
    tempo del video (start_time) per coprire inquadrature diverse;
  * 1 frame per clip, posizione casuale sull'intera clip (i tiri hanno una probabilità
    --shot-late-prob di essere presi nella parte finale, vicino al ferro);
  * i file sono numerati (0001_..., 0002_...) in un ordine che mescola video e gruppi in modo
    proporzionale: annotando in ordine alfabetico ogni prefisso (es. i primi 150) è già un
    campione bilanciato, e puoi fermarti e addestrare quando vuoi;
  * idle/non-gioco inclusi: se la palla non c'è restano frame senza box (negativi).

Fase 1:  python prepare_annotation_frames.py --dataset-root dataset_hd --out-dir annotation/br \
             --n-train 360 --n-val 80 [--prelabel --prelabel-weights yolo26m.pt]
Round successivo senza ripescare clip già annotate:
         ... --out-dir annotation/br2 --seed 43 --exclude annotation/br/index.csv
Controllo bias dopo l'annotazione (labels/ esportate in YOLO nella stessa cartella):
         python prepare_annotation_frames.py --out-dir annotation/br --report
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

GROUPS = ("shots", "passaggio", "idle", "non-gioco")
DEFAULT_QUOTA = "shots=0.40,passaggio=0.30,idle=0.15,non-gioco=0.15"
COCO_SPORTS_BALL = 32


def group_of(label: str) -> str:
    return "shots" if label.startswith("tiro") else (label if label in GROUPS else "other")


def parse_quota(text: str) -> dict[str, float]:
    quota = {k: float(v) for k, v in (item.split("=") for item in text.split(","))}
    unknown = set(quota) - set(GROUPS)
    if unknown or abs(sum(quota.values()) - 1.0) > 1e-6:
        raise ValueError(f"--quota: gruppi validi {GROUPS}, somma = 1 (ricevuto {quota})")
    return quota


def read_manifest(path: Path, split: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = {"split", "clip_id", "path", "label"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest: mancano le colonne {sorted(missing)}")
        rows = [r for r in reader if r["split"] == split]
    for r in rows:
        r.setdefault("video_id", "video")
        r["group"] = group_of(r["label"])
    return rows


def spread_pick(items: list[dict], k: int, rng: random.Random) -> list[dict]:
    """k clip distribuite lungo il tempo: una a caso per ciascuno di k intervalli di start_time."""
    if k <= 0 or not items:
        return []
    items = sorted(items, key=lambda r: float(r.get("start_time") or 0))
    if k >= len(items):
        return items
    return [items[rng.choice(chunk.tolist())] for chunk in np.array_split(np.arange(len(items)), k) if len(chunk)]


def select_clips(rows: list[dict], n_total: int, quota: dict[str, float], rng: random.Random) -> list[dict]:
    videos = sorted({r["video_id"] for r in rows})
    chosen = []
    for i, video in enumerate(videos):
        n_video = n_total // len(videos) + (1 if i < n_total % len(videos) else 0)
        for group, share in quota.items():
            pool = [r for r in rows if r["video_id"] == video and r["group"] == group]
            chosen += spread_pick(pool, round(n_video * share), rng)
    return chosen


def interleave(clips: list[dict], rng: random.Random) -> list[dict]:
    """Ordina le clip mescolando proporzionalmente (video, gruppo): ogni prefisso della lista
    ha circa le stesse proporzioni dell'insieme completo."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in clips:
        cells[(row["video_id"], row["group"])].append(row)
    ranked = []
    for items in cells.values():
        rng.shuffle(items)
        ranked += [((i + rng.uniform(0.1, 0.9)) / len(items), row) for i, row in enumerate(items)]
    return [row for _, row in sorted(ranked, key=lambda t: t[0])]


def read_frame(video: Path, index: int):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = None
    for back in range(3):  # gli ultimi frame a volte non si decodificano
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(min(index, total - 1) - back, 0))
        ok, img = cap.read()
        if ok:
            frame = img
            break
    cap.release()
    return frame, total


def prelabel(model, image, conf: float, imgsz: int) -> list[str]:
    """Box 'ball' dalla classe COCO 'sports ball'. Il rim va annotato a mano."""
    res = model.predict(source=image, imgsz=imgsz, conf=conf, classes=[COCO_SPORTS_BALL], verbose=False)[0]
    h, w = image.shape[:2]
    return [f"0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
            for x1, y1, x2, y2 in (res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else [])]


def extract(args) -> None:
    root = Path(args.dataset_root)
    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"
    out, rng, quota = Path(args.out_dir), random.Random(args.seed), parse_quota(args.quota)
    model = None
    if args.prelabel:
        from ultralytics import YOLO
        model = YOLO(args.prelabel_weights)
    excluded: set[str] = set()
    for path in args.exclude:
        with open(path, encoding="utf-8", newline="") as stream:
            excluded |= {r["clip_id"] for r in csv.DictReader(stream)}
    if excluded:
        print(f"Escluse {len(excluded)} clip già usate in round precedenti")
    index_rows = []
    for split, count in (("train", args.n_train), ("val", args.n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        if model:
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        pool = [r for r in read_manifest(manifest, split) if r["clip_id"] not in excluded]
        clips = select_clips(pool, count, quota, rng)
        clips = interleave(clips, rng)
        saved = 0
        for row in clips:
            late = row["group"] == "shots" and rng.random() < args.shot_late_prob
            pos = rng.uniform(0.6, 1.0) if late else rng.uniform(0.05, 0.95)
            video = root / row["path"]
            cap = cv2.VideoCapture(str(video)); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
            if total <= 0:
                print(f"[WARN] clip non leggibile: {video}"); continue
            idx = int(round(pos * (total - 1)))
            frame, _ = read_frame(video, idx)
            if frame is None:
                print(f"[WARN] frame {idx} non leggibile: {video}"); continue
            name = f"{saved + 1:04d}_{row['clip_id']}_f{idx:05d}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if model:
                (out / "labels" / split / f"{name}.txt").write_text(
                    "\n".join(prelabel(model, frame, args.prelabel_conf, args.imgsz)), encoding="utf-8")
            index_rows.append({"image": f"{split}/{name}.jpg", "split": split, "clip_id": row["clip_id"],
                               "label": row["label"], "group": row["group"], "video_id": row["video_id"],
                               "frame_idx": idx, "rel_pos": round(pos, 3), "start_time": row.get("start_time", "")})
            saved += 1
        print(f"{split}: {saved} frame salvati (richiesti {count})")
    with (out / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index_rows[0]))
        writer.writeheader(); writer.writerows(index_rows)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n  1: rim\n", encoding="utf-8")
    print("Fatto. Annota ball (0) e rim (1) in OGNI frame in cui sono visibili, poi esporta in YOLO.")


def report(args) -> None:
    out = Path(args.out_dir)
    with (out / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for split in ("train", "val"):
        sub = [r for r in rows if r["split"] == split]
        stats, widths = [], []
        widths_by_video = defaultdict(list)
        for r in sub:
            label_file = out / "labels" / split / (Path(r["image"]).stem + ".txt")
            lines = label_file.read_text().split("\n") if label_file.exists() else []
            boxes = [l.split() for l in lines if l.strip()]
            img_w = cv2.imread(str(out / "images" / r["image"])).shape[1]
            balls = [b for b in boxes if b[0] == "0"]
            widths += [float(b[3]) * img_w for b in balls]
            widths_by_video[r["video_id"]] += [float(b[3]) * img_w for b in balls]
            stats.append((r["video_id"], r["group"], bool(balls), any(b[0] == "1" for b in boxes), not boxes))
        print(f"\n== {split}: {len(sub)} frame ==")
        print("frame per video x gruppo:")
        for key, n in sorted(Counter((s[0], s[1]) for s in stats).items()):
            print(f"  {key[0]:15s} {key[1]:10s} {n}")
        print("per gruppo -> con palla / con ferro / senza box:")
        by_group = defaultdict(list)
        for s in stats:
            by_group[s[1]].append(s)
        for g, items in sorted(by_group.items()):
            n = len(items)
            print(f"  {g:10s} n={n:3d} palla={sum(i[2] for i in items)/n:.0%} ferro={sum(i[3] for i in items)/n:.0%} "
                  f"vuoti={sum(i[4] for i in items)/n:.0%}")
        if widths:
            q = np.percentile(widths, [10, 50, 90])
            print(f"larghezza palla (px, frame originale) P10/P50/P90 = {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}; "
                  f"palle < 15px: {np.mean(np.array(widths) < 15):.0%} (box: {len(widths)})")
            print("larghezza palla per video (px) P10/P50/P90 | % < 15px | n:")
            for video, vw in sorted(widths_by_video.items()):
                q = np.percentile(vw, [10, 50, 90])
                print(f"  {video:15s} {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} | {np.mean(np.array(vw) < 15):.0%} | {len(vw)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="dataset_hd")
    p.add_argument("--manifest")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-train", type=int, default=360)
    p.add_argument("--n-val", type=int, default=80)
    p.add_argument("--quota", default=DEFAULT_QUOTA)
    p.add_argument("--shot-late-prob", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude", nargs="*", default=[], help="index.csv di round precedenti: le loro clip non vengono ripescate")
    p.add_argument("--prelabel", action="store_true")
    p.add_argument("--prelabel-weights", default="yolo11m.pt")
    p.add_argument("--prelabel-conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--report", action="store_true", help="statistiche delle label annotate (controllo bias)")
    args = p.parse_args()
    report(args) if args.report else extract(args)


if __name__ == "__main__":
    main()