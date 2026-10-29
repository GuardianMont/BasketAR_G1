"""Crea gli elenchi di immagini e i data.yaml per addestrare/valutare YOLO.

Per ogni video passato con --holdout-video crea anche un esperimento "leave-one-video-out":
addestra senza quel video e lo usa come validazione extra (video mai visto), oltre al val.

File creati in <out-dir> (nella stessa cartella di images/ e labels/):
  train_all.txt, val.txt, data_all.yaml
  train_no_<video>.txt, heldout_<video>.txt, data_no_<video>.yaml (val = video di val),
  data_heldout_<video>.yaml (val = il video escluso, per `yolo val`)

--path e' la cartella del dataset VISTA DAL COMPUTER CHE ADDESTRA (es. /content/br su Colab).
Uso:
  python make_yolo_splits.py --out-dir annotation/br --path /content/br --holdout-video psa_corato
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

NAMES = "names:\n  0: ball\n  1: rim\n"


def write_list(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(f"./images/{r['image']}" for r in rows) + "\n", encoding="utf-8")


def write_yaml(path: Path, root: str, train: str, val: str) -> None:
    path.write_text(f"path: {root}\ntrain: {train}\nval: {val}\n{NAMES}", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--path", help="cartella del dataset sul computer che addestra (default: out-dir assoluto)")
    p.add_argument("--holdout-video", nargs="*", default=[])
    args = p.parse_args()

    out = Path(args.out_dir)
    root = args.path or out.resolve().as_posix()
    with (out / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    train = sorted((r for r in rows if r["split"] == "train"), key=lambda r: r["image"])
    val = sorted((r for r in rows if r["split"] == "val"), key=lambda r: r["image"])
    videos = {r["video_id"] for r in train}
    unknown = set(args.holdout_video) - videos
    if unknown:
        raise SystemExit(f"Video non presenti nel train: {sorted(unknown)} (disponibili: {sorted(videos)})")

    write_list(out / "train_all.txt", train)
    write_list(out / "val.txt", val)
    write_yaml(out / "data_all.yaml", root, "train_all.txt", "val.txt")
    print(f"data_all.yaml: train {len(train)} | val {len(val)}")
    for video in args.holdout_video:
        rest = [r for r in train if r["video_id"] != video]
        held = [r for r in train if r["video_id"] == video]
        write_list(out / f"train_no_{video}.txt", rest)
        write_list(out / f"heldout_{video}.txt", held)
        write_yaml(out / f"data_no_{video}.yaml", root, f"train_no_{video}.txt", "val.txt")
        write_yaml(out / f"data_heldout_{video}.yaml", root, f"train_no_{video}.txt", f"heldout_{video}.txt")
        print(f"data_no_{video}.yaml: train {len(rest)} | val {len(val)} ; data_heldout_{video}.yaml: val {len(held)}")


if __name__ == "__main__":
    main()