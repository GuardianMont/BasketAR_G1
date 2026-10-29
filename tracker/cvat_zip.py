"""Crea lo zip da caricare in CVAT (Actions -> Upload annotations, formato Ultralytics YOLO Detection).

Struttura dello zip (come da documentazione CVAT):
  data.yaml            path: ./   <split>: <split>.txt   names: 0 ball, 1 rim
  <split>.txt          elenco immagini
  images/<split>/*.jpg
  labels/<split>/*.txt

Uso:
  python make_cvat_zip.py --out-dir annotation/br --split train --first 5    # zip di PROVA (5 immagini)
  python make_cvat_zip.py --out-dir annotation/br --split train              # zip completo
I file in CVAT si abbinano per NOME: le immagini del task e queste devono avere gli stessi nomi.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--first", type=int, default=0, help="solo le prime N immagini (zip di prova)")
    args = p.parse_args()

    root = Path(args.out_dir)
    images = sorted((root / "images" / args.split).glob("*.jpg"))
    if args.first:
        images = images[: args.first]
    if not images:
        raise SystemExit(f"Nessuna immagine in {root / 'images' / args.split}")

    suffix = f"_first{args.first}" if args.first else ""
    target = root / f"cvat_import_{args.split}{suffix}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", f"path: ./\n{args.split}: {args.split}.txt\nnames:\n  0: ball\n  1: rim\n")
        zf.writestr(f"{args.split}.txt", "\n".join(f"./images/{args.split}/{i.name}" for i in images) + "\n")
        missing = 0
        for image in images:
            label = root / "labels" / args.split / (image.stem + ".txt")
            if label.exists():
                zf.write(label, f"labels/{args.split}/{label.name}")
            else:
                zf.writestr(f"labels/{args.split}/{image.stem}.txt", "")  # nessun box = sfondo
                missing += 1
            zf.write(image, f"images/{args.split}/{image.name}")  # servono: senza immagini l'import fallisce
    print(f"{target}  ({len(images)} immagini, {missing} senza file label)")


if __name__ == "__main__":
    main()