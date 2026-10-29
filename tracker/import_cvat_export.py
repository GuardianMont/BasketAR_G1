"""Riporta nelle cartelle del progetto le etichette esportate da CVAT (Ultralytics YOLO Detection).

Cerca nello zip esportato i file .txt il cui nome coincide con un'immagine di images/<split>/,
verifica che nel data.yaml esportato ball=0 e rim=1, salva un backup delle etichette presenti
e le sostituisce. Funziona qualunque sia la struttura di cartelle dello zip.

Uso:
  python import_cvat_export.py --out-dir annotation/br --split train --export-zip C:/percorso/task_train.zip --dry-run
  python import_cvat_export.py --out-dir annotation/br --split train --export-zip C:/percorso/task_train.zip
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import yaml


def class_names(zf: zipfile.ZipFile) -> dict[int, str] | None:
    for member in zf.namelist():
        if Path(member).name == "data.yaml":
            names = yaml.safe_load(zf.read(member)).get("names")
            return dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in (names or {}).items()}
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--split", required=True, choices=["train", "val"])
    p.add_argument("--export-zip", required=True)
    p.add_argument("--dry-run", action="store_true", help="controlla senza scrivere nulla")
    p.add_argument("--skip-class-check", action="store_true", help="NON consigliato: salta il controllo ball=0/rim=1")
    args = p.parse_args()

    root = Path(args.out_dir)
    stems = {i.stem for i in (root / "images" / args.split).glob("*.jpg")}
    if not stems:
        raise SystemExit(f"Nessuna immagine in {root / 'images' / args.split}")

    with zipfile.ZipFile(args.export_zip) as zf:
        names = class_names(zf)
        if names != {0: "ball", 1: "rim"}:
            msg = f"Classi nell'export: {names} (attese {{0: 'ball', 1: 'rim'}})"
            if not args.skip_class_check:
                raise SystemExit(f"STOP. {msg}\nOrdine delle etichette sbagliato nel task CVAT o data.yaml mancante.")
            print(f"[ATTENZIONE] {msg}")
        found: dict[str, str] = {}
        for member in zf.namelist():
            stem = Path(member).stem
            if member.endswith(".txt") and stem in stems:
                if stem in found:
                    raise SystemExit(f"Due file label per la stessa immagine: {found[stem]} e {member}")
                found[stem] = member
        contents = {stem: zf.read(member).decode("utf-8") for stem, member in found.items()}

    missing = sorted(stems - set(found))
    classes = Counter()
    frames_with = Counter()
    for text in contents.values():
        seen = set()
        for line in text.splitlines():
            if line.strip():
                classes[line.split()[0]] += 1
                seen.add(line.split()[0])
        for c in seen:
            frames_with[c] += 1
    print(f"Immagini {args.split}: {len(stems)} | label trovate nello zip: {len(found)} | senza label: {len(missing)}")
    print(f"Box: palle (0) = {classes['0']} in {frames_with['0']} frame; ferri (1) = {classes['1']} in {frames_with['1']} frame; "
          f"altre classi: {sum(v for k, v in classes.items() if k not in ('0', '1'))}")
    if missing:
        print("  Senza label (verranno scritte VUOTE):", ", ".join(missing[:10]), "..." if len(missing) > 10 else "")
    if args.dry_run:
        print("Dry-run: non ho scritto nulla.")
        return

    target = root / "labels" / args.split
    backup = root / f"labels_{args.split}_prima_di_cvat"
    if target.exists() and not backup.exists():
        shutil.copytree(target, backup)
        print(f"Backup delle etichette precedenti: {backup}")
    target.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (target / f"{stem}.txt").write_text(contents.get(stem, ""), encoding="utf-8")
    print(f"Scritte {len(stems)} etichette in {target}")


if __name__ == "__main__":
    main()