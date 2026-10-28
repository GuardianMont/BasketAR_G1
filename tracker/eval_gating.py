"""Valuta in modo analitico il gating della palla (stadio 2) sui frame ANNOTATI.

Per ogni clip con un frame annotato (annotation/<dir>/index.csv + labels/) ricostruisce la traccia
della palla con le stesse funzioni dello stadio 2 (_candidate_rows, _gate_ball) e confronta la palla
accettata dal gating, in quel frame, con la palla annotata. Un candidato e' "corretto" se il suo centro
cade entro mezza dimensione del box vero (stesso criterio di eval_detector.py).

Metriche per ogni combinazione di parametri (e per il riferimento "senza gate", cioe' il candidato piu'
confidente):  recall, precisione, F1, quante volte il gate accetta un oggetto SBAGLIATO, quante volte
accetta qualcosa in frame senza palla annotata; piu' la visibilita' media della traccia (frame con palla
accettata / frame totali) e la frazione di frame con candidati scartati dal gate.

ATTENZIONE: usa lo split val (frame mai visti dal detector finale). Lo split train da' numeri ottimistici,
perche' quei frame hanno addestrato il detector. Il val ha palle grandi (video piu' vicino): non riflette
le palle piccole.

Uso (dalla cartella principale del progetto):
  python tracker/eval_gating.py --dataset-root dataset_hd --detector-id <ID> --annotation-dir annotation/br \\
      --split val --thresholds 0.2 0.3 0.4 --max-speeds 0.92 1.5 1.95 --max-gaps 0.1 0.2 0.4
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.extract_tracking_features import _candidate_rows, _gate_ball  # noqa: E402


def load_gt(label_file: Path, w: int, h: int):
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            p = line.split()
            if len(p) == 5 and int(p[0]) == 0:
                cx, cy, bw, bh = map(float, p[1:])
                boxes.append((cx * w, cy * h, bw * w, bh * h))
    return boxes


def hits(gts, cand, w, h) -> bool:
    cx, cy = float(cand[0]) * w, float(cand[1]) * h
    return any(math.hypot(g[0] - cx, g[1] - cy) <= 0.5 * max(g[2], g[3]) for g in gts)


def top1(rows: np.ndarray) -> np.ndarray:
    out = np.zeros((len(rows), 5), np.float32)
    for i, cand in enumerate(rows):
        valid = cand[cand[:, 4] > 0]
        if len(valid):
            out[i] = valid[int(np.argmax(valid[:, 4]))]
    return out


def evaluate(data, thr, speed, gap, use_gate):
    total, groups = Counter(), defaultdict(Counter)
    vis, rej = [], []
    for cd in data:
        raw = _candidate_rows({"candidates": cd["cand"]}, 0, thr)
        if use_gate:
            acc, nrej = _gate_ball(raw, cd["times"], cd["sizes"], speed, gap)
        else:
            acc, nrej = top1(raw), 0
        n_cand = int(np.count_nonzero(np.any(raw[:, :, 4] > 0, axis=1)))
        vis.append(float(np.mean(acc[:, 4] > 0)))
        rej.append(nrej / max(1, n_cand))
        for idx, gts, group in cd["frames"]:
            w, h = int(cd["sizes"][idx, 0]), int(cd["sizes"][idx, 1])
            has_acc = acc[idx, 4] > 0
            if gts and has_acc:
                key = "tp" if hits(gts, acc[idx], w, h) else "wrong"
            elif gts:
                key = "miss"
            else:
                key = "false_accept" if has_acc else "tn"
            total[key] += 1
            groups[group][key] += 1
    return total, groups, float(np.mean(vis)), float(np.mean(rej))


def summarize(c: Counter) -> dict:
    n_gt = c["tp"] + c["wrong"] + c["miss"]
    n_none = c["false_accept"] + c["tn"]
    prec_den = c["tp"] + c["wrong"] + c["false_accept"]
    rec = c["tp"] / n_gt if n_gt else float("nan")
    prec = c["tp"] / prec_den if prec_den else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"n_gt": n_gt, "recall": rec, "precision": prec, "f1": f1,
            "wrong": c["wrong"] / n_gt if n_gt else float("nan"),
            "false_accept": c["false_accept"] / n_none if n_none else float("nan"), "n_none": n_none}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default="dataset_hd")
    p.add_argument("--detector-id", required=True)
    p.add_argument("--annotation-dir", required=True)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--detections-root")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3])
    p.add_argument("--max-speeds", type=float, nargs="+", default=[0.92, 1.95])
    p.add_argument("--max-gaps", type=float, nargs="+", default=[0.2])
    args = p.parse_args()

    ann = Path(args.annotation_dir)
    cache_dir = Path(args.detections_root or Path(args.dataset_root) / "detections") / args.detector_id / args.split
    with (ann / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] == args.split]

    by_clip = defaultdict(list)
    for r in rows:
        by_clip[r["clip_id"]].append(r)
    data, skipped = [], 0
    align = Counter()
    for clip_id, frames in by_clip.items():
        path = cache_dir / f"{clip_id}.npz"
        if not path.is_file():
            skipped += 1
            continue
        c = np.load(path, allow_pickle=False)
        cand, times, sizes = c["candidates"], c["timestamps_sec"].astype(np.float64), c["source_wh"]
        entry = {"cand": cand, "times": times, "sizes": sizes, "frames": []}
        for r in frames:
            idx = int(r["frame_idx"])
            if idx >= len(cand):
                skipped += 1
                continue
            w, h = int(sizes[idx, 0]), int(sizes[idx, 1])
            gts = load_gt(ann / "labels" / args.split / (Path(r["image"]).stem + ".txt"), w, h)
            entry["frames"].append((idx, gts, r["group"]))
            if gts:  # controllo allineamento frame annotato <-> frame della cache
                low = _candidate_rows({"candidates": cand}, 0, 0.0)
                for off in (-1, 0, 1):
                    j = idx + off
                    if 0 <= j < len(low) and any(hits(gts, k, int(sizes[j, 0]), int(sizes[j, 1])) for k in low[j] if k[4] > 0):
                        align[off] += 1
                align["n"] += 1
        if entry["frames"]:
            data.append(entry)
    n_frames = sum(len(d["frames"]) for d in data)
    print(f"Clip valutate: {len(data)} | frame annotati: {n_frames} | saltati (cache o indice mancanti): {skipped}")
    if align["n"]:
        print("Allineamento frame annotato/cache (palle annotate con un candidato entro tolleranza): "
              f"idx-1 {align[-1]/align['n']:.0%} | idx {align[0]/align['n']:.0%} | idx+1 {align[1]/align['n']:.0%}"
              "   <- idx deve essere il piu' alto; se e' idx-1 o idx+1, c'e' uno sfasamento di un frame")

    results = []
    for thr in args.thresholds:
        total, groups, vis, rej = evaluate(data, thr, 0, 0, use_gate=False)
        results.append((f"thr={thr} SENZA gate (top-1)", summarize(total), vis, rej, groups))
        for speed, gap in itertools.product(args.max_speeds, args.max_gaps):
            total, groups, vis, rej = evaluate(data, thr, speed, gap, use_gate=True)
            results.append((f"thr={thr} speed={speed} gap={gap}", summarize(total), vis, rej, groups))

    print(f"\n{'configurazione':38s} {'recall':>6s} {'prec':>6s} {'F1':>6s} {'sbagl.':>6s} {'falsi/no-palla':>14s} {'vis':>5s} {'scart':>6s}")
    for name, s, vis, rej, _ in results:
        print(f"{name:38s} {s['recall']:6.2f} {s['precision']:6.2f} {s['f1']:6.2f} {s['wrong']:6.2f} "
              f"{s['false_accept']:8.2f} ({s['n_none']:2d}) {vis:5.2f} {rej:6.2f}")
    gated = [r for r in results if "SENZA" not in r[0]]
    if gated:
        best = max(gated, key=lambda r: r[1]["f1"])
        print(f"\nMiglior F1 con gate: {best[0]}  (F1 {best[1]['f1']:.2f}). Per gruppo di azione:")
        for g, c in sorted(best[4].items()):
            s = summarize(c)
            print(f"  {g:10s} palle annotate={s['n_gt']:3d} recall={s['recall']:.2f} sbagl.={s['wrong']:.2f} "
                  f"falsi/no-palla={s['false_accept']:.2f} ({s['n_none']})")
    print("\nLegenda: sbagl. = palla annotata presente ma il gate ha accettato un altro oggetto; falsi/no-palla = frame "
          "senza palla annotata in cui il gate accetta comunque qualcosa (tra parentesi: quanti frame senza palla); "
          "vis = frame con palla accettata / frame totali delle clip; scart = frazione di frame con candidati scartati dal gate.")


if __name__ == "__main__":
    main()