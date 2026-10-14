"""Confronto appaiato fra due esperimenti addestrati con scripts/train_phase1_baseline.py.

Uso (da root repo):
  python -m scripts.compare_experiments CONTROLLO VARIANTE [--metric val_macro_f1] [--alpha 0.05] [--detail]
CONTROLLO e VARIANTE sono cartelle (--output-dir del training) o direttamente runs.csv.
Il confronto e' appaiato per (fase, architettura, lr, seed): usa gli stessi seed nei due training.
Non carica torch ne' il test split. delta = VARIANTE - CONTROLLO (positivo = variante migliore).

Disegno (fissato a priori, per non gonfiare i falsi positivi con pochi seed):
  1. PRIMARIO: composito = media della metrica su p1, p2, p3 (per architettura e seed). Un solo test per
     esperimento, quindi nessuna correzione per confronti multipli. 'flat' (9 classi in un solo passo) e'
     solo un riferimento: non fa parte del composito e non entra nelle decisioni.
  2. REPLICA FRA ARCHITETTURE: un effetto vale solo se ha lo stesso segno in TUTTE le architetture e se
     il p-value PEGGIORE fra le architetture e' < alpha (intersection-union: valido a livello alpha senza
     correzioni aggiuntive).
  3. PER FASE: stessa replica, con correzione di Holm su sole k fasi (k <= 3). Serve a non dichiarare
     "significativa" una fase solo perche' ne hai guardate tre.

Colonne della tabella --detail (una riga per fase e architettura):
  n         numero di seed appaiati.
  A, B      metrica media del CONTROLLO (A) e della VARIANTE (B) sugli n seed.
  delta     media delle n differenze per seed (B - A).
  sd        deviazione standard delle n differenze per seed (non dei punteggi): quanto la differenza
            cambia da un seed all'altro.
  ci_lo/hi  estremi dell'intervallo di confidenza al 95% di delta: delta -/+ t(0.975, n-1) * sd/sqrt(n).
            Se contiene 0, la differenza non e' distinguibile da zero.
  wins      in quanti seed la variante batte il controllo (B > A).
  p         p-value a DUE CODE del t-test appaiato, con ipotesi nulla "la differenza vera e' zero".
            E' la probabilita' di osservare una differenza grande almeno cosi' SE non ci fosse nessun effetto
            (il rumore e' la variabilita' fra seed). NON e' la probabilita' che l'effetto sia reale, NON
            misura la grandezza dell'effetto (per quella: delta e IC) e in questa tabella NON e' corretto per
            confronti multipli. Assume differenze per seed circa normali (con pochi seed non verificabile).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["phase", "architecture", "learning_rate", "seed"]
DECISION_PHASES = ["p1", "p2", "p3"]


def load_runs(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        found = glob.glob(os.path.join(path, "PHASE1_*", "runs.csv"))
        if not found:
            raise FileNotFoundError(f"Nessun runs.csv in '{path}/PHASE1_*'.")
        path = max(found, key=os.path.getmtime)  # run piu' recente
    print(f"[INFO] {path}")
    return pd.read_csv(path)


def holm(pvals) -> np.ndarray:
    """Correzione di Holm: p ordinati crescenti, moltiplicati per k, k-1, ..., 1; mai decrescenti, max 1."""
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_stats(a, b) -> dict:
    """Statistiche appaiate su due vettori (stessi seed, stesso ordine). Vedi le colonne nel docstring."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    if n > 1 and se > 0:
        half = stats.t.ppf(0.975, n - 1) * se                      # semi-ampiezza IC 95%
        p = float(2 * stats.t.sf(abs(mean / se), n - 1))           # p-value a due code, t con n-1 gdl
    else:  # n=1 oppure differenze tutte uguali
        half = 0.0 if n > 1 else float("nan")
        p = (0.0 if mean != 0 else 1.0) if n > 1 else float("nan")
    return dict(n=n, A=float(np.mean(a)), B=float(np.mean(b)), delta=mean, sd=sd,
                ci_lo=mean - half, ci_hi=mean + half, wins=int((d > 0).sum()), p=p)


def intersection_union(rows: list[dict]) -> tuple[float, int]:
    """Replica: stesso segno in tutte le unita' (architetture); p = massimo dei p. Segni discordi -> (1, 0)."""
    deltas = [r["delta"] for r in rows]
    if all(x > 0 for x in deltas):
        return max(r["p"] for r in rows), +1
    if all(x < 0 for x in deltas):
        return max(r["p"] for r in rows), -1
    return 1.0, 0


def verdict(p: float, direction: int, alpha: float) -> str:
    if direction == 0 or not p < alpha:
        return "n.s."
    return "B migliore" if direction > 0 else "B peggiore"


def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str, alpha: float):
    m = a.merge(b, on=KEYS, suffixes=("_A", "_B"), how="outer", indicator=True)
    if (m["_merge"] != "both").any():
        raise ValueError(
            "Run non appaiabili: fase/architettura/lr/seed diversi fra i due esperimenti. "
            "Usa gli stessi --seed, --num-runs, --architectures, --learning-rates."
        )
    m = m.sort_values(KEYS)
    ca, cb = f"{metric}_A", f"{metric}_B"

    detail = []  # una riga per (fase, architettura, lr): p NON corretto
    for (phase, arch, lr), g in m.groupby(["phase", "architecture", "learning_rate"]):
        detail.append(dict(phase=phase, arch=arch, lr=lr, **paired_stats(g[ca], g[cb])))
    detail = pd.DataFrame(detail)

    # per fase: replica fra architetture + Holm su k fasi. delta/IC = media fra architetture per seed.
    sec = []
    for phase in [p for p in DECISION_PHASES if p in set(detail.phase)]:
        rows = detail[detail.phase == phase].to_dict("records")
        p_iu, direction = intersection_union(rows)
        g = m[m.phase == phase].groupby("seed")[[ca, cb]].mean()
        ps = paired_stats(g[ca], g[cb])
        sec.append(dict(phase=phase, units=len(rows), delta=ps["delta"], ci_lo=ps["ci_lo"],
                        ci_hi=ps["ci_hi"], p_replica=p_iu, direction=direction))
    sec = pd.DataFrame(sec)
    if len(sec):
        sec["p_holm"] = holm(sec["p_replica"].to_numpy())
        sec["verdict"] = [verdict(r.p_holm, r.direction, alpha) for r in sec.itertuples()]

    # primario: composito p1,p2,p3 per (arch, lr, seed), replica fra architetture, nessuna correzione
    prim = None
    phases = [p for p in DECISION_PHASES if p in set(m.phase)]
    if len(phases) == len(DECISION_PHASES):
        comp = (m[m.phase.isin(phases)].groupby(["architecture", "learning_rate", "seed"])[[ca, cb]]
                .mean().reset_index())
        rows = [dict(arch=arch, lr=lr, **paired_stats(g[ca], g[cb]))
                for (arch, lr), g in comp.groupby(["architecture", "learning_rate"])]
        p_iu, direction = intersection_union(rows)
        g = comp.groupby("seed")[[ca, cb]].mean()
        prim = dict(units=pd.DataFrame(rows), pooled=paired_stats(g[ca], g[cb]), p_replica=p_iu,
                    direction=direction, verdict=verdict(p_iu, direction, alpha))
    return detail, sec, prim


def _fmt_row(label, d, p_verdict, direction, verdict_):
    p_txt = f"{p_verdict:.3f}" if direction != 0 else "-"
    ci = f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
    return (f"{label:<20s}{d['delta']:+.3f}   {ci:<23s}{p_txt:<12s}"
            f"{'si' if direction else 'no':<10s}{verdict_}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("variant")
    ap.add_argument("--metric", default="val_macro_f1", choices=["val_macro_f1", "val_accuracy"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--detail", action="store_true", help="stampa anche la tabella per architettura")
    args = ap.parse_args()

    detail, sec, prim = compare(load_runs(args.control), load_runs(args.variant), args.metric, args.alpha)
    pd.set_option("display.width", 220)
    n = int(detail["n"].min())
    print(f"\n{args.metric} | delta = variante - controllo | {n} seed appaiati, medi sulle architetture")
    if n < 8:
        print(f"[ATTENZIONE] con {n} seed l'IC e' largo: usa >=10 seed per decidere.")
    print(f"\n{'':<20s}{'delta':<9s}{'IC 95% (media archi.)':<23s}{'p verdetto':<12s}{'coerente':<10s}verdetto")
    if prim:
        print(_fmt_row("PRIMARIO (p1+p2+p3)", prim["pooled"], prim["p_replica"], prim["direction"], prim["verdict"]))
    for r in sec.itertuples():
        d = dict(delta=r.delta, ci_lo=r.ci_lo, ci_hi=r.ci_hi)
        print(_fmt_row(f"  fase {r.phase}", d, r.p_holm, r.direction, r.verdict))
    print("\ndelta e IC: media fra architetture per ogni seed (descrittivi). 'flat' escluso da tutto.")
    print("p verdetto: il piu' sfavorevole fra le architetture (replica); per fase corretto con Holm su"
          f" {len(sec)} fasi. Verdetto 'n.s.' se p >= {args.alpha} o se i segni sono discordi (coerente = no).")
    if args.detail:
        print("\n[dettaglio per fase e architettura; 'p' NON corretto, vedi legenda]")
        print(detail.round(4).to_string(index=False))
        if prim:
            print("\n[composito p1+p2+p3 per architettura: da qui viene il p del PRIMARIO]")
            print(prim["units"].round(4).to_string(index=False))
        print("\nLegenda: delta = media(B-A) per seed | sd = dev. std. delle differenze per seed | ci = IC 95% (t)"
              " | wins = seed con B>A")
        print("p = p-value a due code del t-test appaiato (H0: differenza vera = 0): probabilita' di vedere una"
              " differenza cosi' grande SE non ci fosse effetto.")
        print("    Non e' la probabilita' che l'effetto sia reale, non misura la grandezza, non e' corretto qui.")


if __name__ == "__main__":
    main()