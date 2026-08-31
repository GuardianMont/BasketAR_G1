import argparse
import os
from pathlib import Path
import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# Impostazioni stile grafici
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 10, "figure.autolayout": True})

# 1. Mapping originale a 9 classi
LABEL_ORIGINAL = {
    "passaggio": 0,
    "tiroDaDue0": 1,
    "tiroDaDue1": 2,
    "tiroDaTre0": 3,
    "tiroDaTre1": 4,
    "tiroLibero0": 5,
    "tiroLibero1": 6,
    "idle": 7,
    "non-gioco": 8,
}

# 2. Mapping aggregato a 8 classi (idle + non-gioco -> altro)
LABEL_AGGREGATED = {
    "passaggio": "passaggio",
    "tiroDaDue0": "tiroDaDue0",
    "tiroDaDue1": "tiroDaDue1",
    "tiroDaTre0": "tiroDaTre0",
    "tiroDaTre1": "tiroDaTre1",
    "tiroLibero0": "tiroLibero0",
    "tiroLibero1": "tiroLibero1",
    "idle": "altro",
    "non-gioco": "altro",
}


def map_hierarchical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le colonne per i tre livelli della cascata gerarchica:

    p1: 0 (passaggio), 1 (tiro: 1..6), 2 (altro: 7,8)
    p2: 0 (da due: 1,2), 1 (da tre: 3,4), 2 (libero: 5,6) -> NaN per non-tiri
    p3: 0 (sbagliato: 1,3,5), 1 (segnato: 2,4,6)      -> NaN per non-tiri
    """
    df["label_p1"] = df["label_id"].apply(
        lambda x: "passaggio"
        if x == 0
        else ("tiro" if 1 <= x <= 6 else "altro")
    )

    df["label_p2"] = df["label_id"].apply(
        lambda x: "da_due"
        if x in [1, 2]
        else ("da_tre" if x in [3, 4] else ("libero" if x in [5, 6] else np.nan))
    )

    df["label_p3"] = df["label_id"].apply(
        lambda x: "sbagliato"
        if x in [1, 3, 5]
        else ("segnato" if x in [2, 4, 6] else np.nan)
    )

    df["label_aggregated"] = df["label_raw"].map(LABEL_AGGREGATED)

    return df


def inspect_video_durations(
    df: pd.DataFrame, dataset_root: Path
) -> pd.DataFrame:
    """Scansiona i file video per estrarre FPS, numero di frame totali e durata in secondi."""
    durations, total_frames_list, fps_list = [], [], []

    print("\n[EDA] Estrazione metadati video (FPS, durate, frame totali)...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Videos"):
        video_path = dataset_root / row["path"]
        if not video_path.exists():
            # Fallback se il path nel manifest è relativo a dataset_root
            video_path = dataset_root / row["path"].lstrip("/")

        if not video_path.exists():
            durations.append(np.nan)
            total_frames_list.append(np.nan)
            fps_list.append(np.nan)
            continue

        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                frames = int(stream.frames or 0)
                fps = float(stream.average_rate or 25.0)

                if frames == 0:
                    frames = sum(1 for _ in container.decode(video=0))

                duration = frames / fps if fps > 0 else 0.0

                durations.append(duration)
                total_frames_list.append(frames)
                fps_list.append(fps)
        except Exception:
            durations.append(np.nan)
            total_frames_list.append(np.nan)
            fps_list.append(np.nan)

    df["duration_sec"] = durations
    df["num_frames"] = total_frames_list
    df["fps"] = fps_list

    return df


def generate_eda_plots(df: pd.DataFrame, output_dir: Path):
    """Genera tutti i grafici della distribuzione delle classi e delle durate."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- PLOT 1: Distribuzione Classi Originali (9 Classi) per Split ---
    plt.figure(figsize=(12, 6))
    sns.countplot(
        data=df,
        y="label_raw",
        hue="split",
        order=sorted(df["label_raw"].unique()),
    )
    plt.title("Distribuzione Classi Originali (9 Classi) per Split")
    plt.xlabel("Numero di Clip")
    plt.ylabel("Classe Originale")
    plt.tight_layout()
    plt.savefig(
        plots_dir / "01_distribution_original_9classes.png", dpi=300
    )
    plt.close()

    # --- PLOT 2: Distribuzione Classi Aggregate (8 Classi) ---
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df,
        y="label_aggregated",
        hue="split",
        order=sorted(df["label_aggregated"].unique()),
    )
    plt.title(
        "Distribuzione Classi Aggregate (8 Classi - idle+non-gioco -> altro)"
    )
    plt.xlabel("Numero di Clip")
    plt.ylabel("Classe Aggregata")
    plt.tight_layout()
    plt.savefig(
        plots_dir / "02_distribution_aggregated_8classes.png", dpi=300
    )
    plt.close()

    # --- PLOT 3: Fasi Gerarchiche (p1, p2, p3) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # p1
    sns.countplot(
        data=df,
        x="label_p1",
        hue="split",
        ax=axes[0],
        order=["passaggio", "tiro", "altro"],
    )
    axes[0].set_title("Fase p1: Macro-Azione")
    axes[0].set_xlabel("Classe p1")
    axes[0].set_ylabel("Conteggio")

    # p2 (Solo Tiri)
    df_shots = df[df["label_p1"] == "tiro"]
    sns.countplot(
        data=df_shots,
        x="label_p2",
        hue="split",
        ax=axes[1],
        order=["da_due", "da_tre", "libero"],
    )
    axes[1].set_title("Fase p2: Tipo di Tiro (Solo Tiri)")
    axes[1].set_xlabel("Tipo Tiro")
    axes[1].set_ylabel("Conteggio")

    # p3 (Solo Tiri)
    sns.countplot(
        data=df_shots,
        x="label_p3",
        hue="split",
        ax=axes[2],
        order=["sbagliato", "segnato"],
    )
    axes[2].set_title("Fase p3: Esito Tiro (Solo Tiri)")
    axes[2].set_xlabel("Esito Tiro")
    axes[2].set_ylabel("Conteggio")

    plt.suptitle(
        "Distribuzioni sulle Fasi Gerarchiche (p1 -> p2 -> p3)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        plots_dir / "03_distribution_hierarchical_phases.png", dpi=300
    )
    plt.close()

    # --- PLOT 4: Distribuzione delle Durate (Secondi e Frame) ---
    if "duration_sec" in df.columns and df["duration_sec"].notnull().any():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Istogramma Durata in Secondi per Fase p1
        sns.histplot(
            data=df,
            x="duration_sec",
            hue="label_p1",
            kde=True,
            element="step",
            ax=axes[0],
        )
        axes[0].set_title("Distribuzione Durata Video (Secondi) per Fase p1")
        axes[0].set_xlabel("Durata (secondi)")
        axes[0].set_ylabel("Frequenza")

        # Boxplot Frame Totali per Tipo di Tiro (p2)
        sns.boxplot(
            data=df_shots,
            x="label_p2",
            y="num_frames",
            hue="label_p3",
            ax=axes[1],
        )
        axes[1].set_title("Numero di Frame per Tipo di Tiro ed Esito")
        axes[1].set_xlabel("Tipo di Tiro (p2)")
        axes[1].set_ylabel("Numero Frame Totali")

        plt.tight_layout()
        plt.savefig(plots_dir / "04_video_durations_analysis.png", dpi=300)
        plt.close()

    print(f"[EDA] Tutti i grafici sono stati salvati in: '{plots_dir}'")


def generate_eda_reports(df: pd.DataFrame, output_dir: Path):
    """Genera file CSV riassuntivi e un report di testo completo."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary Conteggi per Split e 9 Classi
    df_9cls = (
        df.groupby(["split", "label_raw"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_9cls.to_csv(reports_dir / "counts_original_9classes.csv", index=False)

    # 2. Summary Conteggi per Split e 8 Classi Aggregate
    df_8cls = (
        df.groupby(["split", "label_aggregated"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    df_8cls.to_csv(reports_dir / "counts_aggregated_8classes.csv", index=False)

    # 3. Summary Gerarchico p1, p2, p3
    p1_summary = df.groupby(["split", "label_p1"]).size().unstack(fill_value=0)
    p2_summary = (
        df[df["label_p1"] == "tiro"]
        .groupby(["split", "label_p2"])
        .size()
        .unstack(fill_value=0)
    )
    p3_summary = (
        df[df["label_p1"] == "tiro"]
        .groupby(["split", "label_p3"])
        .size()
        .unstack(fill_value=0)
    )

    p1_summary.to_csv(reports_dir / "counts_phase_p1.csv")
    p2_summary.to_csv(reports_dir / "counts_phase_p2.csv")
    p3_summary.to_csv(reports_dir / "counts_phase_p3.csv")

    # 4. Report Testuale Dettagliato
    report_file = output_dir / "eda_summary_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("   BASKET ACTION RECOGNITION - EXPLORATORY DATA ANALYSIS (EDA)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Totale Campioni nel Dataset: {len(df)}\n")
        f.write(
            f"Distribuzione per Split: {df['split'].value_counts().to_dict()}\n\n"
        )

        f.write("-" * 50 + "\n1. FASE p1 (Macro-Azione):\n" + "-" * 50 + "\n")
        f.write(f"{p1_summary.to_string()}\n\n")

        f.write("-" * 50 + "\n2. FASE p2 (Tipo di Tiro - Solo Tiri):\n" + "-" * 50 + "\n")
        f.write(f"{p2_summary.to_string()}\n\n")

        f.write("-" * 50 + "\n3. FASE p3 (Esito del Tiro - Solo Tiri):\n" + "-" * 50 + "\n")
        f.write(f"{p3_summary.to_string()}\n\n")

        if "duration_sec" in df.columns and df["duration_sec"].notnull().any():
            f.write("-" * 50 + "\n4. STATISTICHE DURATE E FRAME:\n" + "-" * 50 + "\n")
            f.write(
                f"Durata Media (secondi): {df['duration_sec'].mean():.2f} +/-"
                f" {df['duration_sec'].std():.2f}\n"
            )
            f.write(
                f"Durata Minima: {df['duration_sec'].min():.2f}s | Durata"
                f" Massima: {df['duration_sec'].max():.2f}s\n"
            )
            f.write(
                f"Frame Totali Medi per Clip: {df['num_frames'].mean():.1f} +/-"
                f" {df['num_frames'].std():.1f}\n"
            )
            f.write(
                f"FPS Medi: {df['fps'].mean():.1f}\n\n"
            )

            f.write("Durate Medie per Tipo di Azione (p1):\n")
            p1_dur = df.groupby("label_p1")["duration_sec"].agg(["mean", "std", "min", "max"])
            f.write(f"{p1_dur.to_string()}\n")

    print(f"[EDA] Report testuale e CSV salvati in: '{reports_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Esegue l'EDA completa del dataset di Basket Action Recognition."
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset/dataset_hd",
        help="Path della radice del dataset contenente manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="eda",
        help="Directory dove salvare i grafici e i report generati.",
    )
    parser.add_argument(
        "--skip-video-inspection",
        action="store_true",
        help="Salta la lettura fisica dei file video per la durata se si vuole un'EDA veloce.",
    )

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    manifest_path = dataset_root / "manifest.csv"

    # Caricamento o costruzione autonoma del DataFrame dei dati
    if manifest_path.exists():
        print(f"[EDA] Caricamento del manifest da: '{manifest_path}'")
        df = pd.read_csv(manifest_path)
    else:
        print(
            f"[WARNING] Manifest non trovato in '{manifest_path}'. Scansione"
            " ricorsiva della cartella..."
        )
        records = []
        for video_path in dataset_root.rglob("*.mp4"):
            parts = video_path.relative_to(dataset_root).parts
            if len(parts) >= 2:
                split = parts[0] if parts[0] in ["train", "val", "test"] else "all"
                label_raw = parts[1] if split != "all" else parts[0]
                records.append({
                    "path": str(video_path.relative_to(dataset_root)),
                    "split": split,
                    "label": label_raw,
                    "clip_id": video_path.stem,
                })
        df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"Impossibile trovare campioni in '{dataset_root}'."
        )

    # Standardizzazione colonne
    if "label" in df.columns:
        df["label_raw"] = df["label"]

    df["label_id"] = df["label_raw"].map(LABEL_ORIGINAL)

    # Mappatura etichette gerarchiche ed aggregate
    df = map_hierarchical_labels(df)

    # Ispezione fisica delle durate dei video (se non saltata)
    if not args.skip_video_inspection:
        df = inspect_video_durations(df, dataset_root)

    # Creazione cartelle di output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvataggio del DataFrame EDA completo
    df.to_csv(output_dir / "eda_dataset_dataframe.csv", index=False)

    # Generazione report e grafici
    generate_eda_reports(df, output_dir)
    generate_eda_plots(df, output_dir)

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Analisi EDA completata con successo!")
    print(f"[OUTPUT] Tutti i risultati sono stati organizzati in: '{output_dir}/'")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()