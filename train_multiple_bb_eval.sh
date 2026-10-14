#!/bin/bash

# Interrompi lo script se un comando fallisce
set -e

TRAIN_DIR="features/train"
VAL_DIR="features/val"

# Verifica che le directory esistano
if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$VAL_DIR" ]; then
    echo "[ERROR] Le cartelle '$TRAIN_DIR' e/o '$VAL_DIR' non esistono."
    exit 1
fi

echo "=========================================================="
echo "AVVIO SPERIMENTAZIONE BATCH SU TUTTI I PAIR TRAIN/VAL"
echo "=========================================================="

for train_file in "$TRAIN_DIR"/train__*.pt; do
    # Se la globbing non trova file, esce dal ciclo
    [ -e "$train_file" ] || continue

    # Estrae il nome base del file (es. train__x3dl__...__4cacfaa4.pt)
    filename=$(basename "$train_file")
    
    # Sostituisce il prefisso "train__" con "val__" per trovare il file corrispondente
    val_filename="${filename/train__/val__}"
    val_file="$VAL_DIR/$val_filename"

    # Verifica la presenza del file di validation corrispondente
    if [ ! -f "$val_file" ]; then
        echo "[WARNING] File di validation non trovato per: $filename"
        echo "          Cercato: $val_file -> Salto questa coppia."
        continue
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "Esecuzione Probing su coppia:"
    echo "  ├─ Train: $train_file"
    echo "  └─ Val:   $val_file"
    echo "----------------------------------------------------------"

    # Esecuzione dello script Python con i parametri di default ed il pair di split coerente
    python evaluate_backbones.py \
        --train_path "$train_file" \
        --test_path "$val_file" \
        --expected_dim 432 \
        --pooling mean \
        --phase all

done

echo ""
echo "=========================================================="
echo "[SUCCESS] Tutti i pair train/val sono stati valutati!"
echo "=========================================================="