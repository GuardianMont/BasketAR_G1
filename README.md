# BasketAR_G1 — Basketball Action Recognition

Sistema di **Action Recognition basato su Deep Learning** per il riconoscimento automatico di azioni di gioco in clip video di partite di basket.

Il progetto esplora differenti strategie di estrazione e classificazione delle caratteristiche visive e temporali, con l'obiettivo di individuare architetture efficaci nel riconoscimento delle azioni, valutandone le prestazioni.

## Obiettivi
Il progetto si concentra sulla classificazione di clip video di basket, assegnando a ciascuna clip una classe corrispondente all'azione rappresentata.

Gli obiettivi principali sono:
- Sviluppare una pipeline modulare per il riconoscimento delle azioni.
- Analizzare e confrontare differenti rappresentazioni delle caratteristiche video.
- Modellare le dipendenze temporali tra le feature estratte dai frame.
- Sperimentare diverse combinazioni di moduli di classificazione e funzioni di loss.
- Gestire lo sbilanciamento tra le classi.
- Valutare le prestazioni mediante metriche di classificazione e analisi degli errori.

Il sistema è progettato per supportare esperimenti riproducibili e confronti tra configurazioni differenti.

## Classi del dataset
Il dataset utilizza nove etichette originali:

| Classe | Descrizione |
|---|---|
| `passaggio` | Passaggio della palla |
| `tiroDaDue0` | Tiro da due punti, punto non realizzato |
| `tiroDaDue1` | Tiro da due punti, punto realizzato |
| `tiroDaTre0` | Tiro da tre punti, punto non realizzato |
| `tiroDaTre1` | Tiro da tre punti, punto realizzato |
| `tiroLibero0` | Tiro libero, punto non realizzato |
| `tiroLibero1` | Tiro libero, punto realizzato |
| `idle` | Fase senza un'azione attiva |
| `non-gioco` | Contenuto estraneo all'azione di gioco considerata |


## Architettura del sistema

L'implementazione adotta una struttura modulare che separa la rappresentazione dei dati, la modellazione temporale e la classificazione.

### Pipeline di riconoscimento

**Input Video → Backbone → Temporal Neck → Classification Head → Prediction**

- **Backbone:** estrae dai frame del video le rappresentazioni visive utilizzate per riconoscere le azioni. Il progetto supporta diverse architetture neurali per l'estrazione delle feature, implementate nella cartella `models/backbone.py`.
- **Temporal Neck:** elabora la sequenza di feature estratte dalla backbone, modellando le dipendenze temporali tra le osservazioni.
- **Classification Head:** aggrega le informazioni temporali e produce i punteggi di classificazione relativi alle classi previste dalla fase corrente.
- **Training and Evaluation:** gestisce l'addestramento dei modelli, la validazione, il calcolo delle metriche e il confronto tra le diverse configurazioni sperimentali.

La struttura modulare consente di sperimentare diverse combinazioni di **backbone, neck, head e funzioni di loss**, facilitando il confronto tra architetture senza dover modificare l'intera pipeline. Inoltre, permette di analizzare separatamente il contributo dell'estrazione delle feature, della modellazione temporale e della classificazione finale.

## Classificazione gerarchica

Oltre alla classificazione delle etichette originali, il progetto implementa una strategia gerarchica articolata in tre fasi.

| Fase | Obiettivo | Classi |
|---|---|---|
| **P1 — Macro-azione** | Identificare la categoria generale dell'azione | Passaggio, tiro, altro |
| **P2 — Tipo di tiro** | Distinguere la tipologia del tiro | Due punti, tre punti, tiro libero |
| **P3 — Esito** | Riconoscere il risultato del tiro | Realizzato, non realizzato |

Le fasi P2 e P3 sono rilevanti per la classificazione dei tiri. Le rispettive etichette vengono ricavate dalle annotazioni originali, mentre le predizioni possono essere ricombinate per ottenere una classificazione finale.

Questa impostazione consente di analizzare sia le prestazioni dei singoli livelli sia quelle del sistema gerarchico complessivo.

##  Struttura del repository

```text
BasketAR_G1/
├── configs/                 # Configurazioni degli esperimenti
├── data/                    # Dataset e gestione delle fasi
├── losses/                  # Funzioni di loss
├── models/                  # Pipeline, backbone, neck, head e training
├── scripts/                 # Utility per la generazione degli esperimenti
├── tests/                   # Test del progetto
├── tracker/                 # Moduli relativi al tracking
├── utils/                   # Feature loader, metriche e utility
├── evaluate_backbones.py    # Valutazione delle rappresentazioni
├── evaluate_test.py         # Valutazione sul test set
├── grid_search.py           # Ricerca delle configurazioni
├── run_eda.py               # Exploratory Data Analysis
├── data_analisys.ipynb      # Notebook di analisi dei dati
├── res_analysis.ipynb       # Notebook di analisi dei risultati
├── requirements.txt         # Dipendenze Python
└── README.md
```

## ⚙️ Requisiti

- Python compatibile con le versioni delle dipendenze specificate nel file `requirements.txt`.
- PyTorch e torchvision.

Le risorse hardware richieste dipendono dalle dimensioni delle feature, dal batch size e dalla configurazione selezionata.

##  Installazione

Clonare il repository:

```bash
git clone https://github.com/GuardianMont/BasketAR_G1.git
cd BasketAR_G1
```

Creare e attivare un ambiente virtuale:

```bash
python -m venv .venv
```

Su Linux/macOS:

```bash
source .venv/bin/activate
```

Su Windows:

```powershell
.venv\Scripts\activate
```

Installare le dipendenze:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Utilizzo

### 1. Exploratory Data Analysis

Per analizzare la distribuzione delle classi, le durate dei video e le statistiche del dataset:

```bash
python run_eda.py --dataset-root PATH_DEL_DATASET --output-dir eda
```

Per saltare l'ispezione dei file video e generare un'analisi più rapida:

```bash
python run_eda.py --dataset-root PATH_DEL_DATASET --output-dir eda --skip-video-inspection
```

Il comando genera report, tabelle CSV e grafici utili per studiare la distribuzione delle classi e le caratteristiche dei campioni.

### 2. Ricerca degli iperparametri

Lo script `grid_search.py` permette di confrontare configurazioni diverse definite nei file YAML.

Esempio di invocazione:

```bash
python grid_search.py \
  --config_path configs/NOME_CONFIG.yaml \
  --grid_name NOME_GRID \
  --phases p1 p2 p3 \
  --batch_size 32
```

I nomi del file di configurazione e della grid devono essere sostituiti con quelli effettivamente presenti nel repository. Le fasi disponibili comprendono `p1`, `p2`, `p3` e `flat`.

Per analizzare risultati già generati, senza avviare un nuovo training:

```bash
python grid_search.py \
  --config_path configs/NOME_CONFIG.yaml \
  --grid_name NOME_GRID \
  --phases p1 p2 p3 \
  --report_only
```

### 3. Valutazione delle backbone

Lo script `evaluate_backbones.py` permette di valutare rappresentazioni estratte da differenti backbone mediante strategie di probing, tra cui classificatori MLP e KNN.

```bash
python evaluate_backbones.py --help
```

I risultati comprendono metriche aggregate e matrici di confusione, utili per confrontare la qualità delle rappresentazioni.

### 4. Valutazione sul test set

Per valutare le architetture configurate:

```bash
python evaluate_test.py \
  --config_path configs/NOME_CONFIG.yaml \
  --test_name NOME_SEZIONE
```

con un file di configurazione che deve avere una struttura del tipo:
```yaml
metadata:
  train_embeddings_path: "path/to/train/embeddings.pt"
  val_embeddings_path: "path/to/val/embeddings.pt"
  backbone_embed_dim: EMBEDDING_DIM
  root_dir: "experiments"
  checkpoint_dir: "checkpoints"
  plot_dir: "plots"
  log_dir: "logs"
  base_seed: 42

global_train_params:
  epochs: 100
  patience: 12
  training_times: 5
  device: "cuda"

grid_search_example:
  neck:
    BiGRU:
      hidden_dim: [128, 256]
      num_layers: [1, 2]
      dropout: [0.2]
    TCNNeck:
      num_channels: # every bullet is a parameter
        - [128, 256]
        - [256, 512]
        - [128, 256, 512]
      kernel_size: [3, 5]
      dropout: [0.2]
  head:
    MeanPooling: {} # {} = default params
  loss:
    CrossEntropy: {}
  optimizer:
    lr: [5e-4]
    weight_decay: [1e-4]
```

Per eseguire la valutazione gerarchica completa:

```bash
python evaluate_test.py \
  --config_path configs/NOME_CONFIG.yaml \
  --test_name NOME_SEZIONE \
  --downstream
```


Per visualizzare tutti i parametri disponibili:

```bash
python evaluate_test.py --help
```

##  Valutazione
Le prestazioni vengono analizzate attraverso diverse metriche:

- **Accuracy:** proporzione di predizioni corrette sul totale dei campioni.
- **Macro F1-score:** media non pesata degli F1-score delle singole classi, utile in presenza di classi sbilanciate.
- **Loss:** valore della funzione obiettivo utilizzata nella valutazione.
- **Confusion matrix:** rappresentazione degli errori e delle confusioni tra le classi.

Il Macro F1-score è particolarmente importante quando alcune azioni sono rappresentate da un numero di campioni molto inferiore rispetto alle altre.

Le valutazioni vengono salvate in file CSV e accompagnate, dove previsto, da matrici di confusione e metadati degli esperimenti.

I risultati numerici devono essere riportati sulla base delle effettive esecuzioni sperimentali, distinguendo le metriche di validazione da quelle ottenute sul test set.


---
