"""Configuración compartida por los tres módulos del experimento."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ── Dataset ──
CSV_PATH = "/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv"
DROP_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}
TARGET_COL = "Attack Type"

# Para evitar el largo bloque inicial de solo tráfico BENIGN, empezamos en la fila 540,000.
START_ROW = 540_000

# Muestras de pre-entrenamiento (dentro del stream iniciado en START_ROW)
N_PRETRAIN = 50_000

# Muestras máximas a procesar a partir de START_ROW
MAX_SAMPLES = 2_000_000

# ── Logging / snapshots ──
LOG_INTERVAL = 10_000
SNAP_INTERVAL = 500

# ── Modelo (HAT) ──
GRACE_PERIOD = 200
SEED = 42

# ── Drift detection (módulos B y C) ──
PHT_DELTA = 0.002
TRAIN_WINDOW = 5000  # muestras a entrenar tras un drift detectado

# ── Métrica ──
ROLLING_F1_WINDOW = 5000
