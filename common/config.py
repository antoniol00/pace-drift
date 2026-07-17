"""Configuration shared by the three modules of the experiment."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# ── Dataset ──
CSV_PATH = "/mnt/AI-DATA-SSD/alara/Data/CIC-IDS2017.csv"
DROP_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp"}
TARGET_COL = "Attack Type"

# To avoid the long initial block of only BENIGN traffic, we start at row 540,000.
START_ROW = 540_000

# Pre-training samples (within the stream started at START_ROW)
N_PRETRAIN = 50_000

# Maximum samples to process starting from START_ROW
MAX_SAMPLES = 2_000_000

# ── Logging / snapshots ──
LOG_INTERVAL = 10_000
SNAP_INTERVAL = 500

# ── Model (HAT) ──
GRACE_PERIOD = 200
SEED = 42

# ── Drift detection (modules B and C) ──
PHT_DELTA = 0.002
TRAIN_WINDOW = 5000  # samples to train after a detected drift

# ── Metric ──
ROLLING_F1_WINDOW = 5000
