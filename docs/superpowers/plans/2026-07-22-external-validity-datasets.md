# External Validity Across Datasets (#4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the base-learner × trigger 2×2 (experiment 3c) on CIC-IDS2018 and UNSW-NB15 to establish external validity of the proactive "when-to-adapt" trigger, keeping CICIDS2017 as the anchor.

**Architecture:** The prequential engine is already feature-agnostic (feature dim/order inferred from the first sample; river models consume dicts). We add a thin *dataset* layer: a per-dataset YAML profile (csv path, target/benign labels, drop columns, stream window, recalibrated monitor thresholds), a `--dataset` CLI flag, a `benign_label` config threaded into the two places that hardcode `"BENIGN"`, and result-path namespacing so new datasets never collide with or rerun 2017.

**Tech Stack:** Python 3, uv (own `.venv`), river 0.23.0, PyTorch (CPU/CUDA), pandas/pyarrow, pytest, ruff, mypy, matplotlib. Config in YAML merged in `config.py`.

## Global Constraints

- Everything in English. NumPy-style docstrings on public functions; type hints everywhere.
- `make lint` (ruff) and `make typecheck` (mypy) MUST stay clean; the existing test suite MUST stay green.
- No bare `print` in library code (`src/driftgan/`); CLI/script prints are fine. Structured logging via `driftgan.logging`.
- Every tunable lives in `ExperimentConfig`; never hardcode experiment constants in modules. Datasets are configured only through `configs/datasets/<name>.yaml`.
- `results/` is gitignored. **Do NOT move, rewrite, or rerun existing CICIDS2017 results.** `dataset == "cicids2017"` must keep the current flat result paths and the current config behavior byte-for-byte.
- Config merge precedence is `base.yaml ← phase.yaml ← dataset.yaml ← CLI overrides` (dataset overrides phase; this is intentional so per-dataset monitor calibration wins over the phase default). The dataset layer is only consulted when `--dataset` names a non-`cicids2017` dataset.
- Use `uv run` for all Python (`uv` at `~/.local/bin/uv`; `export PATH="$HOME/.local/bin:$PATH"` if missing).
- CSVs live at `/mnt/AI-DATA-SSD/alara/Data/` (outside the repo).

---

### Task 1: `benign_label` in row preprocessing

**Files:**
- Modify: `src/driftgan/data/stream.py`
- Test: `tests/test_dataset_support.py` (create)

**Interfaces:**
- Produces: `preprocess_row(row, drop_cols, target_col, benign_label="BENIGN") -> Sample` and `stream_csv(path, drop_cols, target_col, start_row=0, max_rows=None, benign_label="BENIGN")`. `BENIGN`/`ATTACK` remain the binary label constants the model sees.

- [ ] **Step 1: Write the failing test**

Create `tests/test_dataset_support.py`:

```python
"""Multi-dataset support: benign-label threading, config layering, namespacing."""
from driftgan.data.stream import ATTACK, BENIGN, preprocess_row


def test_preprocess_row_custom_benign_label():
    row = {"f1": "1.0", "f2": "2.0", "Label": "Benign"}
    x, y, y_raw = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == BENIGN
    assert y_raw == "Benign"
    assert x == {"f1": 1.0, "f2": 2.0}


def test_preprocess_row_attack_under_custom_benign():
    row = {"f1": "1.0", "Label": "DoS"}
    _, y, _ = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == ATTACK


def test_preprocess_row_2017_string_is_attack_for_other_dataset():
    # When the dataset's benign class is "Benign", the literal "BENIGN" is a
    # normal attack subtype string, not the benign class.
    row = {"f1": "1.0", "Label": "BENIGN"}
    _, y, _ = preprocess_row(row, drop_cols=set(), target_col="Label", benign_label="Benign")
    assert y == ATTACK
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_support.py -v`
Expected: FAIL (`preprocess_row() got an unexpected keyword argument 'benign_label'`).

- [ ] **Step 3: Implement**

In `src/driftgan/data/stream.py`, change the two function signatures and bodies. `preprocess_row`:

```python
def preprocess_row(
    row: dict[str, str],
    drop_cols: frozenset[str] | set[str],
    target_col: str,
    benign_label: str = BENIGN,
) -> Sample:
```

and its first two body lines:

```python
    y_raw = row.pop(target_col, benign_label).strip()
    y = BENIGN if y_raw == benign_label else ATTACK
```

`stream_csv`: add the parameter and forward it:

```python
def stream_csv(
    path: str,
    drop_cols: frozenset[str] | set[str],
    target_col: str,
    start_row: int = 0,
    max_rows: int | None = None,
    benign_label: str = BENIGN,
) -> Iterator[Sample]:
```

and in the loop:

```python
            x, y, y_raw = preprocess_row(dict(row), drop_cols, target_col, benign_label)
```

Also update the module docstring line to note the benign label is configurable.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_support.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/driftgan/data/stream.py tests/test_dataset_support.py
git commit -m "feat(data): configurable benign_label in row preprocessing"
```

---

### Task 2: `benign_label` in the label-derived schedule

**Files:**
- Modify: `src/driftgan/drift/schedule.py`
- Test: `tests/test_dataset_support.py` (append)

**Interfaces:**
- Consumes: `BENIGN` constant (already imported in schedule.py).
- Produces: `build_schedule_from_labels(labels, min_gap=5000, n_pretrain=0, benign_label="BENIGN") -> DriftSchedule`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_support.py`:

```python
from driftgan.drift.schedule import build_schedule_from_labels


def test_schedule_from_labels_custom_benign():
    labels = ["Normal"] * 10 + ["DoS"] * 3 + ["Normal"] * 10 + ["Worms"] * 3
    sched = build_schedule_from_labels(labels, min_gap=1, n_pretrain=0, benign_label="Normal")
    assert sched.indices == [10, 23]


def test_schedule_from_labels_default_benign_unchanged():
    labels = ["BENIGN"] * 5 + ["DoS"] * 2 + ["BENIGN"] * 5 + ["DDoS"] * 2
    sched = build_schedule_from_labels(labels, min_gap=1, n_pretrain=0)
    assert sched.indices == [5, 12]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_support.py -k schedule_from_labels -v`
Expected: FAIL (`build_schedule_from_labels() got an unexpected keyword argument 'benign_label'`).

- [ ] **Step 3: Implement**

In `src/driftgan/drift/schedule.py`, change the signature:

```python
def build_schedule_from_labels(
    labels: list[str] | Any,
    min_gap: int = 5_000,
    n_pretrain: int = 0,
    benign_label: str = BENIGN,
) -> DriftSchedule:
```

and the benign skip in the loop:

```python
        if y_raw == benign_label:
            continue
```

Add `benign_label` to the returned `metadata` dict:

```python
        metadata={"min_gap": min_gap, "n_pretrain": n_pretrain, "benign_label": benign_label},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_support.py -k schedule_from_labels -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/driftgan/drift/schedule.py tests/test_dataset_support.py
git commit -m "feat(drift): configurable benign_label in label schedule"
```

---

### Task 3: `dataset` + `benign_label` config fields, dataset merge layer, run-dir namespacing

**Files:**
- Modify: `src/driftgan/config.py`
- Test: `tests/test_dataset_support.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `ExperimentConfig.dataset: str = "cicids2017"`, `ExperimentConfig.benign_label: str = "BENIGN"`, `ExperimentConfig.run_dir(seed)` namespaced by dataset, and `load_config(path, overrides=None, dataset=None)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_support.py`:

```python
from driftgan import config as cfg_mod
from driftgan.config import ExperimentConfig, load_config


def test_config_defaults_dataset_and_benign():
    c = ExperimentConfig(phase="B")
    assert c.dataset == "cicids2017"
    assert c.benign_label == "BENIGN"


def test_run_dir_flat_for_default_dataset():
    c = ExperimentConfig(phase="B", results_dir="/tmp/r")
    assert c.run_dir(0).as_posix().endswith("/runs/B/seed_0")


def test_run_dir_namespaced_for_named_dataset():
    c = ExperimentConfig(phase="B", dataset="cicids2018", results_dir="/tmp/r")
    assert c.run_dir(0).as_posix().endswith("/runs/cicids2018/B/seed_0")


def test_load_config_dataset_layer_overrides_phase(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "base.yaml").write_text("recon_pht_threshold: 2.0\nstart_row: 0\n")
    (configs / "datasets" / "ds.yaml").write_text(
        "dataset: ds\ncsv_path: /x.csv\ntarget_col: Label\n"
        "benign_label: Benign\nrecon_pht_threshold: 0.05\n"
    )
    phase_yaml = configs / "phase_c_early.yaml"
    phase_yaml.write_text("phase: C_early\nrecon_pht_threshold: 0.1\n")
    monkeypatch.setattr(cfg_mod, "CONFIGS_DIR", configs)
    c = load_config(phase_yaml, dataset="ds")
    assert c.dataset == "ds"
    assert c.benign_label == "Benign"
    assert c.csv_path == "/x.csv"
    assert c.recon_pht_threshold == 0.05  # dataset layer overrides the phase default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_support.py -k "dataset or run_dir or benign" -v`
Expected: FAIL (`dataset` field / `run_dir` namespacing / `load_config(dataset=...)` not present).

- [ ] **Step 3: Implement**

In `src/driftgan/config.py`:

Add two fields in the "Dataset / stream" block of `ExperimentConfig` (right after `phase: str` is fine; keep near the stream fields):

```python
    # Dataset identity
    dataset: str = "cicids2017"  # names configs/datasets/<dataset>.yaml; also namespaces results
    benign_label: str = "BENIGN"  # raw benign class string for this dataset
```

Add a namespacing helper and update `run_dir`:

```python
    def _ns_segments(self) -> tuple[str, ...]:
        """Result-path segment for the dataset (empty for the 2017 anchor)."""
        return () if self.dataset == "cicids2017" else (self.dataset,)

    def run_dir(self, seed: int) -> Path:
        """Directory where artifacts of (phase, seed) are written."""
        return Path(self.results_dir).joinpath("runs", *self._ns_segments(), self.phase, f"seed_{seed}")
```

Change `load_config` to accept `dataset` and insert the dataset layer with `base ← phase ← dataset ← overrides` precedence:

```python
def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    dataset: str | None = None,
) -> ExperimentConfig:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        phase_cfg: dict[str, Any] = yaml.safe_load(f) or {}

    base_path = CONFIGS_DIR / "base.yaml"
    base_cfg: dict[str, Any] = {}
    if base_path.exists() and base_path.resolve() != path.resolve():
        with open(base_path, encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f) or {}

    dataset_cfg: dict[str, Any] = {}
    if dataset and dataset != "cicids2017":
        ds_path = CONFIGS_DIR / "datasets" / f"{dataset}.yaml"
        if not ds_path.exists():
            raise FileNotFoundError(f"dataset profile not found: {ds_path}")
        with open(ds_path, encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f) or {}

    merged = _deep_merge(base_cfg, phase_cfg)
    merged = _deep_merge(merged, dataset_cfg)  # dataset overrides phase (per-dataset calibration)
    if overrides:
        merged = _deep_merge(merged, overrides)
    if dataset:
        merged["dataset"] = dataset  # the --dataset flag is authoritative

    if "drop_cols" in merged:
        merged["drop_cols"] = tuple(merged["drop_cols"])
    known = {f_.name for f_ in ExperimentConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    extra = {k: v for k, v in merged.items() if k not in known}
    kwargs = {k: v for k, v in merged.items() if k in known and k != "extra"}
    return ExperimentConfig(**kwargs, extra=extra)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_support.py -k "dataset or run_dir or benign" -v`
Expected: PASS.

- [ ] **Step 5: Typecheck + commit**

```bash
make typecheck
git add src/driftgan/config.py tests/test_dataset_support.py
git commit -m "feat(config): dataset profile layer + result namespacing + benign_label"
```

---

### Task 4: Namespace recovery parquet by dataset

**Files:**
- Modify: `src/driftgan/metrics/recovery.py`
- Test: `tests/test_dataset_support.py` (append)

**Interfaces:**
- Produces: `save_recovery(events, phase, seed, results_dir, dataset="cicids2017") -> Path` writing `recovery/{phase}_{seed}.parquet` for 2017 and `recovery/{dataset}_{phase}_{seed}.parquet` otherwise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dataset_support.py`:

```python
from driftgan.metrics.recovery import save_recovery


def test_save_recovery_flat_for_default(tmp_path):
    p = save_recovery([], "B", 0, tmp_path)
    assert p.name == "B_0.parquet"


def test_save_recovery_namespaced(tmp_path):
    p = save_recovery([], "B", 0, tmp_path, dataset="cicids2018")
    assert p.name == "cicids2018_B_0.parquet"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_dataset_support.py -k save_recovery -v`
Expected: FAIL (`save_recovery() got an unexpected keyword argument 'dataset'`).

- [ ] **Step 3: Implement**

In `src/driftgan/metrics/recovery.py`, replace `save_recovery`:

```python
def save_recovery(
    events: list[RecoveryEvent],
    phase: str,
    seed: int,
    results_dir: str | Path,
    dataset: str = "cicids2017",
) -> Path:
    """Write ``results/recovery/{prefix}{phase}_{seed}.parquet`` and return its path.

    ``prefix`` is empty for the 2017 anchor and ``{dataset}_`` otherwise, so
    per-dataset recovery frames never collide with the existing 2017 outputs.
    """
    out_dir = Path(results_dir) / "recovery"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if dataset == "cicids2017" else f"{dataset}_"
    path = out_dir / f"{prefix}{phase}_{seed}.parquet"
    events_to_dataframe(events).to_parquet(path, index=False)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_dataset_support.py -k save_recovery -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/driftgan/metrics/recovery.py tests/test_dataset_support.py
git commit -m "feat(metrics): namespace recovery parquet by dataset"
```

---

### Task 5: Thread `benign_label` + `dataset` through the runner

**Files:**
- Modify: `src/driftgan/experiments/runner.py`

**Interfaces:**
- Consumes: `config.benign_label`, `config.dataset` (Task 3); `stream_csv(..., benign_label=...)` (Task 1); `save_recovery(..., dataset=...)` (Task 4).
- Produces: no signature changes; runs now honor the dataset profile end-to-end.

- [ ] **Step 1: Modify the stream construction**

In `src/driftgan/experiments/runner.py`, the `stream = stream_csv(...)` call (near line 253) becomes:

```python
    stream = stream_csv(
        config.csv_path,
        drop_cols=set(config.drop_cols),
        target_col=config.target_col,
        start_row=config.start_row,
        max_rows=config.max_samples,
        benign_label=config.benign_label,
    )
```

- [ ] **Step 2: Modify the recovery write**

In `_write_outputs`, the `save_recovery(...)` call (near line 450) becomes:

```python
    save_recovery(events, result.phase, result.seed, config.results_dir, dataset=config.dataset)
```

- [ ] **Step 3: Run the full suite + typecheck as the gate**

Run: `make test typecheck`
Expected: PASS (all existing tests + the new `test_dataset_support.py` stay green; 2017 behavior unchanged because defaults are `cicids2017`/`BENIGN`).

- [ ] **Step 4: Commit**

```bash
git add src/driftgan/experiments/runner.py
git commit -m "feat(runner): honor dataset profile (benign_label, namespaced recovery)"
```

---

### Task 6: `--dataset` CLI flag + Makefile targets

**Files:**
- Modify: `run_experiment.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `load_config(path, overrides, dataset)` (Task 3).
- Produces: `run_experiment.py --dataset <name>` threading the dataset into both the phase-run and `--build-schedule` config loads.

- [ ] **Step 1: Add the CLI flag**

In `run_experiment.py`, add the argument (after `--out`):

```python
    parser.add_argument("--dataset", default="cicids2017",
                        help="Dataset profile in configs/datasets/<name>.yaml (default: cicids2017)")
```

Thread it into the build-schedule branch:

```python
        config = _fit_pretrain(load_config(config_path, overrides, dataset=args.dataset))
```

and into the phase-run branch:

```python
    config = _fit_pretrain(load_config(config_path, overrides, dataset=args.dataset))
```

(Both existing `load_config(config_path, overrides)` calls gain `dataset=args.dataset`.)

- [ ] **Step 2: Add Makefile targets**

In `Makefile`, add (use tabs for recipe lines; `SEEDS` defaults to the existing project convention):

```make
# ── External-validity datasets (#4): 2018 + UNSW ──────────────────────
DATA2018 ?= cicids2018
DATAUNSW ?= unsw
MINPHASES = C_early ARF_early SRP_early
SCHEDPHASES = B ARF_sched SRP_sched

schedule-2018:
	uv run python run_experiment.py --build-schedule --dataset $(DATA2018) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATA2018).json

schedule-unsw:
	uv run python run_experiment.py --build-schedule --dataset $(DATAUNSW) \
		--config configs/phase_b.yaml --out results/drift_schedule_$(DATAUNSW).json

run-min-2018:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--drift-schedule results/drift_schedule_$(DATA2018).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(MINPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATA2018) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done

run-min-unsw:
	@for p in $(SCHEDPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--drift-schedule results/drift_schedule_$(DATAUNSW).json \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
	@for p in $(MINPHASES); do \
		uv run python run_experiment.py --phase $$p --dataset $(DATAUNSW) \
			--seeds $(SEEDS) --n-jobs $(NJOBS); done
```

If `NJOBS`/`SEEDS` are not already defined in the Makefile, add near the top: `SEEDS ?= 0,1,2,3,4` and `NJOBS ?= 1`. (Check first with `grep -n "SEEDS\|NJOBS\|n-jobs" Makefile`.)

- [ ] **Step 3: Smoke-test that 2017 behavior is unchanged**

Run:
```bash
uv run python run_experiment.py --phase B --dataset cicids2017 --seeds 0 --max-samples 3000
```
Expected: completes and writes to the **flat** path `results/runs/B/seed_0/` (no `cicids2017/` segment), printing a summary row. Confirm no `results/runs/cicids2017/` directory was created:
```bash
test ! -d results/runs/cicids2017 && echo "OK: 2017 stays flat"
```

- [ ] **Step 4: Lint + commit**

```bash
make lint typecheck
git add run_experiment.py Makefile
git commit -m "feat(cli): --dataset flag + 2018/UNSW schedule and run targets"
```

---

### Task 7: Dataset profiles for 2018 and UNSW (with drift-window scan)

**Files:**
- Create: `scripts/scan_drift_window.py`
- Create: `configs/datasets/cicids2018.yaml`
- Create: `configs/datasets/unsw.yaml`

**Interfaces:**
- Produces: two dataset profiles selectable via `--dataset cicids2018` / `--dataset unsw`.

Confirmed facts (from the label-ordering diagnostic already run this session):
- **CIC-IDS2018** — `csv_path: /mnt/AI-DATA-SSD/alara/Data/CIC-IDS2018.csv`, `target_col: Label`, `benign_label: Benign`. Header has no `Flow ID`/IP columns; drop `Timestamp` (identifier), keep `Dst Port`/`Protocol`.
- **UNSW-NB15** — `csv_path: /mnt/AI-DATA-SSD/alara/Data/UNSW-NB15.csv`, `target_col: Label`, `benign_label: Benign`. CICFlowMeter re-extraction; drop identifiers `Flow ID, Src IP, Dst IP, Src Port, Timestamp`, keep `Dst Port`/`Protocol`.

- [ ] **Step 1: Create the scan script**

Create `scripts/scan_drift_window.py`:

```python
"""Scan attack density + subtype diversity along a stream to choose a
drift-rich (start_row, max_samples) window for a dataset profile.

Usage:
    uv run python scripts/scan_drift_window.py <csv> <label_col> <benign_label> [bucket]
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

csv.field_size_limit(10_000_000)


def main() -> None:
    path, label_col, benign = sys.argv[1], sys.argv[2], sys.argv[3]
    bucket = int(sys.argv[4]) if len(sys.argv) > 4 else 250_000

    n = 0
    b_idx = 0
    b_attack = 0
    b_subtypes: Counter[str] = Counter()
    print(f"scanning {path} (bucket={bucket})\n")
    print(f"{'bucket':>7} {'rows':>9} {'attack%':>8} {'#subtypes':>10}  top-subtypes")
    with open(path, encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        for row in r:
            lab = (row.get(label_col) or "").strip()
            if lab != benign:
                b_attack += 1
                b_subtypes[lab] += 1
            n += 1
            if n % bucket == 0:
                top = ", ".join(f"{k}({v})" for k, v in b_subtypes.most_common(4))
                print(f"{b_idx:>7} {bucket:>9} {100 * b_attack / bucket:>7.1f}% "
                      f"{len(b_subtypes):>10}  {top}")
                b_idx += 1
                b_attack = 0
                b_subtypes = Counter()
    print(f"\ntotal rows: {n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the scan for both datasets**

```bash
uv run python scripts/scan_drift_window.py /mnt/AI-DATA-SSD/alara/Data/CIC-IDS2018.csv Label Benign 250000 | tee /tmp/scan_2018.txt
uv run python scripts/scan_drift_window.py /mnt/AI-DATA-SSD/alara/Data/UNSW-NB15.csv Label Benign 250000 | tee /tmp/scan_unsw.txt
```

Selection criteria (write the chosen numbers into the YAMLs in Step 3):
- Pick `start_row` at the first bucket boundary where the following window covers **≥3 distinct attack subtypes** and has a **non-degenerate attack fraction** (roughly 5–60%, not ~0% and not ~100%).
- Pick `max_samples` to span enough buckets to include **multiple subtype transitions** (drift events), capping around 2,000,000 for 2018 (parity with 2017) and up to the file for UNSW (≈3,000,000; it is 3.48M rows total, ~97% benign, so include the whole attack-bearing region).
- Set `n_pretrain: 50000` (parity with 2017). Ensure `start_row + n_pretrain` still leaves a long evaluation tail within the chosen window.

- [ ] **Step 3: Write the profiles**

Create `configs/datasets/cicids2018.yaml` (fill `start_row`/`max_samples` from the scan; the values below are the structural template — replace the two numeric window values with the scanned choices):

```yaml
# CIC-IDS2018 profile (#4 external validity). CICFlowMeter feature family
# (same as 2017); natural bursty drift. Window chosen by scripts/scan_drift_window.py.
dataset: cicids2018
csv_path: /mnt/AI-DATA-SSD/alara/Data/CIC-IDS2018.csv
target_col: Label
benign_label: Benign
drop_cols: ["Timestamp"]
start_row: 0            # <- set from scan (first drift-rich bucket boundary)
max_samples: 2000000    # <- set from scan
n_pretrain: 50000
# monitor thresholds recalibrated in Task 8 (appended there)
```

Create `configs/datasets/unsw.yaml`:

```yaml
# UNSW-NB15 profile (#4 external validity). CICFlowMeter re-extraction; a
# genuinely different attack taxonomy. Window chosen by scripts/scan_drift_window.py.
dataset: unsw
csv_path: /mnt/AI-DATA-SSD/alara/Data/UNSW-NB15.csv
target_col: Label
benign_label: Benign
drop_cols: ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"]
start_row: 0            # <- set from scan
max_samples: 3000000    # <- set from scan
n_pretrain: 50000
# monitor thresholds recalibrated in Task 8 (appended there)
```

- [ ] **Step 4: Sanity-check the profiles load**

```bash
uv run python -c "from driftgan.config import load_config; \
c=load_config('configs/phase_b.yaml', dataset='cicids2018'); \
print(c.dataset, c.benign_label, c.start_row, c.max_samples, c.csv_path); \
c2=load_config('configs/phase_b.yaml', dataset='unsw'); \
print(c2.dataset, c2.benign_label, c2.drop_cols)"
```
Expected: prints the 2018 and UNSW fields resolved from the profiles.

- [ ] **Step 5: Commit**

```bash
git add scripts/scan_drift_window.py configs/datasets/cicids2018.yaml configs/datasets/unsw.yaml
git commit -m "feat(data): CIC-IDS2018 + UNSW-NB15 dataset profiles + drift-window scan"
```

---

### Task 8: Per-dataset monitor calibration (C_early smoke)

**Files:**
- Modify: `configs/datasets/cicids2018.yaml`
- Modify: `configs/datasets/unsw.yaml`

**Interfaces:**
- Produces: recalibrated `recon_pht_threshold` / `recon_pht_delta` in each profile so C_early's reconstruction monitor fires on real bursts (not never, not every window).

The `*_early` phase YAMLs set `recon_pht_threshold: 0.1` (calibrated on 2017). The reconstruction-error scale differs per dataset, so the dataset profile must override it (dataset layer wins over phase, per the merge precedence).

- [ ] **Step 1: Smoke-run C_early on 2018 with the ported threshold**

```bash
uv run python run_experiment.py --phase C_early --dataset cicids2018 --seeds 0 --max-samples 200000
```
Then inspect the anticipation report:
```bash
cat results/runs/cicids2018/C_early/seed_0/lead_time.json
cat results/runs/cicids2018/C_early/seed_0/summary.json
```

- [ ] **Step 2: Choose the threshold from the smoke result**

Interpretation rule:
- If `lead_time.json` shows **zero recon (anticipated) fires** → the threshold is too high for this dataset's error scale. Lower it: try `recon_pht_threshold: 0.05`, then `0.02`. Re-run Step 1 after each change until recon fires appear.
- If recon fires on **essentially every monitor check** (anticipated-FP rate near 1, or fires ≈ `max_samples / monitor_interval`) → too low. Raise it (e.g. `0.2`).
- Target: recon fires present, clustered near attack bursts, `n_drifts` in `summary.json` in a sane range (tens, not thousands, for 200K samples).

Append the chosen values to `configs/datasets/cicids2018.yaml`:

```yaml
# recalibrated for this dataset's reconstruction-error scale (Task 8)
recon_pht_threshold: 0.05   # <- set from the smoke run
recon_pht_delta: 0.0005
```

- [ ] **Step 3: Repeat for UNSW**

```bash
uv run python run_experiment.py --phase C_early --dataset unsw --seeds 0 --max-samples 200000
cat results/runs/unsw/C_early/seed_0/lead_time.json
```
Append the chosen `recon_pht_threshold` / `recon_pht_delta` to `configs/datasets/unsw.yaml` using the same rule.

- [ ] **Step 4: Record the calibration**

Add a one-line comment in each profile stating the smoke evidence (e.g. `# 200K smoke: N recon fires, mean lead ~K samples`). Update `docs/methodology.md` §6 with a sentence that monitor thresholds are recalibrated per dataset (same offline rule as 2017).

- [ ] **Step 5: Commit**

```bash
git add configs/datasets/cicids2018.yaml configs/datasets/unsw.yaml docs/methodology.md
git commit -m "feat(data): per-dataset monitor threshold calibration (2018, UNSW)"
```

---

### Task 9: Full run — CIC-IDS2018 (6 phases × 5 seeds)

**Files:** none (produces `results/runs/cicids2018/...`, `results/recovery/cicids2018_*.parquet`, `results/drift_schedule_cicids2018.json`).

**Runtime:** hours. Run each command to completion; monitor progress. Consider `--n-jobs` per available cores/GPU memory (SRP/ARF are heavy).

- [ ] **Step 1: Build the shared schedule (pass-0 Phase B dynamics)**

```bash
make schedule-2018 SEEDS=0
```
Expected: `results/drift_schedule_cicids2018.json` written, printing the event count. Sanity: the event count should be in a plausible range (tens–low hundreds), comparable to 2017's schedule scale.

- [ ] **Step 2: Run the 6 phases × 5 seeds**

```bash
make run-min-2018 SEEDS=0,1,2,3,4 NJOBS=1
```
This runs B, ARF_sched, SRP_sched (scheduled) then C_early, ARF_early, SRP_early (dynamic early).

- [ ] **Step 3: Verify outputs**

```bash
ls results/runs/cicids2018/*/seed_*/summary.json | wc -l   # expect 30 (6 phases x 5 seeds)
ls results/recovery/cicids2018_*.parquet | wc -l           # expect 30
```
Spot-check one `summary.json` per phase for sane `final_f1` and `n_drifts`.

- [ ] **Step 4: Commit any tracked changes** (results are gitignored; this is a checkpoint only if config comments were touched)

```bash
git add -A && git commit -m "run: CIC-IDS2018 minimal 2x2 (6 phases x 5 seeds)" --allow-empty
```

---

### Task 10: Full run — UNSW-NB15 (6 phases × 5 seeds)

**Files:** none (produces `results/runs/unsw/...`, `results/recovery/unsw_*.parquet`, `results/drift_schedule_unsw.json`).

**Runtime:** hours.

- [ ] **Step 1: Build the shared schedule**

```bash
make schedule-unsw SEEDS=0
```
Expected: `results/drift_schedule_unsw.json` written with a plausible event count. (UNSW is ~97% benign; if the schedule has very few events, note it — it reflects genuinely sparse drift and is still valid.)

- [ ] **Step 2: Run the 6 phases × 5 seeds**

```bash
make run-min-unsw SEEDS=0,1,2,3,4 NJOBS=1
```

- [ ] **Step 3: Verify outputs**

```bash
ls results/runs/unsw/*/seed_*/summary.json | wc -l   # expect 30
ls results/recovery/unsw_*.parquet | wc -l           # expect 30
```

- [ ] **Step 4: Checkpoint commit**

```bash
git add -A && git commit -m "run: UNSW-NB15 minimal 2x2 (6 phases x 5 seeds)" --allow-empty
```

---

### Task 11: Cross-dataset figure + summary table (fig12)

**Files:**
- Modify: `scripts/paper_figures.py`
- Create: `docs/figures/fig12_external_validity.png` (copied from `results/report/`)

**Interfaces:**
- Consumes: `results/runs/<dataset>/<phase>/seed_*/summary.json` and `results/recovery/<dataset>_<phase>_<seed>.parquet`.
- Produces: `fig12_external_validity` — the reactive→proactive Δ per base learner, grouped by dataset (2017 / 2018 / UNSW); and a printed cross-dataset summary table.

- [ ] **Step 1: Inspect the existing fig11 loader**

Read `scripts/paper_figures.py` around `fig11_early_ensemble` and the `RUNS`/`REC` path helpers and the overall-local-F1 helper (`_local_f1_by_segment` / trajectory loaders). Reuse the same overall-local-F1 metric and the same Okabe-Ito `COLOR`/`LABEL` dicts.

- [ ] **Step 2: Add a dataset-aware run loader**

Add a helper that resolves a run directory for `(dataset, phase, seed)`:

```python
def _run_dir(dataset: str, phase: str, seed: int) -> Path:
    seg = "" if dataset == "cicids2017" else f"{dataset}/"
    return RUNS / f"{seg}{phase}" / f"seed_{seed}"
```

and a mean-overall-local-F1 accessor per `(dataset, phase)` reusing the existing metric (the same one fig11 uses for the 2×2). If fig11 reads a precomputed local-F1 from `summary.json`/snapshots, mirror that exact computation here so the numbers are comparable to the existing figures.

- [ ] **Step 3: Add `fig12_external_validity`**

```python
def fig12_external_validity() -> None:
    """Reactive (*_sched) vs proactive (*_early) overall local F1 per base
    learner, across datasets — the generalization of experiment 3c."""
    datasets = ["cicids2017", "cicids2018", "unsw"]
    learners = [("HAT", "B", "C_early"),
                ("ARF", "ARF_sched", "ARF_early"),
                ("SRP", "SRP_sched", "SRP_early")]
    # ... build grouped bars: x = (dataset, learner), two bars (reactive, proactive),
    # annotate Δ. Skip a (dataset, learner) cell gracefully if runs are absent.
```

Implement grouped bars (dataset panels or grouped x-ticks), annotate the Δ from reactive to proactive per learner, and save to `OUT / "fig12_external_validity.png"` at 300 dpi. Also `print` a table of mean overall local F1 and mean adaptation count (`n_drifts`) per `(dataset, phase)` for the doc.

- [ ] **Step 4: Wire it into the figure `main()`**

Add `fig12_external_validity()` to the script's main entry (next to the `fig11` call).

- [ ] **Step 5: Generate + copy the figure**

```bash
uv run python scripts/paper_figures.py
cp results/report/figures/fig12_external_validity.png docs/figures/  # adjust to the script's actual OUT dir
```
(Confirm `OUT` in the script; copy from wherever it writes.)

- [ ] **Step 6: Lint + commit**

```bash
make lint typecheck
git add scripts/paper_figures.py docs/figures/fig12_external_validity.png
git commit -m "feat(figures): fig12 cross-dataset external-validity 2x2"
```

---

### Task 12: Experiment write-up + README/architecture update

**Files:**
- Create: `docs/experiments/05-external-validity.md`
- Modify: `docs/experiments/README.md` (index)
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Interfaces:** documentation only.

- [ ] **Step 1: Write `docs/experiments/05-external-validity.md`**

Match the existing per-experiment format (question / setup / results tables / figure / interpretation / caveats). Content:
- **Question:** does the proactive early trigger's 3c result (helps every base learner, fewer adaptations) hold on datasets beyond CICIDS2017?
- **Setup:** natural-drift track — CIC-IDS2018 + UNSW-NB15, CICFlowMeter family; the minimal 2×2 (B, C_early, ARF_sched, SRP_sched, ARF_early, SRP_early) × 5 seeds; per-dataset schedule and monitor calibration. Note the label-ordering evidence for dataset choice (IoT rejected: CICIoT2023 shuffled, Edge-IIoTset hard-blocked).
- **Results:** paste the per-(dataset, phase) table printed by Task 11 (overall local F1 + adaptation count); embed `../figures/fig12_external_validity.png`.
- **Interpretation:** state honestly whether the direction (proactive ≥ reactive per learner, fewer adaptations) replicates on each dataset, or where it does not.
- **Caveats:** UNSW ~97% benign (sparse drift, many invalid F1 windows); designed by re-extraction; thresholds recalibrated per dataset.

- [ ] **Step 2: Add the index row**

In `docs/experiments/README.md`, add a row for `05-external-validity.md` in the same style as the existing entries.

- [ ] **Step 3: Update `README.md` §8**

Add an "External validity" subsection under Results referencing 2018/UNSW and embedding/pointing to fig12 and doc 05. Keep the existing 2017 results intact.

- [ ] **Step 4: Update `docs/architecture.md`**

Add the dataset-profile slot to the pluggable-components description: `configs/datasets/<name>.yaml` + `--dataset` selects csv/target/benign/drop/window/thresholds; results namespaced by dataset; `cicids2017` stays flat.

- [ ] **Step 5: Final quality gate**

```bash
make lint typecheck test
```
Expected: all green (ruff clean, mypy clean, full test suite passing including `test_dataset_support.py`).

- [ ] **Step 6: Commit**

```bash
git add docs/experiments/05-external-validity.md docs/experiments/README.md README.md docs/architecture.md
git commit -m "docs: external-validity write-up (#4) + README/architecture update"
```

---

## Self-Review

**Spec coverage:**
- Natural-drift track (2017 anchor + 2018 + UNSW): Tasks 7–10. ✓
- Minimal 6-phase set: Tasks 6 (Makefile phase lists), 9, 10. ✓
- `benign_label` config, two hardcoded sites: Tasks 1, 2, 3, 5. ✓
- Per-dataset YAML profiles + merge layer (base ← phase ← dataset ← CLI): Tasks 3, 7. ✓
- Result namespacing (2017 stays flat): Tasks 3 (run_dir), 4 (recovery), verified in Task 6 Step 3. ✓
- Drift-window calibration: Task 7. ✓
- Monitor recalibration: Task 8. ✓
- Run protocol (pass-0 schedule + scheduled + dynamic-early): Tasks 9, 10. ✓
- fig12 + cross-dataset table + doc 05 + README/architecture: Tasks 11, 12. ✓
- Tests for profile merge + benign_label threading + namespacing: Tasks 1–4. ✓
- Risks (feature leakage via drop_cols, monitor firing, drift-poor window): addressed in Tasks 7 (drop_cols per dataset, scan) and 8 (monitor smoke). ✓

**Placeholder scan:** The two numeric window values in Task 7 (`start_row`, `max_samples`) and the monitor thresholds in Task 8 are **execution-time data-dependent values** with explicit selection rules and the exact commands that produce them — not logic placeholders. All code steps show complete code.

**Type consistency:** `benign_label: str` default `"BENIGN"` consistent across `preprocess_row`, `stream_csv`, `build_schedule_from_labels`, `ExperimentConfig`. `dataset: str` default `"cicids2017"` consistent across `ExperimentConfig`, `load_config(dataset=...)`, `save_recovery(dataset=...)`, `run_dir` namespacing, and the `--dataset` CLI default. `save_recovery` signature `(events, phase, seed, results_dir, dataset="cicids2017")` matches the runner call in Task 5.
