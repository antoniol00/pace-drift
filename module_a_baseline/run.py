"""
Módulo A — Baseline: Modelo Estático (Hoeffding Adaptive Tree)

Se pre-entrena una única vez y después solo predice (sin actualización)
durante el resto del stream. Sirve de referencia inferior para medir cuánto
degrada el rendimiento un modelo "congelado" ante concept drift.
"""

import time
from collections import Counter

import numpy as np
from river import metrics, tree

from common.config import (
    CSV_PATH,
    GRACE_PERIOD,
    LOG_INTERVAL,
    MAX_SAMPLES,
    N_PRETRAIN,
    RESULTS_DIR,
    ROLLING_F1_WINDOW,
    SEED,
    SNAP_INTERVAL,
    START_ROW,
)
from common.metrics import RollingF1
from common.streaming import stream_csv


def run():
    print("=" * 70)
    print("  MÓDULO A: Baseline — Modelo Estático (Hoeffding Adaptive Tree)")
    print("=" * 70)

    model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=GRACE_PERIOD, seed=SEED)
    f1 = RollingF1(window_size=ROLLING_F1_WINDOW)
    acc = metrics.Accuracy()

    snap_x, snap_f1, snap_acc, snap_time = [], [], [], []
    interval_time = 0.0
    label_dist = Counter()
    n_processed = 0

    stream = stream_csv(CSV_PATH, start_row=START_ROW, max_rows=MAX_SAMPLES)

    # ── Pre-entrenamiento ──
    print(f"\n🔄 Pre-entrenando con {N_PRETRAIN:,} muestras...")
    for x, y, y_raw in stream:
        model.learn_one(x, y)
        label_dist[y_raw] += 1
        n_processed += 1
        if n_processed >= N_PRETRAIN:
            break
    print(f"✅ Pre-entrenamiento completado ({n_processed:,} muestras)")
    print(f"   Distribución inicial: {dict(label_dist)}")

    # ── Evaluación prequential (Test-Only) ──
    print("\n🔍 Evaluando modelo estático (Test-Only)...")
    for x, y, y_raw in stream:
        n_processed += 1
        label_dist[y_raw] += 1

        t0 = time.perf_counter()
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = "BENIGN"

        f1.update(y, y_pred)
        acc.update(y, y_pred)
        interval_time += time.perf_counter() - t0

        if n_processed % SNAP_INTERVAL == 0:
            snap_x.append(n_processed)
            snap_f1.append(f1.get())
            snap_acc.append(acc.get())
            snap_time.append(interval_time * 1000)
            interval_time = 0.0

        if n_processed % LOG_INTERVAL == 0:
            print(
                f"  [{n_processed:>9,}] Acc={acc.get():.4f} | F1(w)={f1.get():.4f} | "
                f"Dist: {dict(label_dist)}"
            )

    print(f"\n✅ Módulo A completado — {n_processed:,} muestras procesadas")
    print(f"   Accuracy final: {acc.get():.4f}")
    print(f"   F1 rolling final: {f1.get():.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "module_a.npz"
    np.savez(
        out_path,
        snap_x=snap_x,
        snap_f1=snap_f1,
        snap_acc=snap_acc,
        snap_time=snap_time,
        n_pretrain=N_PRETRAIN,
    )
    print(f"✅ Resultados guardados en {out_path}")


if __name__ == "__main__":
    run()
