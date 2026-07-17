"""
Módulo B — Reactivo: Hoeffding Adaptive Tree + Page-Hinkley Test (PHT)

El detector PHT monitoriza la señal de error. Cuando detecta un drift, el
modelo "despierta" y reentrena (learn_one) durante una ventana de
TRAIN_WINDOW muestras antes de volver a congelarse. Es la referencia
reactiva sin generación sintética, frente a la que se compara la propuesta
del módulo C.
"""

import time
from collections import Counter

import numpy as np
from river import drift, metrics, tree

from common.config import (
    CSV_PATH,
    GRACE_PERIOD,
    LOG_INTERVAL,
    MAX_SAMPLES,
    N_PRETRAIN,
    PHT_DELTA,
    RESULTS_DIR,
    ROLLING_F1_WINDOW,
    SEED,
    SNAP_INTERVAL,
    START_ROW,
    TRAIN_WINDOW,
)
from common.metrics import RollingF1
from common.streaming import stream_csv


def run():
    print("=" * 70)
    print("  MÓDULO B: Reactivo — Hoeffding Adaptive Tree + PHT")
    print("=" * 70)

    model = tree.HoeffdingAdaptiveTreeClassifier(grace_period=GRACE_PERIOD, seed=SEED)
    pht = drift.PageHinkley(delta=PHT_DELTA)
    f1 = RollingF1(window_size=ROLLING_F1_WINDOW)
    acc = metrics.Accuracy()

    snap_x, snap_f1, snap_acc, snap_time = [], [], [], []
    interval_time = 0.0
    drift_points = []
    label_dist = Counter()
    n_processed = 0
    n_drifts = 0

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

    # ── Evaluación prequential con PHT ──
    print("\n🔍 Evaluando modelo reactivo con PHT...")
    training_enabled = False
    samples_to_train = 0
    for x, y, y_raw in stream:
        n_processed += 1
        label_dist[y_raw] += 1

        # 1) Predecir
        t0 = time.perf_counter()
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = "BENIGN"

        # 2) Actualizar métricas
        f1.update(y, y_pred)
        acc.update(y, y_pred)

        # 3) PHT monitoriza el error
        error = 0 if y_pred == y else 1
        pht.update(error)

        # 4) Detección de drift
        if pht.drift_detected:
            n_drifts += 1
            training_enabled = True
            samples_to_train = TRAIN_WINDOW
            drift_points.append(n_processed)
            if n_drifts <= 30:
                print(f"  ⚠️  DRIFT detectado en muestra {n_processed:,} (total: {n_drifts})")

        # 5) Aprender si se ha detectado drift
        if training_enabled:
            model.learn_one(x, y)
            samples_to_train -= 1
            if samples_to_train <= 0:
                training_enabled = False

        interval_time += time.perf_counter() - t0

        if n_processed % SNAP_INTERVAL == 0:
            snap_x.append(n_processed)
            snap_f1.append(f1.get())
            snap_acc.append(acc.get())
            snap_time.append(interval_time * 1000)
            interval_time = 0.0

        if n_processed % LOG_INTERVAL == 0:
            status = "TRAIN" if training_enabled else "FROZEN"
            print(
                f"  [{n_processed:>9,}] Acc={acc.get():.4f} | F1(w)={f1.get():.4f} | "
                f"Drifts={n_drifts} | {status}"
            )

    print(f"\n✅ Módulo B completado — {n_processed:,} muestras procesadas")
    print(f"   Accuracy final: {acc.get():.4f}")
    print(f"   F1 rolling final: {f1.get():.4f}")
    print(f"   Total drifts: {n_drifts}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "module_b.npz"
    np.savez(
        out_path,
        snap_x=snap_x,
        snap_f1=snap_f1,
        snap_acc=snap_acc,
        snap_time=snap_time,
        drift_points=drift_points,
        n_pretrain=N_PRETRAIN,
    )
    print(f"✅ Resultados guardados en {out_path}")


if __name__ == "__main__":
    run()
