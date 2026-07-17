"""
Módulo C — Propuesta: Hoeffding Adaptive Tree + PHT + DriftGAN

Igual que el módulo B, pero al detectar un drift entrena una DriftGAN con el
buffer reciente de muestras de ataque y usa las variaciones sintéticas
generadas para reforzar el reentrenamiento del árbol antes/junto con las
muestras reales de la ventana de adaptación.
"""

import time
from collections import Counter, deque

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
from module_c_gan.drift_gan import DriftGAN

RECENT_BUFFER_SIZE = 500
GAN_NOISE_DIM = 16
GAN_EPOCHS = 30
GAN_SYNTHETIC_SAMPLES = 200
MIN_ATTACK_SAMPLES_FOR_GAN = 5


def run():
    print("=" * 70)
    print("  MÓDULO C: Propuesta — Hoeffding Adaptive Tree + PHT + DriftGAN")
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

    # ── Evaluación prequential con PHT y DriftGAN ──
    print("\n🔍 Evaluando modelo reactivo con PHT y DriftGAN...")
    training_enabled = False
    samples_to_train = 0
    recent_buffer = deque(maxlen=RECENT_BUFFER_SIZE)
    drift_gan = None

    for x, y, y_raw in stream:
        n_processed += 1
        label_dist[y_raw] += 1

        t0 = time.perf_counter()
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = "BENIGN"

        f1.update(y, y_pred)
        acc.update(y, y_pred)

        error = 0 if y_pred == y else 1
        pht.update(error)

        recent_buffer.append((x, y))

        if pht.drift_detected:
            n_drifts += 1
            training_enabled = True
            samples_to_train = TRAIN_WINDOW
            drift_points.append(n_processed)

            # ── Generación temprana de muestras sintéticas con DriftGAN ──
            attack_samples = [item[0] for item in recent_buffer if item[1] == "ATTACK"]
            num_syn = 0
            if len(attack_samples) >= MIN_ATTACK_SAMPLES_FOR_GAN:
                if drift_gan is None:
                    drift_gan = DriftGAN(input_dim=len(x), noise_dim=GAN_NOISE_DIM, epochs=GAN_EPOCHS)

                synthetic_attacks = drift_gan.train_and_generate(
                    attack_samples, num_samples=GAN_SYNTHETIC_SAMPLES
                )
                num_syn = len(synthetic_attacks)

                # Reentrenar el modelo inmediatamente con las variaciones sintéticas
                for syn_x in synthetic_attacks:
                    model.learn_one(syn_x, "ATTACK")

            if n_drifts <= 30:
                print(
                    f"  ⚠️  DRIFT {n_drifts} detectado en muestra {n_processed:,} -> "
                    f"GAN generó {num_syn} variaciones sintéticas"
                )

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
            status = "TRAIN+GAN" if training_enabled else "FROZEN"
            print(
                f"  [{n_processed:>9,}] Acc={acc.get():.4f} | F1(w)={f1.get():.4f} | "
                f"Drifts={n_drifts} | {status}"
            )

    print(f"\n✅ Módulo C completado — {n_processed:,} muestras procesadas")
    print(f"   Accuracy final: {acc.get():.4f}")
    print(f"   F1 rolling final: {f1.get():.4f}")
    print(f"   Total drifts: {n_drifts}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "module_c.npz"
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
