"""
Comparativa Módulo A (baseline) vs Módulo B (PHT) vs Módulo C (propuesta: PHT + DriftGAN).

Requiere haber ejecutado antes module_a_baseline/run.py, module_b_pht/run.py
y module_c_gan/run.py (cada uno guarda su resultado en results/module_*.npz).
"""

import numpy as np

from common.config import RESULTS_DIR
from common.plotting import finalize_and_save, new_three_panel_figure


def run():
    data_a = np.load(RESULTS_DIR / "module_a.npz")
    data_b = np.load(RESULTS_DIR / "module_b.npz")
    data_c = np.load(RESULTS_DIR / "module_c.npz")

    n_pretrain = int(data_a["n_pretrain"])
    drift_points_b = data_b["drift_points"]

    fig, axes = new_three_panel_figure("Concept Drift PoC — Comparativa Módulos A, B y C")
    ax1, ax2, ax3 = axes

    series = [
        (data_a, "#e74c3c", "Módulo A — Baseline (Estático)"),
        (data_b, "#2ecc71", "Módulo B — Reactivo (PHT)"),
        (data_c, "#8e44ad", "Módulo C — Propuesta (PHT + DriftGAN)"),
    ]

    for data, color, label in series:
        ax1.plot(data["snap_x"], data["snap_f1"], color=color, linewidth=1.5, alpha=0.85, label=label)
    for i, dp in enumerate(drift_points_b[:20]):
        ax1.axvline(x=dp, color="#f39c12", alpha=0.35, linewidth=1.0, linestyle="--",
                    label="Drift detectado" if i == 0 else None)
    ax1.axvline(x=n_pretrain, color="#3498db", alpha=0.6, linewidth=1.5, linestyle=":",
                label="Fin pre-entrenamiento")
    ax1.set_ylabel("F1 Score (ventana deslizante)")
    ax1.set_title("Evolución del F1-Score (Ventana = 5,000)")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)

    for data, color, label in series:
        ax2.plot(data["snap_x"], data["snap_acc"], color=color, linewidth=1.5, alpha=0.85, label=label)
    ax2.axvline(x=n_pretrain, color="#3498db", alpha=0.6, linewidth=1.5, linestyle=":",
                label="Fin pre-entrenamiento")
    ax2.set_xlabel("Nº de muestras totales (acumulativo)")
    ax2.set_ylabel("Accuracy (acumulativa)")
    ax2.set_title("Evolución de la Accuracy a lo largo del Stream")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_ylim(0.5, 1.02)

    for data, color, label in series:
        ax3.plot(data["snap_x"], data["snap_time"], color=color, lw=1.5, alpha=0.85, label=label)
    ax3.axvline(x=n_pretrain, color="#3498db", alpha=0.6, lw=1.5, ls=":", label="Fin pre-train")
    ax3.set_xlabel("Muestras")
    ax3.set_ylabel("Tiempo de CPU (ms / 500 obj)")
    ax3.legend(loc="upper right")

    finalize_and_save(fig, axes, RESULTS_DIR / "comparison.png")


if __name__ == "__main__":
    run()
