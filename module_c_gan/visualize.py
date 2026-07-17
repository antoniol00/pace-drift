"""Visualización aislada del módulo C (F1 / Accuracy / Tiempo de CPU)."""

import numpy as np

from common.config import RESULTS_DIR
from common.plotting import finalize_and_save, new_three_panel_figure


def run():
    data = np.load(RESULTS_DIR / "module_c.npz")
    snap_x = data["snap_x"]
    snap_f1 = data["snap_f1"]
    snap_acc = data["snap_acc"]
    snap_time = data["snap_time"]
    drift_points = data["drift_points"]
    n_pretrain = int(data["n_pretrain"])

    fig, axes = new_three_panel_figure("Concept Drift PoC — Módulo C (HAT + PHT + DriftGAN)")

    ax1, ax2, ax3 = axes

    ax1.plot(snap_x, snap_f1, color="#8e44ad", linewidth=1.5, alpha=0.85, label="Módulo C — Reactivo con DriftGAN")
    for i, dp in enumerate(drift_points[:30]):
        ax1.axvline(x=dp, color="#f39c12", alpha=0.35, linewidth=1.0, linestyle="--",
                    label="Drift detectado" if i == 0 else None)
    ax1.axvline(x=n_pretrain, color="#3498db", alpha=0.6, linewidth=1.5, linestyle=":",
                label="Fin pre-entrenamiento")
    ax1.set_ylabel("F1 Score (ventana deslizante)")
    ax1.set_title("Evolución del F1-Score (Ventana = 5,000)")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(-0.05, 1.05)

    ax2.plot(snap_x, snap_acc, color="#8e44ad", linewidth=1.5, alpha=0.85, label="Módulo C — Reactivo con DriftGAN")
    ax2.axvline(x=n_pretrain, color="#3498db", alpha=0.6, linewidth=1.5, linestyle=":",
                label="Fin pre-entrenamiento")
    ax2.set_xlabel("Nº de muestras totales (acumulativo desde fila 0)")
    ax2.set_ylabel("Accuracy (acumulativa)")
    ax2.set_title("Evolución de la Accuracy a lo largo del Stream")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.set_ylim(0.5, 1.02)

    ax3.plot(snap_x, snap_time, color="#8e44ad", lw=1.5, alpha=0.85, label="Módulo C")
    ax3.axvline(x=n_pretrain, color="#3498db", alpha=0.6, lw=1.5, ls=":", label="Fin pre-train")
    ax3.set_xlabel("Muestras")
    ax3.set_ylabel("Tiempo de CPU (ms / 500 obj)")
    ax3.legend(loc="upper right")

    finalize_and_save(fig, axes, RESULTS_DIR / "module_c_results.png")


if __name__ == "__main__":
    run()
