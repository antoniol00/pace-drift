"""Style helpers shared by the visualization scripts of each module."""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PLOT_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
}


def new_three_panel_figure(title: str):
    """Creates the 3-panel figure (F1 / Accuracy / CPU Time) shared by all modules."""
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    return fig, axes


def finalize_and_save(fig, axes, save_path):
    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"\n✅ Plot saved to {save_path}")
