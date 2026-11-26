import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MODE_COLOR = {"baseline": "#2a78d6", "augmented": "#eb6834"}
MODE_MARKER = {"baseline": "o", "augmented": "s"}
K_COLOR = {1: "#9ec5f4", 2: "#5598e7", 4: "#256abf", 7: "#0d366b"}
OUTCOME_COLOR = {"correct": "#2a78d6", "wrong": "#eb6834"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlecolor": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "lines.markersize": 6, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False, "savefig.dpi": 160,
    "savefig.bbox": "tight",
})


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path
