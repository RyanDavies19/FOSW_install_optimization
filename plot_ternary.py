"""
Ternary plot of the weight simplex for the offshore wind installation optimizer.

Reads pareto_results.csv (exported from ERE291_OSW_Installation_ExampleCode.jl)
and produces three side-by-side subplots, each colored by a different output metric.
Saves to pareto_ternary.png.

Usage:
    python plot_ternary.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

df = pd.read_csv("pareto_results.csv")


def to_cart(w1, w2, w3):
    """Convert ternary weights to 2-D Cartesian coordinates.

    Corner mapping:
        w1 = 1  →  bottom-left  (0, 0)
        w2 = 1  →  bottom-right (1, 0)
        w3 = 1  →  top          (0.5, sqrt(3)/2)
    """
    total = w1 + w2 + w3  # should always be 1; kept for safety
    x = 0.5 * (2 * w2 + w3) / total
    y = (np.sqrt(3) / 2) * w3 / total
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


x_cart, y_cart = to_cart(df["w1"], df["w2"], df["w3"])

TRI = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
H   = np.sqrt(3) / 2  # triangle height

metrics = [
    ("time_days",    "Duration (days)", "Colored by\nInstallation Duration"),
    ("cost_k",       "Installation Cost ($K)", "Colored by\nInstallation Cost"),
    ("emissions_kg", "CO₂ (kg)",        "Colored by\nCO₂ Emissions"),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 7))

for ax, (col, cbar_label, title) in zip(axes, metrics):
    # Triangle boundary
    ax.add_patch(Polygon(TRI, closed=True, fill=False,
                         edgecolor="black", linewidth=1.5, zorder=3))

    # Scatter — all weight combinations, including duplicates
    sc = ax.scatter(x_cart, y_cart, c=df[col], cmap="plasma",
                    s=80, zorder=5, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8,
                 orientation="horizontal", location="bottom", pad=0.05)

    # Corner labels
    ax.text(-0.07, -0.04,    "Time\n(w₁)",       ha="right",  va="top",    fontsize=9, fontweight="bold")
    ax.text( 1.07, -0.04,    "Cost\n(w₂)",       ha="left",   va="top",    fontsize=9, fontweight="bold")
    ax.text( 0.5,  H + 0.05, "Emissions\n(w₃)",  ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.15, H + 0.25)
    ax.set_title(title, fontsize=11, pad=8)

fig.suptitle(
    "Weight Simplex: Optimal Objective Values Across the Full Weight Space",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
plt.savefig("pareto_ternary.png", dpi=300, bbox_inches="tight")
print("Saved pareto_ternary.png")
