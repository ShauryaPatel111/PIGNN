"""
visualize.py
------------
Person 2's responsibility.

Reads results.csv and loss_history.csv and generates publication-quality plots:

  1. Predicted deflection field (contour)
  2. Analytical deflection field (contour)
  3. Absolute error field (contour)
  4. Training loss curve
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os


def plot_all(results_csv: str = "results.csv",
             loss_csv:    str = "loss_history.csv"):

    # ── Load data ────────────────────────────────────────────
    if not os.path.exists(results_csv):
        print(f"ERROR: '{results_csv}' not found. Run train.py first.")
        sys.exit(1)

    df   = pd.read_csv(results_csv)
    x    = df["x"].values
    y    = df["y"].values
    w_p  = df["w_predicted"].values
    w_a  = df["w_analytical"].values
    err  = df["abs_error"].values

    # Build triangulation for plotting (Delaunay on the mesh nodes)
    triang = tri.Triangulation(x, y)

    # ── Figure 1: Deflection comparison ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("PIGNN — Simply-Supported Plate Bending (Uniform Load q₀=1)",
                 fontsize=13, fontweight="bold")

    vmin = min(w_p.min(), w_a.min())
    vmax = max(w_p.max(), w_a.max())

    # Predicted
    ax = axes[0]
    tc = ax.tricontourf(triang, w_p, levels=20, cmap="viridis",
                        vmin=vmin, vmax=vmax)
    ax.tricontour(triang, w_p, levels=10, colors="k", linewidths=0.3)
    plt.colorbar(tc, ax=ax, label="Deflection w")
    ax.set_title("PIGNN Predicted w")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Analytical
    ax = axes[1]
    tc = ax.tricontourf(triang, w_a, levels=20, cmap="viridis",
                        vmin=vmin, vmax=vmax)
    ax.tricontour(triang, w_a, levels=10, colors="k", linewidths=0.3)
    plt.colorbar(tc, ax=ax, label="Deflection w")
    ax.set_title("Analytical (Navier Series)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    # Absolute error
    ax = axes[2]
    tc = ax.tricontourf(triang, err, levels=20, cmap="hot_r")
    plt.colorbar(tc, ax=ax, label="|error|")
    ax.set_title("Absolute Error |w_pred − w_exact|")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("deflection_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: deflection_comparison.png")
    plt.close()

    # ── Figure 2: Training loss curve ────────────────────────
    if os.path.exists(loss_csv):
        df_loss = pd.read_csv(loss_csv)
        epochs  = df_loss["epoch"].values
        losses  = df_loss["loss"].values

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, losses, linewidth=1.5, color="#2563EB")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Potential Energy Π (Loss)")
        ax.set_title("PIGNN Training Loss — Physics Energy Minimisation\n"
                     "(No dataset used — physics equation is the supervisor)")
        ax.grid(True, alpha=0.3)
        # Highlight final loss
        ax.axhline(losses[-1], color="red", linestyle="--", linewidth=1,
                   label=f"Final loss: {losses[-1]:.5f}")
        ax.legend()
        plt.tight_layout()
        plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
        print("Saved: training_loss.png")
        plt.close()

    # ── Print summary stats ──────────────────────────────────
    interior = df[df["is_boundary"] == 0]
    print("\n" + "=" * 40)
    print("SUMMARY")
    print("=" * 40)
    print(f"Interior nodes        : {len(interior)}")
    print(f"Max predicted w       : {w_p.max():.6f}  at ({x[w_p.argmax()]:.3f}, {y[w_p.argmax()]:.3f})")
    print(f"Max analytical w      : {w_a.max():.6f}")
    print(f"Mean absolute error   : {err[df['is_boundary']==0].mean():.6f}")
    rel_l2 = np.linalg.norm(w_p - w_a) / (np.linalg.norm(w_a) + 1e-12)
    print(f"Relative L2 error     : {rel_l2*100:.3f}%")


if __name__ == "__main__":
    plot_all()

