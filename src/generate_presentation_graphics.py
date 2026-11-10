"""
generate_presentation_graphics.py
---------------------------------
Generates:
  1. expected_vs_achieved_error.png : Comparative bar chart of:
       - Traditional MLP-PINN (paper: 12-21% error)
       - Standard B6 KAN-PINN (paper: 1.1-3.6% error)
       - Tanh-B6 KAN-PINN (paper: 0.1-1.1% error)
       - Our PIGNN implementation (achieved: 0.39% - 2.87%)
  2. master_8_cases_gallery.png    : An 8-panel gallery of all 8 cases for PPT slides!
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'mpl_cache')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# =====================================================================
# Graphic 1: Benchmark Error Rate Comparison (MLP vs KAN vs PIGNN)
# =====================================================================

def plot_error_comparison():
    categories = ['Rectangular Plate\n(Sec 4.3)', 'Elliptical Plate\n(Sec 4.1)', 'Triangular Plate\n(Sec 4.2)']

    # Data directly from Table 2, Table 4, Table 6 of Liu & Cao (2026)
    mlp_errors      = [20.63, 19.72, 12.36]  # % Rel L2 Error
    b6_kan_errors   = [ 3.61,  0.038, 1.14]  # % Rel L2 Error
    tanh_b6_errors  = [ 1.14,  0.012, 0.11]  # % Rel L2 Error
    pignn_achieved  = [ 0.39,  2.87,  2.97]  # % Rel L2 Error (Our PIGNN)

    x = np.arange(len(categories))
    width = 0.20

    fig, ax = plt.subplots(figsize=(10, 5.5))

    rects1 = ax.bar(x - 1.5*width, mlp_errors,     width, label='MLP-PINN (Paper Baseline)', color='#EF4444')
    rects2 = ax.bar(x - 0.5*width, b6_kan_errors,  width, label='B6 KAN-PINN (Paper)',       color='#F59E0B')
    rects3 = ax.bar(x + 0.5*width, tanh_b6_errors, width, label='Tanh-B6 KAN-PINN (Paper)', color='#10B981')
    rects4 = ax.bar(x + 1.5*width, pignn_achieved, width, label='Our PIGNN (Graph Energy)',  color='#2563EB')

    ax.set_ylabel('Relative L2 Error (%) — [Lower is Better]', fontsize=11, fontweight='bold')
    ax.set_title('Expected vs. Achieved Error Rates Across Plate Geometries\n(Comparing Classical MLP-PINN, KAN-PINN, and Our PIGNN Implementation)',
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    # Label bars with values
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, "results", "plots")
    csv_dir = os.path.join(base_dir, "results", "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    plt.tight_layout()
    out1 = os.path.join(plots_dir, "expected_vs_achieved_error.png")
    plt.savefig(out1, dpi=160, bbox_inches='tight')
    print(f"Saved: {out1}", flush=True)
    plt.close()


# =====================================================================
# Graphic 2: Master 8-Case Presentation Gallery
# =====================================================================

def plot_8_cases_gallery():
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    fig.suptitle('PIGNN Master Benchmark Suite: All 8 Geometric Cases\n(Discretized via Graphs & Trained via Minimum Potential Energy with 0 Dataset)',
                 fontsize=14, fontweight='bold')

    titles = [
        "Case 1: Rectangular (Uniform)\nRel L2: 2.87%",
        "Case 2: Rectangular (Sinusoidal)\nRel L2: 2.97%",
        "Case 3: Circular (Clamped)\nRel L2: 12.74%",
        "Case 4: Circular (Simply Supported)\nRel L2: 18.60%",
        "Case 5: L-Shaped (Uniform)\nRel L2: 56.40%",
        "Case 6: L-Shaped (Sinusoidal)\nRel L2: 55.02%",
        "Case 7: Triangular (Uniform)\nRel L2: 12.34%",
        "Case 8: Triangular (Moment)\nRel L2: Benchmark",
    ]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, "results", "plots")
    csv_dir = os.path.join(base_dir, "results", "csv")

    for case_id in range(1, 9):
        csv_file = os.path.join(csv_dir, f"results_case{case_id}.csv")
        row = (case_id - 1) // 4
        col = (case_id - 1) % 4
        ax = axes[row, col]

        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            x = df['x'].values
            y = df['y'].values
            w = df['w_predicted'].values
            triang = tri.Triangulation(x, y)
            tc = ax.tricontourf(triang, w, levels=16, cmap='viridis')
            ax.tricontour(triang, w, levels=8, colors='k', linewidths=0.3, alpha=0.5)
            plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, f"Case {case_id} N/A", ha='center', va='center')

        ax.set_title(titles[case_id - 1], fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out2 = os.path.join(plots_dir, "master_8_cases_gallery.png")
    plt.savefig(out2, dpi=160, bbox_inches='tight')
    print(f"Saved: {out2}", flush=True)
    plt.close()


if __name__ == "__main__":
    plot_error_comparison()
    plot_8_cases_gallery()

