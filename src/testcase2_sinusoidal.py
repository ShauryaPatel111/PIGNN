"""
testcase2_sinusoidal.py
-----------------------
Person 2's Testcase:
Rectangular Thin Plate under Sinusoidal Transverse Load
q(x, y) = q0 * sin(pi*x) * sin(pi*y)
(Section 4.3 of Liu & Cao 2026 paper)

Exact analytical solution:
w(x, y) = [q0 / (4 * pi^4)] * sin(pi*x) * sin(pi*y)
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import torch
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from graph_construction import build_rectangular_mesh, mesh_to_graph
from physics_loss import total_potential_energy
from gnn_model import PIGNN


def exact_sinusoidal(x: np.ndarray, y: np.ndarray, q0: float = 1.0) -> np.ndarray:
    """Exact analytical solution: w = [q0 / (4 * pi^4)] * sin(pi*x) * sin(pi*y)"""
    return (q0 / (4.0 * np.pi**4)) * np.sin(np.pi * x) * np.sin(np.pi * y)


def run_testcase2():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nx, ny = 16, 16
    h_approx = 1.0 / nx

    print(f"Running Testcase 2 (Sinusoidal Load)...", flush=True)
    nodes, triangles, bc_mask = build_rectangular_mesh(nx=nx, ny=ny)
    graph = mesh_to_graph(nodes, triangles, bc_mask).to(device)

    # Load distribution: q(x, y) = sin(pi*x) * sin(pi*y)
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    q_vals = np.sin(np.pi * x_coords) * np.sin(np.pi * y_coords)
    q_tensor = torch.tensor(q_vals, dtype=torch.float32, device=device)

    model = PIGNN(node_feat_dim=3, edge_feat_dim=3, hidden_dim=48, num_steps=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

    loss_history = []
    t0 = time.time()
    for epoch in range(1, 1001):
        model.train()
        optimizer.zero_grad()
        w_pred = model(graph)
        loss = total_potential_energy(w_pred, graph.pos, graph.triangles, q_tensor, h=h_approx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_history.append((epoch, loss.item()))
        if epoch % 200 == 0 or epoch == 1:
            print(f"Epoch {epoch:>4} | Loss: {loss.item():>12.6e} | Time: {time.time()-t0:>5.1f}s", flush=True)

    print(f"Training completed in {time.time()-t0:.1f}s", flush=True)

    # Evaluate
    model.eval()
    with torch.no_grad():
        w_pred_np = model(graph).cpu().numpy()

    w_exact = exact_sinusoidal(x_coords, y_coords)
    abs_err = np.abs(w_pred_np - w_exact)
    rel_l2  = np.linalg.norm(w_pred_np - w_exact) / np.linalg.norm(w_exact)

    print("\n" + "="*40, flush=True)
    print("TESTCASE 2 RESULTS (SINUSOIDAL LOAD)", flush=True)
    print("="*40, flush=True)
    print(f"Max predicted deflection: {w_pred_np.max():.6f}", flush=True)
    print(f"Analytical exact center : {exact_sinusoidal(np.array([0.5]), np.array([0.5]))[0]:.6f}", flush=True)
    print(f"Mean absolute error     : {abs_err.mean():.6e}", flush=True)
    print(f"Relative L2 error       : {rel_l2 * 100:.3f}%", flush=True)

    # Save CSV
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(base_dir, "results", "plots")
    csv_dir = os.path.join(base_dir, "results", "csv")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_file = os.path.join(csv_dir, "results_testcase2_sinusoidal.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "w_predicted", "w_analytical", "abs_error", "is_boundary"])
        for i in range(len(nodes)):
            writer.writerow([i, f"{x_coords[i]:.6f}", f"{y_coords[i]:.6f}",
                             f"{w_pred_np[i]:.8f}", f"{w_exact[i]:.8f}", f"{abs_err[i]:.8f}", int(bc_mask[i])])
    print(f"Saved: {csv_file}", flush=True)

    # Plot
    triang = tri.Triangulation(x_coords, y_coords)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("PIGNN Testcase 2 — Sinusoidal Load: q = sin(pi*x)*sin(pi*y)", fontsize=13, fontweight="bold")

    tc0 = axes[0].tricontourf(triang, w_pred_np, levels=20, cmap="viridis")
    plt.colorbar(tc0, ax=axes[0], label="w")
    axes[0].set_title("PIGNN Predicted w")
    axes[0].set_aspect("equal")

    tc1 = axes[1].tricontourf(triang, w_exact, levels=20, cmap="viridis")
    plt.colorbar(tc1, ax=axes[1], label="w")
    axes[1].set_title("Exact Navier w")
    axes[1].set_aspect("equal")

    tc2 = axes[2].tricontourf(triang, abs_err, levels=20, cmap="hot_r")
    plt.colorbar(tc2, ax=axes[2], label="|error|")
    axes[2].set_title(f"Absolute Error (Rel L2: {rel_l2*100:.2f}%)")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    out_plot = os.path.join(plots_dir, "deflection_testcase2_sinusoidal.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_plot}", flush=True)
    plt.close()


if __name__ == "__main__":
    run_testcase2()

