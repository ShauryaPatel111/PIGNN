"""
train.py
--------
Person 2's responsibility.

Trains the PIGNN to solve thin plate bending using ONLY the physics loss
(total potential energy). No simulation data is required!

Training procedure (from PIGNN paper):
  1. Build 2D mesh → PyG graph
  2. Initialize GNN (decoder weights = 0 → predicts w=0 initially)
  3. Minimize Π(w̃) = strain energy − external work
  4. Boundary conditions enforced hard inside the GNN forward pass
  5. Save predicted vs. analytical deflections to results.csv

Output files:
  - results.csv        : x, y, w_pred, w_analytical, abs_error
  - loss_history.csv   : epoch, loss
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

from graph_construction import build_rectangular_mesh, mesh_to_graph
from physics_loss import total_potential_energy
from gnn_model import PIGNN
from analytical import navier_solution


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

CONFIG = dict(
    nx           = 16,          # mesh divisions (16×16 = 289 nodes)
    ny           = 16,
    hidden_dim   = 48,          # GNN embedding size M
    num_steps    = 4,           # K message passing steps
    lr           = 1.5e-3,      # learning rate
    epochs       = 1000,        # training iterations
    q0           = 1.0,         # uniform load (non-dimensional)
    log_every    = 100,         # print loss every N epochs
    seed         = 42,
)


def train():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Build mesh & graph ───────────────────────────────────
    print(f"\nBuilding mesh ({CONFIG['nx']}×{CONFIG['ny']})...")
    nodes, triangles, bc_mask = build_rectangular_mesh(
        nx=CONFIG["nx"], ny=CONFIG["ny"]
    )
    graph = mesh_to_graph(nodes, triangles, bc_mask).to(device)

    pos       = graph.pos                        # (N, 2)
    tris      = graph.triangles                  # (T, 3)
    bc        = graph.bc_mask                    # (N,) bool

    # Uniform load at all nodes
    q = torch.full((graph.num_nodes,), CONFIG["q0"], device=device)

    # Mesh spacing estimate for FD Laplacian
    h_approx = 1.0 / CONFIG["nx"]

    # ── Build model ──────────────────────────────────────────
    model = PIGNN(
        node_feat_dim = 3,
        edge_feat_dim = 3,
        hidden_dim    = CONFIG["hidden_dim"],
        num_steps     = CONFIG["num_steps"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-5
    )

    # ── Training loop ────────────────────────────────────────
    print("\nTraining (physics-informed, no dataset needed)...")
    print(f"{'Epoch':>8} | {'Loss (Pi)':>14} | {'LR':>10} | {'Time(s)':>8}")
    print("-" * 50)

    loss_history = []
    t0 = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass: GNN predicts w at all nodes
        w_pred = model(graph)

        # Physics loss: total potential energy
        loss = total_potential_energy(w_pred, pos, tris, q, h=h_approx)

        # Backprop
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_history.append((epoch, loss_val))

        if epoch % CONFIG["log_every"] == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"{epoch:>8} | {loss_val:>14.6f} | {lr_now:>10.2e} | {elapsed:>8.1f}", flush=True)

    total_time = time.time() - t0
    print(f"\nTraining complete in {total_time:.1f}s")

    # ── Evaluate & compare with analytical solution ──────────
    model.eval()
    with torch.no_grad():
        w_pred_np = model(graph).cpu().numpy()

    xy = nodes  # (N, 2)
    w_analytical = navier_solution(xy[:, 0], xy[:, 1], n_terms=30)

    abs_error = np.abs(w_pred_np - w_analytical)
    rel_l2 = np.linalg.norm(w_pred_np - w_analytical) / (np.linalg.norm(w_analytical) + 1e-12)

    print("\n" + "=" * 40)
    print("RESULTS SUMMARY")
    print("=" * 40)
    print(f"Max predicted deflection: {w_pred_np.max():.6f}")
    print(f"Analytical centre value : {navier_solution(np.array([0.5]), np.array([0.5]))[0]:.6f}")
    print(f"Mean absolute error     : {abs_error.mean():.6f}")
    print(f"Relative L2 error       : {rel_l2 * 100:.3f}%")

    # ── Save results.csv ─────────────────────────────────────
    results_path = "results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id", "x", "y",
            "w_predicted", "w_analytical", "abs_error",
            "is_boundary"
        ])
        for i in range(len(nodes)):
            writer.writerow([
                i,
                f"{xy[i, 0]:.6f}",
                f"{xy[i, 1]:.6f}",
                f"{w_pred_np[i]:.8f}",
                f"{w_analytical[i]:.8f}",
                f"{abs_error[i]:.8f}",
                int(bc_mask[i])
            ])
    print(f"\nSaved: {results_path}")

    # ── Save loss_history.csv ────────────────────────────────
    loss_path = "loss_history.csv"
    with open(loss_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss"])
        for ep, lv in loss_history:
            writer.writerow([ep, f"{lv:.8f}"])
    print(f"Saved: {loss_path}", flush=True)

    # ── Auto-generate plots ──────────────────────────────────
    try:
        from visualize import plot_all
        print("\nGenerating visualization plots...", flush=True)
        plot_all(results_path, loss_path)
    except Exception as e:
        print(f"Plot generation notice: {e}", flush=True)

    return w_pred_np, w_analytical, loss_history


if __name__ == "__main__":
    train()
