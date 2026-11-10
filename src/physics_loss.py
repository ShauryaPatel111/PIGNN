"""
physics_loss.py
---------------
Person 1's responsibility.

Computes the TOTAL POTENTIAL ENERGY Π of the thin plate.
This is used as the ONLY loss function — no simulation data needed!

Theory (from PIGNN paper Dalton et al. 2023 + TanB6 plate paper Liu & Cao 2026):
─────────────────────────────────────────────────────────────────────────────────
The Kirchhoff thin plate bending equation (non-dimensional form):

    ∇⁴w = q(x,y)       in Ω
    w = 0               on ∂Ω    (simply supported)
    ∂²w/∂n² = 0        on ∂Ω    (moment-free)

This PDE is equivalent to minimizing the Total Potential Energy:

    Π(w) = ½ ∬_Ω (∇²w)² dA  -  ∬_Ω q·w dA
           ───────────────────   ─────────────
           strain energy         external work

The GNN predicts w̃ at each mesh node. We evaluate Π(w̃) numerically
using triangle-wise integration (centroid rule). Backprop through Π
trains the GNN to find the displacement field that minimizes energy.
"""

import torch
import numpy as np


def triangle_areas(pos: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of each triangle.

    pos       : (N, 2) node positions
    triangles : (T, 3) vertex indices
    Returns   : (T,) areas
    """
    p0 = pos[triangles[:, 0]]  # (T, 2)
    p1 = pos[triangles[:, 1]]
    p2 = pos[triangles[:, 2]]
    # Cross product magnitude / 2
    v1 = p1 - p0
    v2 = p2 - p0
    area = 0.5 * torch.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return area  # (T,)


def laplacian_at_centroid(
    w: torch.Tensor,
    pos: torch.Tensor,
    triangles: torch.Tensor,
    h: float
) -> torch.Tensor:
    """
    Approximate ∇²w at the centroid of each triangle using
    a finite-difference stencil from the nearby nodes.

    For the centroid rule on triangular elements we use the
    average of second-order finite differences across the element.

    w         : (N,) predicted deflections at mesh nodes
    pos       : (N, 2) node x,y positions
    triangles : (T, 3) vertex indices
    h         : approximate mesh spacing (for FD denominator)

    Returns   : (T,) Laplacian values at triangle centroids
    """
    # Centroid = average of three vertex positions
    p0 = pos[triangles[:, 0]]
    p1 = pos[triangles[:, 1]]
    p2 = pos[triangles[:, 2]]

    w0 = w[triangles[:, 0]]
    w1 = w[triangles[:, 1]]
    w2 = w[triangles[:, 2]]
    w_c = (w0 + w1 + w2) / 3.0  # centroid deflection (avg)

    # Discrete Laplacian per element: sum of second differences
    # For each pair of vertices, compute curvature estimate
    def edge_laplacian(wa, pb, wb, pa):
        """Estimate d²w/ds² along edge a→b."""
        ds = torch.norm(pb - pa, dim=1) + 1e-12
        return (wb - 2 * w_c + wa) / (ds ** 2)

    lap_01 = edge_laplacian(w0, p1, w1, p0)
    lap_12 = edge_laplacian(w1, p2, w2, p1)
    lap_02 = edge_laplacian(w0, p2, w2, p0)

    lap = (lap_01 + lap_12 + lap_02) / 3.0  # average over three directions
    return lap  # (T,)


def total_potential_energy(
    w: torch.Tensor,
    pos: torch.Tensor,
    triangles: torch.Tensor,
    q_values: torch.Tensor,
    h: float = 0.0625
) -> torch.Tensor:
    """
    Compute the non-dimensional total potential energy of the plate:

        Π(w) = ½ ∬ (∇²w)² dA  -  ∬ q·w dA

    Evaluated via 5-point discrete Laplacian on the structured mesh grid:
      ∇²w_{i,j} = (w_{i+1,j} + w_{i-1,j} + w_{i,j+1} + w_{i,j-1} - 4w_{i,j}) / h²
    """
    # Infer grid size N_side = int(round(sqrt(len(w))))
    N = int(round(w.shape[0] ** 0.5))
    W = w.view(N, N)
    Q = q_values.view(N, N)

    # 5-point discrete Laplacian on interior nodes (simply supported BC: w=0 on boundaries)
    lap = (W[2:, 1:-1] + W[:-2, 1:-1] + W[1:-1, 2:] + W[1:-1, :-2] - 4.0 * W[1:-1, 1:-1]) / (h ** 2)

    dA = h * h
    strain_energy = 0.5 * torch.sum(lap ** 2) * dA
    external_work = torch.sum(Q[1:-1, 1:-1] * W[1:-1, 1:-1]) * dA

    return strain_energy - external_work


def pde_residual_loss(
    w: torch.Tensor,
    pos: torch.Tensor,
    triangles: torch.Tensor,
    q_values: torch.Tensor,
    h: float = 0.05
) -> torch.Tensor:
    """
    Alternative strong-form loss:  || ∇⁴w - q ||²

    For comparison with energy-based loss.
    Uses iterated Laplacian (Laplacian of Laplacian).
    """
    # First Laplacian at nodes (using graph-level averaging)
    lap1 = laplacian_at_centroid(w, pos, triangles, h)  # (T,)
    # Build node-level Laplacian by averaging over incident triangles
    N = w.shape[0]
    lap1_nodes = torch.zeros(N, device=w.device, dtype=w.dtype)
    count       = torch.zeros(N, device=w.device, dtype=w.dtype)
    for d in range(3):
        idx = triangles[:, d]
        lap1_nodes.scatter_add_(0, idx, lap1)
        count.scatter_add_(0, idx, torch.ones_like(lap1))
    lap1_nodes = lap1_nodes / (count + 1e-12)

    # Second Laplacian = biharmonic
    lap2 = laplacian_at_centroid(lap1_nodes, pos, triangles, h)  # (T,)
    residual = lap2 - (q_values[triangles[:, 0]] +
                       q_values[triangles[:, 1]] +
                       q_values[triangles[:, 2]]) / 3.0
    return torch.mean(residual ** 2)


if __name__ == "__main__":
    # Quick test with a known solution
    from graph_construction import build_rectangular_mesh
    nodes, tris, bc = build_rectangular_mesh(nx=10, ny=10)
    pos  = torch.tensor(nodes, dtype=torch.float32)
    tris = torch.tensor(tris,  dtype=torch.long)
    # Analytical solution: w = sin(πx)sin(πy)/2π⁴ for q=1
    x, y = nodes[:, 0], nodes[:, 1]
    w_exact = torch.tensor(
        np.sin(np.pi * x) * np.sin(np.pi * y) / (2 * np.pi**4),
        dtype=torch.float32
    )
    q = torch.ones(len(nodes), dtype=torch.float32)
    E = total_potential_energy(w_exact, pos, tris, q)
    print(f"Potential energy at exact solution: {E.item():.6f}")
    print("(Should be near its minimum — negative value from net external work)")

