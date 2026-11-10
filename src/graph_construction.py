"""
graph_construction.py
---------------------
Person 1's responsibility.

Builds a 2D triangular mesh over [0,1]x[0,1] and converts it into
a PyTorch Geometric (PyG) Data object (a graph).

GNN Insight from PIGNN paper (Dalton et al. 2023):
  - Graph NODES  = FEM mesh nodes (each carries its (x,y) coordinate as features)
  - Graph EDGES  = FEM mesh edges (undirected → converted to two directed edges)
  - This lets the GNN "talk" between neighbouring nodes via message passing
  - No data needed — the graph structure encodes the domain geometry
"""

import numpy as np
import torch
from torch_geometric.data import Data


def build_rectangular_mesh(nx: int = 20, ny: int = 20):
    """
    Create a uniform triangular mesh over the unit square [0,1]x[0,1].

    Parameters
    ----------
    nx, ny : grid divisions along x and y

    Returns
    -------
    nodes       : (N, 2) float array  — (x, y) coordinates
    triangles   : (T, 3) int array    — vertex indices of each triangle
    bc_mask     : (N,) bool array     — True if node is on a boundary
    """
    # --- Node coordinates ---
    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])  # shape (N, 2)

    # --- Triangulation: each grid cell → 2 triangles ---
    triangles = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            triangles.append([n0, n1, n3])  # lower triangle
            triangles.append([n0, n3, n2])  # upper triangle
    triangles = np.array(triangles)        # shape (T, 3)

    # --- Boundary mask: nodes on x=0, x=1, y=0, or y=1 ---
    tol = 1e-10
    bc_mask = (
        (nodes[:, 0] < tol) | (nodes[:, 0] > 1 - tol) |
        (nodes[:, 1] < tol) | (nodes[:, 1] > 1 - tol)
    )

    return nodes, triangles, bc_mask


def mesh_to_graph(nodes: np.ndarray, triangles: np.ndarray, bc_mask: np.ndarray) -> Data:
    """
    Convert FEM mesh to a PyTorch Geometric graph.

    Node features:
      - x, y coordinates  (2 values)
      - is_boundary flag  (1 value)
      → feature dim = 3

    Edge features:
      - Δx, Δy, |Δr| from source to target node  (3 values)
      → feature dim = 3
    """
    N = len(nodes)

    # Build edge set from triangle connectivity (undirected → directed pairs)
    edge_set = set()
    for tri in triangles:
        a, b, c = tri
        for u, v in [(a, b), (b, c), (a, c)]:
            edge_set.add((min(u, v), max(u, v)))

    # Convert to directed edges (both directions)
    src_list, dst_list = [], []
    for u, v in edge_set:
        src_list += [u, v]
        dst_list += [v, u]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)  # (2, E)

    # Node features: [x, y, is_boundary]
    node_features = torch.tensor(
        np.column_stack([nodes, bc_mask.astype(float)]),
        dtype=torch.float32
    )  # (N, 3)

    # Edge features: [Δx, Δy, distance]
    src_coords = nodes[src_list]
    dst_coords = nodes[dst_list]
    delta = dst_coords - src_coords
    dist  = np.linalg.norm(delta, axis=1, keepdims=True)
    edge_features = torch.tensor(
        np.concatenate([delta, dist], axis=1),
        dtype=torch.float32
    )  # (E, 3)

    graph = Data(
        x          = node_features,     # (N, 3)
        edge_index = edge_index,        # (2, E)
        edge_attr  = edge_features,     # (E, 3)
        pos        = torch.tensor(nodes, dtype=torch.float32),  # (N, 2)
        bc_mask    = torch.tensor(bc_mask, dtype=torch.bool),   # (N,)
        num_nodes  = N,
    )
    # Attach triangle info (needed for physics loss integration)
    graph.triangles = torch.tensor(triangles, dtype=torch.long)  # (T, 3)

    return graph


if __name__ == "__main__":
    nodes, tris, bc = build_rectangular_mesh(nx=15, ny=15)
    g = mesh_to_graph(nodes, tris, bc)
    print("Graph summary:")
    print(f"  Nodes     : {g.num_nodes}")
    print(f"  Edges     : {g.edge_index.shape[1]}")
    print(f"  Triangles : {g.triangles.shape[0]}")
    print(f"  BC nodes  : {g.bc_mask.sum().item()}")
    print(f"  Node feat : {g.x.shape}")
    print(f"  Edge feat : {g.edge_attr.shape}")

