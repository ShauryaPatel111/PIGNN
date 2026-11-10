"""
generate_architecture_diagram.py
---------------------------------
Generates a detailed architecture diagram of the PIGNN model
showing all layers, dimensions, activations, and parameter counts.
Total params: 199,777
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

os.environ['MPLCONFIGDIR'] = os.path.join(os.environ.get('TEMP', '.'), 'mpl_cache')

# ─── Color palette ───────────────────────────────────────────────
C_INPUT    = "#1E3A5F"   # dark blue  - inputs
C_ENCODER  = "#1565C0"   # blue       - encoders
C_LINEAR   = "#0288D1"   # light blue - linear layers
C_ACTIV    = "#F57F17"   # amber      - activations
C_LN       = "#2E7D32"   # green      - layer norm
C_MSG      = "#6A1B9A"   # purple     - message passing block
C_PROC     = "#7B1FA2"   # purple2    - processor header
C_DECODER  = "#C62828"   # red        - decoder
C_BC       = "#37474F"   # dark grey  - hard BC
C_OUTPUT   = "#1B5E20"   # dark green - output
C_ARROW    = "#455A64"
C_BG       = "#FAFAFA"
C_PANEL    = "#ECEFF1"
TEXT_W     = "white"
TEXT_D     = "#212121"

FIG_W, FIG_H = 28, 20

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ─── Helper functions ────────────────────────────────────────────

def box(ax, x, y, w, h, color, text_lines, fontsize=8.5, text_color=TEXT_W,
        alpha=1.0, radius=0.3, bold_first=True):
    """Draw a rounded rectangle with multi-line text."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0.05,rounding_size={radius}",
                          facecolor=color, edgecolor="white",
                          linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    # center text
    cx, cy = x + w/2, y + h/2
    n = len(text_lines)
    spacing = min(h / (n + 1), 0.38)
    for i, line in enumerate(text_lines):
        ty = cy + (n - 1) / 2 * spacing - i * spacing
        weight = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(cx, ty, line, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight=weight, zorder=4,
                fontfamily="DejaVu Sans")

def arrow(ax, x1, y1, x2, y2, label="", color=C_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.1, my, label, fontsize=7, color=color, va="center", zorder=6)

def section_bg(ax, x, y, w, h, color=C_PANEL, label="", fontsize=9):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.1,rounding_size=0.4",
                          facecolor=color, edgecolor="#B0BEC5",
                          linewidth=1.5, alpha=0.5, zorder=1)
    ax.add_patch(rect)
    if label:
        ax.text(x + w/2, y + h - 0.25, label,
                ha="center", va="top", fontsize=fontsize,
                fontweight="bold", color="#37474F", zorder=2)

# ═══════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════
ax.text(FIG_W/2, 19.6,
        "PIGNN Architecture — Physics-Informed Graph Neural Network",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#1A237E",
        zorder=6)
ax.text(FIG_W/2, 19.1,
        "Encode → Process (×5 Message Passing Steps) → Decode    |    Total Parameters: 199,777",
        ha="center", va="center", fontsize=11, color="#37474F", zorder=6)

# ═══════════════════════════════════════════════════════════════
# SECTION BACKGROUNDS
# ═══════════════════════════════════════════════════════════════
# INPUT block
section_bg(ax, 0.3, 14.5, 3.5, 4.0, "#E3F2FD", "INPUT\nGraph G=(V,E)", 9)
# ENCODER block
section_bg(ax, 4.2, 14.5, 7.0, 4.0, "#E8EAF6", "ENCODER", 9)
# PROCESSOR
section_bg(ax, 11.7, 8.5, 12.5, 10.5, "#F3E5F5", "PROCESSOR  (K=5 Identical Message-Passing Blocks)", 9)
# DECODER + OUTPUT
section_bg(ax, 0.3, 8.5, 11.0, 5.5, "#FCE4EC", "DECODER + OUTPUT", 9)

# ═══════════════════════════════════════════════════════════════
# INPUT SECTION  (left column, top)
# ═══════════════════════════════════════════════════════════════
# Node features
box(ax, 0.5, 17.4, 3.0, 0.8, C_INPUT,
    ["NODE FEATURES  (N × 3)",
     "x  |  y  |  is_boundary"],
    fontsize=8)

# Edge features
box(ax, 0.5, 16.3, 3.0, 0.8, C_INPUT,
    ["EDGE FEATURES  (E × 3)",
     "Δx  |  Δy  |  distance"],
    fontsize=8)

# Mesh info box
box(ax, 0.5, 15.1, 3.0, 0.8, "#37474F",
    ["MESH  (Delaunay triangulation)",
     "Rect: N≈225 nodes  |  E≈1,220 edges",
     "bc_mask  (N,) bool — boundary nodes"],
    fontsize=7.5)

# ═══════════════════════════════════════════════════════════════
# NODE ENCODER  MLP(3→64→64) + LayerNorm
# ═══════════════════════════════════════════════════════════════
# Title
ax.text(7.7, 18.55, "NODE ENCODER", ha="center", fontsize=8.5,
        fontweight="bold", color="#1A237E", zorder=5)
ax.text(7.7, 18.3, "MLP(3→64→64, 2 layers)  |  4,544 params", ha="center",
        fontsize=7.5, color="#37474F", zorder=5)

# layer 1
box(ax, 4.4, 17.3, 2.0, 0.65, C_LINEAR,
    ["Linear(3 → 64)", "W:[64×3]+b:[64]", "params: 256"], fontsize=7.5)
# SiLU 1
box(ax, 6.6, 17.3, 1.1, 0.65, C_ACTIV,
    ["SiLU", "activation"], fontsize=7.5)
# layer 2
box(ax, 7.9, 17.3, 2.0, 0.65, C_LINEAR,
    ["Linear(64 → 64)", "W:[64×64]+b:[64]", "params: 4,160"], fontsize=7.5)
# LayerNorm
box(ax, 10.1, 17.3, 1.0, 0.65, C_LN,
    ["LN(64)", "128 p"], fontsize=7.5)

# arrows node encoder
arrow(ax, 3.5, 17.78, 4.4, 17.63)
arrow(ax, 6.4, 17.63, 6.6, 17.63)
arrow(ax, 7.7, 17.63, 7.9, 17.63)
arrow(ax, 9.9, 17.63, 10.1, 17.63)

# ─── output label
ax.text(11.3, 17.63, "→ node_emb\n   (N × 64)", ha="left", va="center",
        fontsize=7.5, color="#1A237E", zorder=5)

# ═══════════════════════════════════════════════════════════════
# EDGE ENCODER  MLP(3→64→64) + LayerNorm
# ═══════════════════════════════════════════════════════════════
ax.text(7.7, 16.55, "EDGE ENCODER", ha="center", fontsize=8.5,
        fontweight="bold", color="#1A237E", zorder=5)
ax.text(7.7, 16.3, "MLP(3→64→64, 2 layers)  |  4,544 params", ha="center",
        fontsize=7.5, color="#37474F", zorder=5)

box(ax, 4.4, 15.3, 2.0, 0.65, C_LINEAR,
    ["Linear(3 → 64)", "W:[64×3]+b:[64]", "params: 256"], fontsize=7.5)
box(ax, 6.6, 15.3, 1.1, 0.65, C_ACTIV,
    ["SiLU", "activation"], fontsize=7.5)
box(ax, 7.9, 15.3, 2.0, 0.65, C_LINEAR,
    ["Linear(64 → 64)", "W:[64×64]+b:[64]", "params: 4,160"], fontsize=7.5)
box(ax, 10.1, 15.3, 1.0, 0.65, C_LN,
    ["LN(64)", "128 p"], fontsize=7.5)

arrow(ax, 3.5, 16.7,  4.4, 15.63)
arrow(ax, 6.4, 15.63, 6.6, 15.63)
arrow(ax, 7.7, 15.63, 7.9, 15.63)
arrow(ax, 9.9, 15.63, 10.1, 15.63)
ax.text(11.3, 15.63, "→ edge_emb\n   (E × 64)", ha="left", va="center",
        fontsize=7.5, color="#1A237E", zorder=5)

# ═══════════════════════════════════════════════════════════════
# PROCESSOR BLOCK (one block detail shown, ×5 repeated)
# ═══════════════════════════════════════════════════════════════

# Connecting arrows from encoders into processor
arrow(ax, 11.2, 17.63, 12.0, 15.9, label="node_emb (N×64)", color="#6A1B9A")
arrow(ax, 11.2, 15.63, 12.0, 15.6, label="edge_emb (E×64)", color="#6A1B9A")

# --- STEP LABEL
ax.text(23.7, 18.7, "× 5 steps\n(K=5)", ha="center", va="center",
        fontsize=12, fontweight="bold", color="#6A1B9A", zorder=6)

# Step counter boxes
for k in range(5):
    bx = 11.9 + k * 2.35
    by = 18.3
    col = C_PROC if k == 0 else "#AB47BC"
    box(ax, bx, by, 2.0, 0.6, col,
        [f"Step {k+1}  |  37,504 params"], fontsize=8)
    if k < 4:
        arrow(ax, bx+2.0, by+0.3, bx+2.35, by+0.3)

# ─── DETAIL BOX for one MessagePassingBlock ─────────────────────
# background
section_bg(ax, 11.9, 8.9, 12.0, 9.2, "#EDE7F6", "", 8)
ax.text(17.9, 17.9, "MESSAGE PASSING BLOCK  (detail for each of the 5 steps)",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#4A148C", zorder=6)
ax.text(17.9, 17.6, "Per block: 37,504 params  |  5 blocks → 187,520 params",
        ha="center", va="center", fontsize=8, color="#6A1B9A", zorder=6)

# --- INPUT CONCAT for EDGE MLP
box(ax, 12.1, 16.6, 2.3, 0.8, "#CE93D8",
    ["CONCAT for edge_mlp",
     "[edge_emb | node_emb[src] | node_emb[dst]]",
     "shape: E × (64+64+64) = E × 192"],
    fontsize=7.5, text_color=TEXT_D)

# --- EDGE MLP
ax.text(17.9, 16.4, "EDGE MLP   MLP(192→64→64→64, 3 layers)  |  20,800 params",
        ha="center", fontsize=8.5, fontweight="bold", color="#4A148C", zorder=5)
# layers
box(ax, 14.6, 15.5, 2.1, 0.75, C_LINEAR,
    ["Linear(192→64)", "W:[64×192]+b:[64]", "12,352 params"], fontsize=7.5)
box(ax, 16.9, 15.5, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 18.2, 15.5, 2.1, 0.75, C_LINEAR,
    ["Linear(64→64)", "W:[64×64]+b:[64]", "4,160 params"], fontsize=7.5)
box(ax, 20.5, 15.5, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 21.8, 15.5, 2.1, 0.75, C_LINEAR,
    ["Linear(64→64)", "W:[64×64]+b:[64]", "4,160 params"], fontsize=7.5)
box(ax, 21.8, 14.65, 1.0, 0.65, C_LN, ["LN(64)", "128 p"], fontsize=7.5)

arrow(ax, 14.4, 17.0, 14.65, 15.87)   # from concat
arrow(ax, 16.7, 15.87, 16.9, 15.87)
arrow(ax, 18.0, 15.87, 18.2, 15.87)
arrow(ax, 20.3, 15.87, 20.5, 15.87)
arrow(ax, 21.6, 15.87, 21.8, 15.87)
arrow(ax, 22.4, 15.5, 22.4, 15.3)     # down to LN
# messages output
ax.text(23.1, 15.4, "messages\n(E × 64)", ha="left", va="center",
        fontsize=7.5, color="#4A148C", zorder=5)

# Newton's 3rd Law note
box(ax, 20.5, 14.0, 3.3, 0.6, "#FF8F00",
    ["⚠ Newton's 3rd Law (soft):  mᵢⱼ ≈ −mⱼᵢ",
     "Anti-symmetry via directed edge pairing"],
    fontsize=7.5, text_color=TEXT_D)

# --- AGGREGATE
box(ax, 12.1, 14.4, 2.3, 0.8, "#CE93D8",
    ["SUM AGGREGATION",
     "agg[i] = Σⱼ messages[j→i]",
     "shape: N × 64   (scatter_add)"],
    fontsize=7.5, text_color=TEXT_D)

arrow(ax, 22.4, 14.65, 13.5, 14.8, color="#6A1B9A")  # messages → aggregation

# --- CONCAT for NODE MLP
box(ax, 12.1, 13.3, 2.3, 0.8, "#CE93D8",
    ["CONCAT for node_mlp",
     "[node_emb | agg_messages]",
     "shape: N × (64+64) = N × 128"],
    fontsize=7.5, text_color=TEXT_D)
arrow(ax, 13.25, 14.4, 13.25, 14.1)

# --- NODE MLP
ax.text(17.9, 13.2, "NODE MLP   MLP(128→64→64→64, 3 layers)  |  16,704 params",
        ha="center", fontsize=8.5, fontweight="bold", color="#4A148C", zorder=5)
box(ax, 14.6, 12.3, 2.1, 0.75, C_LINEAR,
    ["Linear(128→64)", "W:[64×128]+b:[64]", "8,256 params"], fontsize=7.5)
box(ax, 16.9, 12.3, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 18.2, 12.3, 2.1, 0.75, C_LINEAR,
    ["Linear(64→64)", "W:[64×64]+b:[64]", "4,160 params"], fontsize=7.5)
box(ax, 20.5, 12.3, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 21.8, 12.3, 2.1, 0.75, C_LINEAR,
    ["Linear(64→64)", "W:[64×64]+b:[64]", "4,160 params"], fontsize=7.5)
box(ax, 21.8, 11.4, 1.0, 0.65, C_LN, ["LN(64)", "128 p"], fontsize=7.5)

arrow(ax, 14.4, 13.7, 14.65, 12.67)
arrow(ax, 16.7, 12.67, 16.9, 12.67)
arrow(ax, 18.0, 12.67, 18.2, 12.67)
arrow(ax, 20.3, 12.67, 20.5, 12.67)
arrow(ax, 21.6, 12.67, 21.8, 12.67)
arrow(ax, 22.4, 12.3, 22.4, 12.05)

ax.text(23.1, 12.0, "new_node_emb\n(N × 64)", ha="left", va="center",
        fontsize=7.5, color="#4A148C", zorder=5)

# RESIDUAL EDGE update
box(ax, 20.0, 11.1, 3.5, 0.65, "#7B1FA2",
    ["EDGE RESIDUAL: edge_emb ← edge_emb + messages",
     "(E × 64)  — retains geometric context across steps"],
    fontsize=7.5, text_color=TEXT_W)

# arrow from last node mlp down to loop
arrow(ax, 22.4, 11.4, 22.4, 11.1)
arrow(ax, 22.0, 11.42, 13.25, 13.3, color="#6A1B9A")  # agg → concat node

# ═══════════════════════════════════════════════════════════════
# DECODER
# ═══════════════════════════════════════════════════════════════
ax.text(5.0, 13.75, "DECODER", ha="center", fontsize=9,
        fontweight="bold", color="#B71C1C", zorder=5)
ax.text(5.0, 13.5, "MLP(64→32→32→1, 3 layers, NO LayerNorm)  |  3,169 params",
        ha="center", fontsize=7.5, color="#37474F", zorder=5)

# input from processor
arrow(ax, 11.9, 11.7, 8.5, 13.0, label="node_emb (N×64)", color="#C62828")

box(ax, 1.0, 12.4, 1.9, 0.75, C_LINEAR,
    ["Linear(64→32)", "W:[32×64]+b:[32]", "2,080 params"], fontsize=7.5)
box(ax, 3.1, 12.4, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 4.4, 12.4, 1.9, 0.75, C_LINEAR,
    ["Linear(32→32)", "W:[32×32]+b:[32]", "1,056 params"], fontsize=7.5)
box(ax, 6.5, 12.4, 1.1, 0.75, C_ACTIV, ["SiLU"], fontsize=8)
box(ax, 7.8, 12.4, 1.9, 0.75, C_LINEAR,
    ["Linear(32→1)", "W:[1×32]+b:[1]", "33 params", "init=ZERO weights"],
    fontsize=7.5)

arrow(ax, 1.0, 12.77, 0.6, 12.77)
arrow(ax, 2.9, 12.77, 3.1, 12.77)
arrow(ax, 4.2, 12.77, 4.4, 12.77)
arrow(ax, 6.3, 12.77, 6.5, 12.77)
arrow(ax, 7.6, 12.77, 7.8, 12.77)
arrow(ax, 9.7, 12.77, 10.5, 12.77)
ax.text(10.6, 12.77, "w̃ (N×1)\nraw deflection", ha="left", va="center",
        fontsize=7.5, color="#C62828", zorder=5)

# ─── HARD BC ENFORCEMENT ─────────────────────────────────────────
box(ax, 2.5, 11.1, 4.5, 0.75, C_BC,
    ["HARD BC ENFORCEMENT",
     "w = w̃  ×  (~bc_mask).float()",
     "Boundary nodes forced to exactly w=0  (no penalty needed)"],
    fontsize=7.5)

arrow(ax, 4.75, 12.4, 4.75, 11.85)   # from decoder to BC

# ─── OUTPUT ──────────────────────────────────────────────────────
box(ax, 2.5, 9.7, 4.5, 0.9, C_OUTPUT,
    ["OUTPUT:  w  (N,)",
     "Deflection at every mesh node",
     "Interior nodes: learned values",
     "Boundary nodes: exactly 0.0"],
    fontsize=8)
arrow(ax, 4.75, 11.1, 4.75, 10.6)

# ═══════════════════════════════════════════════════════════════
# PHYSICS LOSS BOX
# ═══════════════════════════════════════════════════════════════
box(ax, 0.5, 8.6, 10.5, 0.85, "#BF360C",
    ["PHYSICS LOSS  (NO dataset, NO labels — only the PDE governs training)",
     "Π(w) = ½·Σ_triangles (∇²w)²·ΔA  −  Σ_nodes q·w·ΔA     (5-point discrete biharmonic Laplacian)",
     "Adam optimizer → minimises Π(w) → w converges to true plate deflection field"],
    fontsize=8, text_color=TEXT_W)

arrow(ax, 4.75, 9.7, 4.75, 9.45)

# ═══════════════════════════════════════════════════════════════
# PARAMETER SUMMARY TABLE (bottom right)
# ═══════════════════════════════════════════════════════════════
section_bg(ax, 11.9, 0.3, 15.8, 8.0, "#E8F5E9", "PARAMETER SUMMARY", 9)

rows = [
    ("Component",             "Layers",                           "Params",   True),
    ("Node Encoder",          "Lin(3→64)+SiLU+Lin(64→64)+LN(64)", "4,544",   False),
    ("Edge Encoder",          "Lin(3→64)+SiLU+Lin(64→64)+LN(64)", "4,544",   False),
    ("",                      "",                                  "",        False),
    ("1× Message Passing Block",  "",                              "37,504",  True),
    ("  edge_mlp",            "Lin(192→64)+SiLU+Lin(64→64)+SiLU+Lin(64→64)+LN", "20,800", False),
    ("  node_mlp",            "Lin(128→64)+SiLU+Lin(64→64)+SiLU+Lin(64→64)+LN", "16,704", False),
    ("",                      "",                                  "",        False),
    ("Processor (5 blocks)",  "5 × MessagePassingBlock",          "187,520", True),
    ("",                      "",                                  "",        False),
    ("Decoder",               "Lin(64→32)+SiLU+Lin(32→32)+SiLU+Lin(32→1)", "3,169", False),
    ("Hard BC",               "w = w̃ × (~bc_mask) — no params", "0",       False),
    ("",                      "",                                  "",        False),
    ("TOTAL",                 "",                                  "199,777", True),
]

col_xs = [12.1, 17.5, 25.5]
row_h  = 0.52
start_y = 7.9
for i, (comp, detail, params, bold) in enumerate(rows):
    ty = start_y - i * row_h
    if comp == "":
        continue
    weight = "bold" if bold else "normal"
    fs = 8.5 if bold else 8
    # background for header / total rows
    if bold and i > 0:
        rect = FancyBboxPatch((12.0, ty - 0.22), 15.5, 0.42,
                               boxstyle="round,pad=0.0,rounding_size=0.1",
                               facecolor="#C8E6C9", edgecolor="none", zorder=2)
        ax.add_patch(rect)
    ax.text(col_xs[0], ty, comp,   ha="left", va="center",
            fontsize=fs, fontweight=weight, color=TEXT_D, zorder=5)
    ax.text(col_xs[1], ty, detail, ha="left", va="center",
            fontsize=7, color="#424242", zorder=5)
    ax.text(col_xs[2], ty, params, ha="right", va="center",
            fontsize=fs, fontweight=weight,
            color="#B71C1C" if bold else TEXT_D, zorder=5)

# header row separator
ax.axhline(y=7.95 - 0.5, xmin=12.1/FIG_W, xmax=27.5/FIG_W,
           color="#90A4AE", linewidth=1.0, zorder=4)

# ═══════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════
legend_items = [
    (C_INPUT,   "Input features"),
    (C_LINEAR,  "Linear layer"),
    (C_ACTIV,   "SiLU activation"),
    (C_LN,      "LayerNorm"),
    (C_PROC,    "Message Passing Block"),
    (C_DECODER, "Decoder"),
    (C_BC,      "Hard BC masking"),
    (C_OUTPUT,  "Output"),
]
lx, ly = 0.4, 7.9
for color, label in legend_items:
    rect = FancyBboxPatch((lx, ly - 0.18), 0.55, 0.36,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax.add_patch(rect)
    ax.text(lx + 0.65, ly, label, ha="left", va="center",
            fontsize=8, color=TEXT_D, zorder=5)
    lx += 1.4
    if lx > 10.5:
        lx = 0.4
        ly -= 0.6

# ═══════════════════════════════════════════════════════════════
# DIMENSIONS FLOW ANNOTATION (bottom of figure)
# ═══════════════════════════════════════════════════════════════
flow_y = 0.6
ax.text(FIG_W/2 - 7, flow_y,
        "Data flow dimensions:  Node (N×3) → Encoder → (N×64)  →  5× Message Passing → (N×64)  →  Decoder → (N×1)  →  Hard BC → w (N,)",
        ha="center", va="center", fontsize=8.5, color="#37474F",
        fontstyle="italic", zorder=6)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(base_dir, "results", "plots", "PIGNN_architecture.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {out_path}")
plt.close()

