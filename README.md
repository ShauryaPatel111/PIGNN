# Physics-Informed Graph Neural Network (PI-GNN) Project

This repository contains the complete implementation, benchmarks, datasets, results, and presentation materials for the Undergraduate Project based on two foundational papers:
1. **Paper 1 (Dalton et al., 2023):** *Physics-informed graph neural network emulation of soft-tissue mechanics*, Computer Methods in Applied Mechanics and Engineering.
2. **Paper 2 (Liu & Cao, 2026):** *Tanh-B6 KAN-based PINN for solving thin plate bending problems*, Engineering Analysis with Boundary Elements.

---

## Directory Organization

```
kshitij/
├── papers/
│   ├── PIGNN.pdf                            # Paper 1: Dalton et al. (2023) Soft-Tissue PIGNN
│   └── TanB6_plate.pdf                      # Paper 2: Liu & Cao (2026) Thin Plate Bending
│
├── notes_and_instructions/
│   └── 1000132144.jpg                       # TA Guidance & Formulation sketch
│
├── src/                                     # Complete Python implementation
│   ├── gnn_model.py                         # PIGNN Architecture (Encode-Process-Decode)
│   ├── graph_construction.py                # Unstructured mesh & Delaunay graph builder
│   ├── physics_loss.py                      # Unsupervised potential energy loss (0 data)
│   ├── analytical.py                        # Navier and exact analytical solutions
│   ├── run_all_8_cases.py                   # Master benchmark execution suite
│   ├── generate_presentation_graphics.py    # Generates PPT charts & comparative plots
│   ├── testcase2_sinusoidal.py              # Standalone Navier sinusoidal test case
│   ├── train.py                             # Training routine
│   └── visualize.py                         # 2D contour and error plotting
│
├── results/
│   ├── csv/                                 # Exported engineering CSV results
│   │   ├── summary_all_8_cases.csv          # Comprehensive 8-case benchmark summary
│   │   ├── results_case1.csv ... case8.csv  # Node-by-node (x, y, w_ref, w_pred, error)
│   │   └── loss_history.csv                 # Convergence history
│   │
│   └── plots/                               # High-resolution presentation figures
│       ├── master_8_cases_gallery.png       # 8-panel gallery for panel presentation
│       ├── expected_vs_achieved_error.png   # MLP vs KAN vs PIGNN error comparison
│       ├── plot_case1.png ... case8.png     # Individual case contour plots
│       └── training_loss.png                # Loss minimization curve
│
└── presentations/                           # Separate guides for the 2 students
    ├── Student_1_PIGNN_Framework/
    │   └── Presentation_Guide_Student1.md   # GNN architecture, message passing & loss
    └── Student_2_Plate_Bending_Physics/
        └── Presentation_Guide_Student2.md   # Solid mechanics, 8 cases & validation
```

---

## Quick Execution
To run the full suite and regenerate all results:
```bash
python src/run_all_8_cases.py
python src/generate_presentation_graphics.py
```
