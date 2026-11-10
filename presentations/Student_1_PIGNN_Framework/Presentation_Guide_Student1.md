# Presentation Guide: Student 1 (Architecture & Machine Learning Lead)

## Project Title
**Physics-Informed Graph Neural Networks (PI-GNN) for Continuum Mechanics and Structural Emulation**

---

## High-Level Role & Strategy for Student 1
- **Focus Area:** Graph Representation Learning, Neural Network Architecture, Mesh-to-Graph Discretization, and Physics-Loss Integration.
- **Tone:** Methodological & Deep Learning innovation — answering *“How do GNNs allow us to discretize arbitrary domains without training datasets?”*
- **Visuals to Use from Project:**
  - `notes_and_instructions/1000132144.jpg` (Conceptual formulation)
  - `results/plots/training_loss.png` (Convergence and energy minimization)
  - `results/plots/expected_vs_achieved_error.png` (Model comparison)

---

## Slide-by-Slide Presentation Structure (10 - 12 Slides)

### Slide 1: Title & Overview
- **Title:** Physics-Informed Graph Neural Network (PI-GNN) Architecture for Unstructured Continuum Mechanics.
- **Candidate:** Student 1 (Undergraduate Project).
- **Key Message:** Traditional PINNs fail on complex, irregular domains due to Euclidean grid limitations and high-order derivative spectral bias. We introduce a Graph-discretized, zero-data physics-informed framework.

### Slide 2: The Bottleneck in Classical PINNs & FEM
- **FEM Limitations:** High computational cost, rigid meshing, slow for real-time inverse problems.
- **Classical MLP-PINNs Limitations:**
  - Spectral bias (cannot capture local high-frequency gradients).
  - Fixed coordinate input $(x, y) \to$ poor generalization to varied boundary topologies.
- **Reference Paper:** Dalton et al. (2023) *Comput. Methods Appl. Mech. Engrg.* (Paper 1).

### Slide 3: Domain Discretization as a Graph ($\mathcal{G} = \{\mathcal{V}, \mathcal{E}\}$)
- **Why GNNs?** Any finite-element mesh (triangles, tetrahedra) is naturally a graph!
- **Graph Nodes $\mathcal{V}$:** Spatial coordinates $X_i$, boundary flags $d_i \in \{0, 1\}$.
- **Graph Edges $\mathcal{E}$:** Connect adjacent vertices; features $e_{i \to j} = [x_j - x_i, \|x_j - x_i\|_2]$.
- **Invariance:** Relative edge features guarantee translation invariance.

### Slide 4: Message-Passing Scheme & Node Force Equilibrium
- **Encode-Process-Decode Paradigm:**
  1. **Encoder:** Embeds nodal coordinates and edge vectors into latent dimension $M$.
  2. **Processor (Message Passing):**
     $$m^k_{i \to j} = -m^k_{j \to i} \quad \text{(Enforces Newton's 3rd Law of Motion)}$$
  3. **Decoder:** Computes physical displacement $\hat{u}_i$ or deflection $\hat{w}_i$ at every node.

### Slide 5: Unsupervised Physics Training — The "Zero Dataset" Paradigm
- Highlight the diagram in `notes_and_instructions/1000132144.jpg`!
- **Crucial Distinction:** Unlike standard ML, we do **NOT** use simulation datasets or labels ($X$ on "Data").
- **Loss Function:**
  $$\mathcal{L} = \Pi(\hat{u}) = U_{\text{internal strain}}(\hat{u}) - W_{\text{external work}}(\hat{u})$$
- By the Principle of Minimum Potential Energy, minimizing $\mathcal{L}$ directly solves the equilibrium equation.

### Slide 6: Handling High-Order Differential Operators on Graphs
- Challenge: 4th-order Kirchhoff plate equation $\nabla^4 w = q/D$.
- Solution: Discrete Differential Operators (Laplacian and Bilaplacian over graphs) vs. Continuous Splines (Paper 2 comparison).
- How graph neighborhood aggregation computes spatial gradients stably without vanishing gradients.

### Slide 7: Implementation Details & Codebase Architecture
- PyTorch / PyG workflow:
  - `src/graph_construction.py`: Delaunay triangulation & graph builders.
  - `src/gnn_model.py`: Encode-Process-Decode message passing network.
  - `src/physics_loss.py`: Total potential energy & boundary penalty evaluator.
- Optimization: Adam optimizer with learning rate warmup and decay.

### Slide 8: Experimental Loss & Convergence Analysis
- Show `results/plots/training_loss.png`.
- Show how the total potential energy drops monotonically, finding the true variational minimum.
- Numerical stability achieved without dataset pre-training.

### Slide 9: Multi-Architecture Benchmark (MLP vs KAN vs GNN)
- Visual from `results/plots/expected_vs_achieved_error.png`.
- Comparison of parameter counts:
  - MLP-PINN: 2,000 - 11,000 parameters (slow, high error).
  - Tanh-B6 KAN: 51 parameters.
  - PIGNN: Compact graph convolution that scales across different geometries.

### Slide 10: Conclusion & Transition to Student 2
- Summary of Student 1's work: Built the graph architecture, implemented message passing, and enabled zero-dataset physics-informed loss optimization.
- Hand-off statement: *"My partner will now present the mechanics benchmarks, deflection field comparisons across 8 geometries, and CSV engineering validation."*
