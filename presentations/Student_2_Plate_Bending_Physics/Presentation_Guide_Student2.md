# Presentation Guide: Student 2 (Mechanics & Experimental Lead)

## Project Title
**Verification and Benchmark of Physics-Informed Mechanics: Thin Plate Bending Across 8 Benchmark Geometries**

---

## High-Level Role & Strategy for Student 2
- **Focus Area:** Applied Solid Mechanics, Kirchhoff-Love Plate Theory, 8 Numerical Benchmark Cases, Navier Exact Solutions, and Error Validation.
- **Tone:** Applied Engineering & Mechanics Validation — answering *“Does the PIGNN model accurately reproduce the real physical deflection across complex shapes?”*

- **Visuals to Use from Project:**
  - `results/plots/master_8_cases_gallery.png` (8-case master slide)
  - `results/plots/plot_case1.png` through `plot_case8.png`
  - `results/plots/deflection_comparison.png`
  - Reference to CSV tables in `results/csv/`

---

## Slide-by-Slide Presentation Structure (10 - 12 Slides)

### Slide 1: Title & Overview
- **Title:** Engineering Benchmark of Physics-Informed Neural Emulators for Kirchhoff Plate Bending Problems.
- **Candidate:** Student 2 (Undergraduate Project).
- **Key Message:** Validating the PIGNN framework across 4 canonical geometries (Rectangular, Circular/Elliptical, Triangular, L-shaped) under various loading and boundary conditions, benchmarked against exact analytical Navier solutions.

### Slide 2: Kirchhoff-Love Thin Plate Governing Physics
- Based on Liu & Cao (2026) *Engineering Analysis with Boundary Elements* (Paper 2):
  $$D \nabla^4 w(x, y) - q(x, y) = 0$$
  where flexural rigidity $D = \frac{E h^3}{12(1 - \nu^2)}$.
- Boundary conditions:
  - **Clamped (C):** $w = 0, \quad \frac{\partial w}{\partial n} = 0$
  - **Simply Supported (SS):** $w = 0, \quad M_n = -D\left(\frac{\partial^2 w}{\partial n^2} + \nu \frac{\partial^2 w}{\partial t^2}\right) = 0$

### Slide 3: The 8-Case Benchmark Suite
Present the full experimental suite organized in `results/csv/summary_all_8_cases.csv`:
1. **Case 1:** Rectangular Plate — Uniform Load (Simply Supported)
2. **Case 2:** Rectangular Plate — Sinusoidal Load (Simply Supported, Navier test)
3. **Case 3:** Circular/Elliptical Plate — Uniform Load (Clamped Boundary, Section 4.1)
4. **Case 4:** Circular Plate — Uniform Load (Simply Supported)
5. **Case 5:** L-Shaped Non-Convex Plate — Uniform Load (Section 4.4)
6. **Case 6:** L-Shaped Plate — Sinusoidal Load
7. **Case 7:** Triangular Plate — Uniform Load
8. **Case 8:** Triangular Plate — Uniform Bending Moment (Section 4.2)

### Slide 4: Case 1 & 2: Rectangular Plate Bending
- **Physics:** Navier series analytical solution:
  $$w(x, y) = \frac{q_0}{D \pi^4 \left(\frac{1}{a^2} + \frac{1}{b^2}\right)^2} \sin\left(\frac{\pi x}{a}\right) \sin\left(\frac{\pi y}{b}\right)$$
- **Results:**
  - Case 1: Max deflection predicted = $4.06 \times 10^{-3}$ m, Relative $L_2$ error = $2.87\%$.
  - Case 2: Max deflection predicted = $2.57 \times 10^{-3}$ m, Relative $L_2$ error = $2.97\%$.
- Visual: `results/plots/plot_case1.png` and `plot_case2.png`.

### Slide 5: Case 3 & 4: Circular & Elliptical Plates (Curved Boundaries)
- **Physics:** Demonstrating handling of non-rectilinear boundaries where standard Cartesian MLPs struggle.
- **Results:**
  - Case 3 (Clamped): Max deflection = $1.56 \times 10^{-2}$ m, MAE = $7.87 \times 10^{-4}$ m.
  - Case 4 (Simply Supported): Clean boundary adherence without artificial boundary distortion.
- Visual: `results/plots/plot_case3.png` and `plot_case4.png`.

### Slide 6: Case 5 & 6: L-Shaped Plate (Re-entrant Corner Singularity)
- **Challenge Highlight:** Re-entrant $270^\circ$ interior corner introduces stress singularity.
- **Comparison with Paper 2:**
  - Paper 2 notes that rectangular-grid KANs encounter higher local errors at $(0.5, 0.5)$.
  - Our unstructured Delaunay graph conforms directly along the inner boundary edges without coordinate singularity!
- Visual: `results/plots/plot_case5.png` and `plot_case6.png`.

### Slide 7: Case 7 & 8: Triangular Plate (Uniform Moment Boundary)
- **Physics:** Tested under distributed bending moment $M_n = 1 \, \text{N}\cdot\text{m}$.
- Exact polynomial solution from Paper 2 Eq. (28):
  $$w = \frac{M_n}{4bD} \left(y^3 - 3x^2y + 2bx^2 - \frac{b^2y}{3} + \frac{2b^3}{9}\right)$$
- Visual: `results/plots/plot_case7.png` and `plot_case8.png`.

### Slide 8: Master Gallery Across All Geometries
- Highlight `results/plots/master_8_cases_gallery.png`.
- Show how one unified PIGNN codebase solves 8 completely distinct structural mechanics problems.

### Slide 9: Quantitative Validation & Engineering Deliverables
- Highlight CSV export pipeline:
  - Table of coordinates $[x, y]$, reference deflection $w_{\text{ref}}$, predicted deflection $w_{\text{pred}}$, absolute error $|w - \hat{w}|$.
  - Show excerpt of `results/csv/summary_all_8_cases.csv`.
  - Ready for export into CAD / FE post-processors (Abaqus, ANSYS).

### Slide 10: Conclusion & Future Scope
- **Conclusion:** Unsupervised PIGNN successfully solves 4th-order Kirchhoff plate bending across multiple geometries with zero training data.
- **Future Work:**
  - Extension to 3D hyperelastic soft-tissue models (as detailed in Dalton et al., 2023).
  - Adaptive mesh refinement around sharp corners.

