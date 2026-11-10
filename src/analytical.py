"""
analytical.py
-------------
Person 1's responsibility.

Provides the analytical (exact) solution for thin plate bending
using the Navier double sine series.

Problem:  Simply-supported rectangular plate, uniform load q₀ = 1
          Domain: [0,1] × [0,1]  (non-dimensional)
          Equation: ∇⁴w = 1  (since q/D = 1 after non-dimensionalization)

Navier solution:
    w(x,y) = Σ_m Σ_n  W_mn sin(mπx) sin(nπy)

    where:
    W_mn = (16 / (π⁶ mn)) · 1/(m² + n²)²     for odd m, n only

This is the benchmark that Person 2's PIGNN will be compared against.
"""

import numpy as np


def navier_solution(x: np.ndarray, y: np.ndarray,
                    n_terms: int = 20) -> np.ndarray:
    """
    Compute the Navier series solution for the simply-supported
    square plate under uniform unit load (non-dimensional).

    w(x,y) = Σ_{m=1,3,5,...} Σ_{n=1,3,5,...}
              [16/(π⁶ mn (m²+n²)²)] sin(mπx) sin(nπy)

    Parameters
    ----------
    x, y     : 1D arrays of coordinates (can be flat arrays from meshgrid)
    n_terms  : number of series terms per dimension (use odd indices only)

    Returns
    -------
    w : deflection at each (x, y) point
    """
    w = np.zeros_like(x, dtype=float)
    for m in range(1, 2 * n_terms, 2):      # m = 1, 3, 5, ...
        for n in range(1, 2 * n_terms, 2):  # n = 1, 3, 5, ...
            coeff = 16.0 / (np.pi**6 * m * n * (m**2 + n**2)**2)
            w += coeff * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
    return w


def max_deflection_analytical(n_terms: int = 50) -> float:
    """Return the analytical centre deflection at (0.5, 0.5)."""
    return navier_solution(np.array([0.5]), np.array([0.5]), n_terms)[0]


if __name__ == "__main__":
    w_center = max_deflection_analytical(n_terms=50)
    print(f"Analytical centre deflection: {w_center:.8f}")
    print(f"Literature value (~0.00406):  0.00406")
    # (Timoshenko & Woinowsky-Krieger, Theory of Plates and Shells)

