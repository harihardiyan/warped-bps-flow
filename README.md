


# Differentiable 5D Einstein-Scalar Field Solver for Warped Geometries

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-red.svg)](https://github.com/google/jax)
[![Physics: GR](https://img.shields.io/badge/Physics-General--Relativity-black.svg)](#)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harihardiyan/warped-bps-flow/blob/main/RSGW_Field_Solver_Results.ipynb)

A high-precision, differentiable numerical solver for the 5D Einstein-Scalar system in warped extra-dimensional models (Randall-Sundrum/Goldberger-Wise). This framework incorporates IR-localized quantum backreactions (Jackiw-Teitelboim and Schwarzian corrections) within a mathematically consistent BPS-flow formalism.

## Overview

In theoretical brane-world modeling, introducing geometric corrections without accounting for backreactions often leads to physical inconsistencies. This solver implements a **First-Order Flow (BPS) Formalism**, ensuring that any modification to the geometry is supported by a dynamically derived bulk potential $V(\phi, y)$. 

The primary metric of success for this solver is the **Hamiltonian Constraint Audit**, where the numerical solutions satisfy the 5D Einstein Equations to machine precision ($\approx 10^{-16}$), ensuring a true vacuum state of the modified action.

## Physical Framework

The solver evolves the coupled system of the warp factor $A(y)$ and the scalar field $\phi(y)$ along the extra dimension $y \in [0, \pi r_c]$.

### Flow Equations
The evolution is governed by the effective superpotential $W(\phi, y)$, which generates the first-order flow:
$$ \frac{dA}{dy} = \frac{\kappa_5^2}{6} W(\phi, y) $$
$$ \frac{d\phi}{dy} = \frac{\partial W}{\partial \phi} $$

### Hamiltonian Consistency
To maintain diffeomorphism invariance and satisfy the Einstein Field Equations, the bulk potential $V(\phi, y)$ is derived as:
$$ V(\phi, y) = \frac{1}{2} \left( \frac{\partial W}{\partial \phi} \right)^2 - \frac{\kappa_5^2}{6} W^2 $$
The solver audits the residual of the Hamiltonian constraint $G_{yy} - \kappa_5^2 T_{yy} = 0$ at every grid point.

## Key Features

- **Consistently Backreacted Geometry**: Geometric deformations (JT/Schwarzian) are integrated into the Lagrangian, preventing unphysical energy-momentum violations.
- **End-to-End Differentiability**: Utilizing JAX, the entire integration chain is differentiable, enabling exact adjoint sensitivity analysis ($dA_{IR}/d\theta$).
- **High-Precision Convergence**: Achieves Hamiltonian residuals at the level of $10^{-16}$, representing the limits of 64-bit floating-point precision.
- **Automated Physical Audit**: Built-in verification of the Einstein-Scalar consistency.

## Implementation Details

- **Integrator**: Fixed-step Runge-Kutta 4 (RK4) implemented via `jax.lax.scan` for XLA-optimized execution.
- **Differentiation**: Exact derivatives of the potential and sensitivities via `jax.grad`.
- **Parallelization**: Support for `jax.vmap` for high-throughput parameter space exploration.

## Usage

```python
from main import PhysicsParams, solve_rs_system, hamiltonian_audit

# Initialize parameters with IR quantum couplings
params = PhysicsParams(rc=36.5, eps_JT=0.1, eps_Sch=0.05)

# Solve for the 5D bulk profile
sol = solve_rs_system(params)

# Verify Einstein consistency
residual = hamiltonian_audit(params, sol)
print(f"Hamiltonian Residual: {residual:.2e}")
```

## Results Summary

Initial simulations targetting the Planck-to-Weak scale hierarchy demonstrate:
- **Consistent Warp Factor ($A_{IR}$)**: $\sim 36-58$ for $Z \approx 10^{-16}$ to $10^{-26}$.
- **Machine Precision Residuals**: Stability at $10^{-16}$ confirms the mathematical validity of the backreacted profiles.
- **Sensitivity Hierarchy**: Exact quantification of the influence of higher-derivative corrections on the compactification modulus.

## Author

**Hari Hardiyan**  
Email: [lorozloraz@gmail.com](mailto:lorozloraz@gmail.com)  
*Research focus: Differentiable Physics, Warped Geometries, and Numerical Relativity.*

## Citation

If you use this solver in your research, please cite:

```bibtex
@software{hardiyan2026diffRS,
  author = {Hari Hardiyan},
  title = {Differentiable 5D Einstein-Scalar Field Solver for Warped Geometries},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/harihardiyan/diff-rsgw-field-solver}}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
