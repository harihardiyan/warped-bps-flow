# Contributing to diff-rsgw-field-solver

We welcome contributions from the theoretical physics and scientific computing communities. To maintain the integrity of the solver, please adhere to the following standards:

## Technical Standards
1. **Physical Consistency**: Any modification to the flow equations must be audited using the `hamiltonian_audit` function. We require an L1 residual of $< 10^{-14}$ for all BPS-consistent solutions.
2. **Differentiability**: All new physics kernels must be implemented using pure JAX functions to maintain end-to-end automatic differentiation (AD).
3. **Precision**: All calculations must use `jax_enable_x64` to prevent numerical drift in deep warped geometries.

## Submission Process
- Fork the repository.
- Create a feature branch focusing on a specific physical correction or optimization.
- Ensure all consistency checks pass.
- Submit a Pull Request with a clear description of the physical motivation.

For inquiries, contact Hari Hardiyan (lorozloraz@gmail.com).
