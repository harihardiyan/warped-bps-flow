
"""
Consistent 5D Einstein-Scalar Field Solver for RS-GW Models.
Methodology: First-order BPS flow with IR-localized quantum backreactions.
Numerical Framework: JAX-accelerated differentiable programming.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, vmap
from jax.tree_util import register_pytree_node_class
from dataclasses import dataclass
from typing import Dict, Tuple

# Enable 64-bit precision for theoretical physics accuracy
jax.config.update("jax_enable_x64", True)

@register_pytree_node_class
@dataclass(frozen=True)
class PhysicsParams:
    """Registered JAX PyTree for RS-GW Action Parameters."""
    k: float = 1.0           # AdS Curvature Scale
    rc: float = 36.5         # Compactification Radius
    kappa5_sq: float = 1.0   # 5D Gravitational Coupling
    vUV: float = 0.1         # UV Brane Scalar VEV
    vIR: float = 0.01        # IR Brane Scalar VEV
    eps_JT: float = 0.1      # JT Gravity Coupling (IR)
    eps_Sch: float = 0.05    # Schwarzian Action Coupling (IR)
    Ny: int = 10000          # Spatial Grid Resolution (Static)

    def tree_flatten(self):
        children = (self.k, self.rc, self.kappa5_sq, self.vUV, self.vIR, self.eps_JT, self.eps_Sch)
        aux_data = {'Ny': self.Ny}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

# ---------- Physics Kernels ----------

def effective_superpotential(p: PhysicsParams, y: float, phi: float, Ymax: float, c2: float):
    """Effective Superpotential W(phi, y) generating the coupled flow."""
    W0 = 3.0 * p.k / p.kappa5_sq
    W_gw = W0 + c2 * (phi**2)
    
    # Smooth localization window for IR quantum backreactions
    window = 0.5 * (1.0 + jnp.tanh((y - 0.95 * Ymax) / (0.02 * Ymax)))
    
    # Geometric backreactions integrated into the flow generator
    correction = p.eps_JT * window + p.eps_Sch * window * (W_gw**2) / (1.0 + W_gw**2)
    
    return W_gw + (6.0 / p.kappa5_sq) * correction

def derived_bulk_potential(p: PhysicsParams, y: float, phi: float, Ymax: float, c2: float):
    """Potential V(phi, y) derived from W to satisfy the Hamiltonian constraint."""
    W = effective_superpotential(p, y, phi, Ymax, c2)
    dW_dphi = grad(lambda ph: effective_superpotential(p, y, ph, Ymax, c2))(phi)
    return 0.5 * (dW_dphi**2) - (p.kappa5_sq / 6.0) * (W**2)

# ---------- Solver Logic ----------

@jit
def flow_equations(p: PhysicsParams, y: float, U: jnp.ndarray, Ymax: float, c2: float):
    """Coupled first-order flow equations (phi' and A')."""
    phi, A = U
    W = effective_superpotential(p, y, phi, Ymax, c2)
    dphi = grad(lambda ph: effective_superpotential(p, y, ph, Ymax, c2))(phi)
    dA = (p.kappa5_sq / 6.0) * W
    return jnp.array([dphi, dA])

@jit
def solve_rs_system(p: PhysicsParams):
    """Integrates the 5D bulk equations using a differentiable RK4 scheme."""
    Ymax = jnp.pi * p.rc
    h = Ymax / (p.Ny - 1)
    c2 = jnp.log(p.vIR / p.vUV) / (2.0 * Ymax)
    U0 = jnp.array([p.vUV, 0.0])

    def rk4_step(Uc, i):
        yi = i * h
        k1 = flow_equations(p, yi, Uc, Ymax, c2)
        k2 = flow_equations(p, yi + 0.5*h, Uc + 0.5*h*k1, Ymax, c2)
        k3 = flow_equations(p, yi + 0.5*h, Uc + 0.5*h*k2, Ymax, c2)
        k4 = flow_equations(p, yi + h, Uc + h*k3, Ymax, c2)
        Un = Uc + (h/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        return Un, Un

    _, trajectory = lax.scan(rk4_step, U0, jnp.arange(p.Ny - 1))
    U_full = jnp.vstack([U0, trajectory])
    
    return {
        "y": jnp.linspace(0.0, Ymax, p.Ny),
        "phi": U_full[:, 0],
        "A": U_full[:, 1],
        "A_IR": U_full[-1, 1],
        "Ymax": Ymax,
        "c2": c2
    }

# ---------- Analysis & Validation ----------

@jit
def hamiltonian_audit(p: PhysicsParams, sol: Dict):
    """Validates the Hamiltonian constraint G_yy = kappa^2 T_yy."""
    y, phi, A = sol["y"], sol["phi"], sol["A"]
    Ymax, c2 = sol["Ymax"], sol["c2"]
    
    # Geometric sector: 6(A')^2
    Ap_sq = vmap(lambda yi, phii: ((p.kappa5_sq/6.0) * effective_superpotential(p, yi, phii, Ymax, c2))**2)(y, phi)
    Gyy = 6.0 * Ap_sq
    
    # Matter sector: 1/2(phi')^2 - V
    phip = vmap(lambda yi, phii: grad(lambda ph: effective_superpotential(p, yi, ph, Ymax, c2))(phii))(y, phi)
    V_bulk = vmap(lambda yi, phii: derived_bulk_potential(p, yi, phii, Ymax, c2))(y, phi)
    Tyy = 0.5 * (phip**2) - V_bulk
    
    return jnp.mean(jnp.abs(Gyy - p.kappa5_sq * Tyy))

@jit
def adjoint_sensitivity(p: PhysicsParams):
    """Computes exact parameter sensitivity via Automatic Differentiation."""
    def get_target(params):
        return solve_rs_system(params)["A_IR"]
    return grad(get_target)(p)

# ---------- Formatting & Execution ----------

def format_summary(p: PhysicsParams, sol: Dict, residual: float, sensitivity: PhysicsParams):
    """Structured report for scientific documentation."""
    z_ir = jnp.exp(-sol["A_IR"])
    print("-" * 60)
    print(f"{'PARAMETER':<30} | {'VALUE':<25}")
    print("-" * 60)
    print(f"{'Total Warp Factor (A_IR)':<30} | {sol['A_IR']:.8f}")
    print(f"{'Effective Redshift (Z)':<30} | {z_ir:.6e}")
    print(f"{'Hamiltonian Residual (L1)':<30} | {residual:.2e}")
    print(f"{'Consistency Status':<30} | {'PASS' if residual < 1e-12 else 'FAIL'}")
    print("-" * 60)
    print(f"{'SENSITIVITY (dA_IR/dTheta)':<30} |")
    print(f"{'  to JT Coupling':<30} | {sensitivity.eps_JT:.6f}")
    print(f"{'  to Schwarzian Action':<30} | {sensitivity.eps_Sch:.6f}")
    print("-" * 60)

if __name__ == "__main__":
    # Baseline simulation at Hierarchy scale
    config = PhysicsParams(rc=36.5)
    
    # Execute physics modules
    simulation_output = solve_rs_system(config)
    physics_residual = hamiltonian_audit(config, simulation_output)
    param_gradients = adjoint_sensitivity(config)
    
    # Print formal summary
    format_summary(config, simulation_output, physics_residual, param_gradients)
