# al995_hot_deformed.py
# Flow stress for Al 99.5, HOT deformation, DEFORMED STATE (Landolt-Börnstein).
# σ = A * exp(m1*θ) * φ^m2 * exp(m4/φ) * (1+φ)^m5 * exp(m7*φ) * (φdot)^m8
# Units: θ in °C, φ dimensionless true strain (>0), φdot in 1/s, σ in MPa.

from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np

Number = Union[float, int, np.ndarray]

@dataclass(frozen=True)
class LBParams:
    A: float
    m1: float
    m2: float
    m4: float
    m5: float
    m7: float
    m8: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    phidot_range: Tuple[float, float]

# Al 99.5 — HOT deformation (deformed state) parameters (pages 1–2)
AL995_HOT_DEF = LBParams(
    A=367.651,
    m1=-0.00463,
    m2=0.32911,
    m4=0.00167,
    m5=-0.00207,
    m7=0.16592,
    m8=0.000241,
    theta_range=(250.0, 550.0),   # °C
    phi_range=(0.03, 1.50),       # true strain
    phidot_range=(0.01, 500.0),   # 1/s
)

def sigma_hot_deformed_al995(theta_C: Number, phi: Number, phidot: Number,
                             strict: bool = True) -> np.ndarray:
    """
    Flow stress σ[MPa] for Al 99.5, HOT deformation, DEFORMED state.

    Parameters
    ----------
    theta_C : temperature in °C
    phi     : true strain (dimensionless, > 0)
    phidot  : strain rate [1/s] (> 0)
    strict  : if True, raise on out-of-range inputs; if False, compute anyway.

    Returns
    -------
    σ in MPa (NumPy array broadcast to input shapes)
    """
    p = AL995_HOT_DEF
    theta = np.asarray(theta_C, dtype=float)
    e     = np.asarray(phi, dtype=float)
    edot  = np.asarray(phidot, dtype=float)

    if np.any(e <= 0.0):
        raise ValueError("phi (true strain) must be > 0 due to exp(m4/phi).")
    if np.any(edot <= 0.0):
        raise ValueError("phidot (strain rate) must be > 0.")

    # Range checks (per Landolt–Börnstein validity)
    if strict:
        if np.any((theta < p.theta_range[0]) | (theta > p.theta_range[1])):
            raise ValueError(f"θ out of valid range {p.theta_range} °C.")
        if np.any((e < p.phi_range[0]) | (e > p.phi_range[1])):
            raise ValueError(f"φ out of valid range {p.phi_range}.")
        if np.any((edot < p.phidot_range[0]) | (edot > p.phidot_range[1])):
            raise ValueError(f"φdot out of valid range {p.phidot_range} 1/s.")

    # σ = A * exp(m1*θ) * φ^m2 * exp(m4/φ) * (1+φ)^m5 * exp(m7*φ) * (φdot)^m8
    sigma = (
        p.A
        * np.exp(p.m1 * theta)
        * (e ** p.m2)
        * np.exp(p.m4 / e)
        * ((1.0 + e) ** p.m5)
        * np.exp(p.m7 * e)
        * (edot ** p.m8)
    )
    return sigma

# --- quick demo ---
if __name__ == "__main__":
    # Example: θ=350 °C, φ=0.2, φdot=1 1/s
    s = sigma_hot_deformed_al995(350.0, 0.20, 1.0)
    print(f"σ(350°C, φ=0.20, φdot=1 s^-1) = {float(s):.2f} MPa")
