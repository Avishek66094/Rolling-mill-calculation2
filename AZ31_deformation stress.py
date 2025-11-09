# az31_flowstress.py
# Flow stress σ(θ, φ, φdot) for AZ31 per Landolt-Börnstein (hot & cold),
# including the two hot routes: (1) heat to 500°C then cool, (2) heat directly to deformation temperature.

from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Union
import numpy as np

Number = Union[float, int, np.ndarray]

@dataclass(frozen=True)
class FlowCurveParams:
    # σ = A * exp(m1*θ) * φ^m2 * exp(m4/φ) * (1+φ)^m5 * exp(m7*φ) * (φdot)^m8
    A: float            # MPa
    m1: float           # temperature exponent
    m2: float           # strain exponent
    m4: float           # 1/strain exponent (inside exp(m4/φ))
    m5: float           # (1+φ)^m5
    m7: float           # exp(m7*φ)
    m8: float           # strain-rate exponent
    # Validity ranges
    theta_range: Tuple[float, float]      # °C (min, max)
    strain_range_fn: Callable[[Number], Tuple[Number, Number]]  # returns (φ_min, φ_max) possibly θ-dependent
    phidot_range: Tuple[float, float]     # 1/s (min, max)
    # Optional: reported base stress (not used in the equation but recorded)
    sigma_F0: float = None                # MPa

def _hot_strain_range(_: Number) -> Tuple[float, float]:
    # 0.03 … 0.75 for both hot cases
    return (0.03, 0.75)

def _cold_strain_range(theta: Number) -> Tuple[Number, Number]:
    # 0.04 … 0.103*exp(0.00753*θ) (θ in °C)
    phi_min = 0.04
    phi_max = 0.103 * np.exp(0.00753 * np.asarray(theta))
    return (phi_min, phi_max)

# --------------------------
# Published coefficients for AZ31 (from your PDF)
# --------------------------
AZ31_HOT_COOL_FROM_500 = FlowCurveParams(
    sigma_F0=95.43,
    A=1055.970,
    m1=-0.00562,
    m2=0.10905,
    m4=-0.01550,
    m5=-0.00221,
    m7=-0.17951,
    m8=0.000382,
    theta_range=(280.0, 450.0),
    strain_range_fn=_hot_strain_range,
    phidot_range=(0.01, 100.0),
)

AZ31_HOT_DIRECT_HEATING = FlowCurveParams(
    sigma_F0=89.96,
    A=961.667,
    m1=-0.00640,
    m2=0.04403,
    m4=-0.00718,
    m5=-0.00042,
    m7=-0.21096,
    m8=0.000435,
    theta_range=(280.0, 450.0),
    strain_range_fn=_hot_strain_range,
    phidot_range=(0.01, 100.0),
)

AZ31_COLD = FlowCurveParams(
    sigma_F0=370.41,
    A=542.330,
    m1=-0.00411,
    m2=0.16785,
    m4=-0.00866,
    m5=-0.01449,
    m7=1.43153,
    m8=0.000159,
    theta_range=(20.0, 200.0),
    strain_range_fn=_cold_strain_range,
    phidot_range=(0.01, 20.0),
)

PARAM_SETS: Dict[str, FlowCurveParams] = {
    # hot route 1: heated to 500°C then cooled to deformation temperature
    "hot_cool_from_500": AZ31_HOT_COOL_FROM_500,
    # hot route 2: heated directly to deformation temperature  ← your note
    "hot_direct_heating": AZ31_HOT_DIRECT_HEATING,
    "cold": AZ31_COLD,
}

def sigma_flow(theta_C: Number, phi: Number, phidot: Number,
               mode: str = "hot_direct_heating") -> np.ndarray:
    """
    Compute flow stress σ [MPa] for AZ31.

    Parameters
    ----------
    theta_C : °C (scalar or array)
    phi     : true strain ε (scalar or array, must be > 0)
    phidot  : strain rate [1/s] (scalar or array, must be > 0)
    mode    : one of {"hot_cool_from_500", "hot_direct_heating", "cold"}

    Returns
    -------
    σ in MPa (numpy array broadcast to input shapes)
    """
    if mode not in PARAM_SETS:
        raise ValueError(f"mode must be one of {list(PARAM_SETS.keys())}")
    p = PARAM_SETS[mode]

    theta = np.asarray(theta_C, dtype=float)
    e = np.asarray(phi, dtype=float)
    edot = np.asarray(phidot, dtype=float)

    if np.any(e <= 0.0):
        raise ValueError("True strain φ must be > 0 to evaluate exp(m4/φ).")
    if np.any(edot <= 0.0):
        raise ValueError("Strain rate φdot must be > 0.")

    # Optional validity checks (warn hard if out-of-range)
    tmin, tmax = p.theta_range
    if np.any((theta < tmin) | (theta > tmax)):
        # Don’t silently extrapolate without telling the caller
        raise ValueError(f"Temperature {theta_C} °C out of valid range {p.theta_range} for mode={mode}.")
    emin, emax = p.strain_range_fn(theta)
    if np.any((e < emin) | (e > emax)):
        raise ValueError("Strain φ out of valid range for this mode and temperature.")
    dmin, dmax = p.phidot_range
    if np.any((edot < dmin) | (edot > dmax)):
        raise ValueError(f"Strain rate φdot out of valid range [{dmin}, {dmax}] 1/s.")

    # Flow law (Landolt-Börnstein, Eq. (2.10) in your pages)
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

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    # Example: θ=350°C, φ=0.2, φdot=1 s^-1
    print("hot_direct_heating:", sigma_flow(350, 0.2, 1.0, mode="hot_direct_heating"))
    print("hot_cool_from_500:", sigma_flow(350, 0.2, 1.0, mode="hot_cool_from_500"))
    # Cold example inside its valid domain: θ=100°C, φ limited by φ_max(θ)
    print("cold:", sigma_flow(100, 0.08, 0.1, mode="cold"))
