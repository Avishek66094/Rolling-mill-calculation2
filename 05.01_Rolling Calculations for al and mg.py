import numpy as np
import scipy.integrate as integrate
from pathlib import Path
import math
import pandas as pd
from datetime import datetime

# ============================================================================
# Base directory: folder where THIS script is located
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# ---------------- tunables ----------------
MU_DEFAULT = 0.25               # friction μ
PHIDOT_DEFAULT = 1.0            # fallback strain rate [1/s]
PRINT_TORQUE_IN_NM_TOO = True   # also show torque in N·m

# =============================================================================
# Landolt–Börnstein flow stress (Simplified - all floats)
# =============================================================================
class LBParams:
    def __init__(self, A, m1, m2, m4, m5, m7, m8,
                 theta_range, phi_range, phidot_range, name):
        self.A = float(A)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.m4 = float(m4)
        self.m5 = float(m5)
        self.m7 = float(m7)
        self.m8 = float(m8)
        self.theta_range = (float(theta_range[0]), float(theta_range[1]))
        self.phi_range = (float(phi_range[0]), float(phi_range[1]))
        self.phidot_range = (float(phidot_range[0]), float(phidot_range[1]))
        self.name = name


def sigma_LB(p, theta_C, phi, phidot, strict=True):
    """
    Updated Landolt–Börnstein flow stress using:
    σ = A * exp(m1*θ) * φ^(m2) * exp(m4/φ) * (1+φ)^(m5*θ) * exp(m7*φ) * (φdot)^(m8*θ)
    """
    theta = float(theta_C)
    e = float(phi)
    edot = float(phidot)

    if e <= 0.0:
        raise ValueError("phi (true strain) must be > 0 due to exp(m4/phi).")
    if edot <= 0.0:
        raise ValueError("phidot (strain rate) must be > 0.")

    if strict:
        if (theta < p.theta_range[0]) or (theta > p.theta_range[1]):
            raise ValueError(f"{p.name}: θ out of valid range {p.theta_range} °C.")
        if (e < p.phi_range[0]) or (e > p.phi_range[1]):
            raise ValueError(f"{p.name}: φ out of valid range {p.phi_range}.")
        if (edot < p.phidot_range[0]) or (edot > p.phidot_range[1]):
            raise ValueError(f"{p.name}: φdot out of valid range {p.phidot_range} 1/s.")

    sigma = (
        p.A
        * math.exp(p.m1 * theta)
        * (e ** p.m2)
        * math.exp(p.m4 / e)
        * ((1.0 + e) ** (p.m5 * theta))
        * math.exp(p.m7 * e)
        * (edot ** (p.m8 * theta))
    )
    return float(sigma)


# ---------------- Materials ----------------
AL995_HOT_DEF = LBParams(
    A=367.651, m1=-0.00463, m2=0.32911, m4=0.00167,
    m5=-0.00207, m7=0.16592, m8=0.000241,
    theta_range=(250.0, 550.0), phi_range=(0.03, 1.50),
    phidot_range=(0.01, 500.0),
    name="Al 99.5 (hot, deformed)"
)

AZ31_HOT_DIRECT = LBParams(
    A=961.667, m1=-0.00640, m2=0.04403, m4=-0.00718,
    m5=-0.00042, m7=-0.21096, m8=0.000435,
    theta_range=(280.0, 450.0), phi_range=(0.03, 0.75),
    phidot_range=(0.01, 100.0),
    name="AZ31 Mg (hot, direct)"
)

MATERIALS = {
    "al": AL995_HOT_DEF,
    "al995": AL995_HOT_DEF,
    "mg": AZ31_HOT_DIRECT,
    "az31": AZ31_HOT_DIRECT,
}

# =============================================================================
# Data Saving Functions (CSV only)
# =============================================================================
def save_calculation_data(input_data, results):
    """Save calculation data to CSV file"""
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = data_dir / f"roll_calculation_{timestamp}.csv"

    csv_data = {**input_data, **results}

    df = pd.DataFrame([csv_data])
    df.to_csv(filename, index=False)
    print(f"Calculation data saved to: {filename}")
    return filename


# =============================================================================
# Kinematics & Calculations (All floats)
# =============================================================================
def calculate_geometry(h1, reduction_ratio_percent, R):
    """
    Calculate rolling geometry parameters
    Returns: h2, delta_h, ld, reduction_percent
    """
    h1 = float(h1)
    reduction_percent = float(reduction_ratio_percent)
    R = float(R)

    h2 = float((h1 * (100.0 - reduction_percent)) / 100.0)
    delta_h = float(h1 - h2)

    # Contact length approximation (kept from the original code)
    # NOTE: ensure inside sqrt stays positive
    inside = R * delta_h - (delta_h ** 2) / 4.0
    if inside <= 0.0:
        ld = 0.0
    else:
        ld = float(math.sqrt(inside))

    return h2, delta_h, ld, reduction_percent


def calculate_strain_parameters(h1, h2, v, ld):
    """
    Calculate strain parameters
    Returns: phi, phi_dot, engineering strain
    """
    h1 = float(h1)
    h2 = float(h2)
    v = float(v)
    ld = float(ld)

    if h2 <= 0.0 or h1 <= 0.0:
        raise ValueError("h1 and h2 must be > 0.")

    phi = float(math.log(h1 / h2))
    phi_dot = float((v / ld) * phi) if ld > 0 else float(PHIDOT_DEFAULT)

    # engineering strain: ε = (h1 - h2) / h1
    strain = float((h1 - h2) / h1)

    return phi, phi_dot, strain


def weighted_avg_stress(sigma_mg, sigma_al, w_mg=1.0, w_al=2.5):
    sigma_mg = float(sigma_mg)
    sigma_al = float(sigma_al)
    w_mg = float(w_mg)
    w_al = float(w_al)
    return float((w_mg * sigma_mg + w_al * sigma_al) / (w_mg + w_al))


def _ask_float(prompt, default=None):
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Not a number; try again.")

# =============================================================================
# FEB analytical force/torque functions (corrected inputs)
# f1 = f1(a, eps) ; f2 = f2(a, eps)
# where a = μ * sqrt(R / h1)   and eps = (h1 - h2)/h1
# =============================================================================

def a_feb(mu: float, R_mm: float, h1_mm: float) -> float:
    """
    Dimensionless coefficient a:
    a = μ * sqrt(R / h1)
    IMPORTANT: uses initial thickness h1 
    """
    mu = float(mu)
    R_mm = float(R_mm)
    h1_mm = float(h1_mm)
    if mu <= 0.0:
        raise ValueError("μ must be > 0.")
    if R_mm <= 0.0:
        raise ValueError("R must be > 0.")
    if h1_mm <= 0.0:
        raise ValueError("h1 must be > 0.")
    return float(mu * np.sqrt(R_mm / h1_mm))


def eps_engineering(h1_mm: float, h2_mm: float) -> float:
    """
    Engineering reduction ε = (h1 - h2) / h1, in [0,1)
    """
    h1_mm = float(h1_mm)
    h2_mm = float(h2_mm)
    if h1_mm <= 0.0 or h2_mm <= 0.0:
        raise ValueError("h1 and h2 must be > 0.")
    eps = (h1_mm - h2_mm) / h1_mm
    return float(eps)


def _safe_eps(eps: float) -> float:
    """
    Keep eps away from 0 and 1 to avoid singularities in logs/tan/etc.
    """
    return float(np.clip(float(eps), 1e-9, 1.0 - 1e-9))


def uppercase_phi_entry(a: float, eps: float) -> float:
    """
    FEB helper (Eq. 46-style expression), using eps in [0,1).
    """
    a = float(a)
    eps = _safe_eps(eps)
    return float((1.0 / a) * np.sqrt(eps / (1.0 - eps)))


def uppercase_phi_neutral_point(a: float, eps: float) -> float:
    """
    FEB helper (Eq. 46-style neutral point expression), using eps in [0,1).
    """
    a = float(a)
    eps = _safe_eps(eps)

    term1 = 0.5 * np.arctan(np.sqrt(eps / (1.0 - eps)))
    term2 = (1.0 / (4.0 * a)) * np.log(1.0 / (1.0 - eps))

    return float((1.0 / a) * np.tan(term1 - term2))


def function_three(a: float, eps: float) -> float:
    """
    Force function f1 (mapped from FEB f3 in theprevious code),
    now strictly using eps = (h1-h2)/h1 and a = μ*sqrt(R/h1).
    """
    a = float(a)
    eps = _safe_eps(eps)

    def function_one() -> float:
        def core_equation(x: float, switch: int = 1) -> float:
            return float(
                (1.0 + (a**2) * (x**2))
                * np.exp(switch * 2.0 * a * np.arctan(a * x))
            )

        x_np = uppercase_phi_neutral_point(a, eps)
        x_en = uppercase_phi_entry(a, eps)

        first_part = integrate.quad(lambda x: core_equation(x, 1), 0.0, x_np)[0]
        second_part = integrate.quad(lambda x: core_equation(x, -1), x_np, x_en)[0]

        return float(
            first_part
            + (1.0 - eps)
            * np.exp(2.0 * a * np.arctan(np.sqrt(eps / (1.0 - eps))))
            * second_part
        )

    return float(a * np.sqrt((1.0 - eps) / eps) * function_one())


def function_four(a: float, eps: float) -> float:
    """
    Torque function f2 (mapped from FEB f4 in theprevious code),
    now strictly using eps = (h1-h2)/h1 and a = μ*sqrt(R/h1).
    """
    a = float(a)
    eps = _safe_eps(eps)

    def function_two() -> float:
        def core_equation(x: float, switch: int = 1) -> float:
            return float(
                (1.0 + (a**2) * (x**2))
                * np.exp(switch * 2.0 * a * np.arctan(a * x))
                * x
            )

        x_np = uppercase_phi_neutral_point(a, eps)
        x_en = uppercase_phi_entry(a, eps)

        first_part = integrate.quad(lambda x: core_equation(x, 1), 0.0, x_np)[0]
        second_part = integrate.quad(lambda x: core_equation(x, -1), x_np, x_en)[0]

        return float(
            first_part
            + (1.0 - eps)
            * np.exp(2.0 * a * np.arctan(np.sqrt(eps / (1.0 - eps))))
            * second_part
        )

    return float((a**2) * ((1.0 - eps) ** 2) * function_two())


def f1_value(mu: float, R_mm: float, h1_mm: float, eps: float) -> float:
    """
    f1 is a function of (μ*sqrt(R/h1), ε)
    """
    a = a_feb(mu, R_mm, h1_mm)
    return float(function_three(a, eps))


def f2_value(mu: float, R_mm: float, h1_mm: float, eps: float) -> float:
    """
    f2 is a function of (μ*sqrt(R/h1), ε)
    """
    a = a_feb(mu, R_mm, h1_mm)
    return float(function_four(a, eps))

# =============================================================================
# Roll force & torque calculations 
# =============================================================================

def contact_length_term_mm(R_mm: float, delta_h_mm: float) -> float:
    """
    sqrt(R*Δh − (Δh^2)/4)  in mm
    Returns 0 if expression becomes <= 0 (to avoid sqrt domain error).
    """
    R_mm = float(R_mm)
    delta_h_mm = float(delta_h_mm)
    inside = R_mm * delta_h_mm - (delta_h_mm ** 2) / 4.0
    return float(math.sqrt(inside)) if inside > 0.0 else 0.0


def roll_force_P(Km_MPa: float, Bm_mm: float, R_mm: float, delta_h_mm: float, f1: float) -> float:
    """
    Rolling force P (from formula).
    P = 1.15 * Km * Bm * L * f1
    1 MPa = 1 N/mm², so result is in N.
    """
    Km_MPa = float(Km_MPa)
    Bm_mm = float(Bm_mm)
    R_mm = float(R_mm)
    delta_h_mm = float(delta_h_mm)
    f1 = float(f1)

    L_term = contact_length_term_mm(R_mm, delta_h_mm)
    return float(1.15 * Km_MPa * Bm_mm * L_term * f1)


def roll_torque_T(Km_MPa: float, Bm_mm: float, R_mm: float, h1_mm: float, h2_mm: float, f2: float) -> float:
    """
    Rolling torque T (from formula):
    T = (2*Km*Bm*R*((h1^2)/h2)*f2) * 1e-3
    1 MPa = 1 N/mm², so result is in N·mm; multiply by 1e-3 to get N·m.
    """
    Km_MPa = float(Km_MPa)
    Bm_mm = float(Bm_mm)
    R_mm = float(R_mm)
    h1_mm = float(h1_mm)
    h2_mm = float(h2_mm)
    f2 = float(f2)

    if h2_mm <= 0.0:
        raise ValueError("h2 must be > 0 for torque formula.")

    return float((2.0 * Km_MPa * Bm_mm * R_mm * ((h1_mm ** 2) / h2_mm) * f2) * 1e-3)


# =============================================================================
# Main Calculation Function
# =============================================================================
def perform_rolling_calculation():
    """Main function to perform rolling calculation with user inputs"""

    print("=== Rolling Mill Calculation (FEB analytical f1/f2) ===")
    print("Please enter the following parameters:")

    # Get user inputs (ε is no longer asked; it will be calculated)
    try:
        theta = float(input("Temperature θ [°C]: ").strip())
        reduction_ratio = float(input("Reduction ratio [%]: ").strip())
        h1 = float(input("Input thickness h1 [mm]: ").strip())
        v = float(input("Rolling speed v [m/s]: ").strip())
        R = float(input("Roller radius R [mm]: ").strip())
        bm = float(input("Strip width b [mm]: ").strip())
        mu = _ask_float("Friction coefficient μ [default 0.25]: ", MU_DEFAULT)
    except Exception as e:
        print(f"Input error: {e}")
        return

    if not (0.0 < reduction_ratio < 100.0):
        print("Reduction ratio must be between 0 and 100 (exclusive).")
        return
    if h1 <= 0.0 or R <= 0.0 or bm <= 0.0:
        print("h1, R, and b must be > 0.")
        return
    if mu <= 0.0:
        print("μ must be > 0.")
        return

    print("\n=== Calculating Geometry Parameters ===")

    # Calculate geometry
    h2, delta_h, ld, reduction_percent = calculate_geometry(h1, reduction_ratio, R)

    # Calculate strain parameters (φ, φ̇, ε)
    try:
        phi, phi_dot, strain = calculate_strain_parameters(h1, h2, v, ld)
    except Exception as e:
        print(f"Strain parameter error: {e}")
        return

    print("Calculated parameters:")
    print(f"  Exit thickness h2 = {h2:.4f} mm")
    print(f"  Height difference Δh = {delta_h:.4f} mm")
    print(f"  Deformation length ld = {ld:.4f} mm")
    print(f"  True strain φ = {phi:.4f}")
    print(f"  Strain rate φ̇ = {phi_dot:.4f} 1/s")
    print(f"  Engineering strain ε = {strain:.4f}")

    # --- engineering reduction epsilon (this is what FEB part needs) ---
    eps = strain  # strain already computed as (h1-h2)/h1

    print("\n=== FEB analytical coefficients (corrected) ===")
    a_val = a_feb(mu, R, h1)
    print(f"  a = μ * √(R/h1) = {a_val:.6f}")
    print(f"  ε = (h1-h2)/h1 = {eps:.6f}")

    try:
        # f1 and f2 depend on (μ*sqrt(R/h1), ε)
        f1 = f1_value(mu, R, h1, eps)
        f2 = f2_value(mu, R, h1, eps)

        print("\nCoefficients obtained (FEB):")
        print(f"  f1(a, ε) = {f1:.6f}")
        print(f"  f2(a, ε) = {f2:.6f}")


    except Exception as e:
        print(f"Error in FEB coefficient calculation: {e}")
        print("Using fallback values for f1 and f2...")
        f1 = 3.5
        f2 = 0.15
        P_value = roll_force_P(Km, bm, R, delta_h, f1)
        T_value = roll_torque_T(Km, bm, R, h1, h2, f2)
        print(f"Using fallback: f1={f1}, f2={f2}")

    print("\n=== Calculating Deformation Stress ===")

    # Calculate deformation stress for both materials
    try:
        sigma_mg = float(sigma_LB(MATERIALS["mg"], theta, phi, phi_dot, strict=False))
        sigma_al = float(sigma_LB(MATERIALS["al"], theta, phi, phi_dot, strict=False))
        sigma_avg = float(weighted_avg_stress(sigma_mg, sigma_al))
 # In your thesis formulas, Km is required.
        # Here we map Km to the mean deformation stress (sigma_avg) in MPa.
        Km = float(sigma_avg)

        print("Deformation stress:")
        print(f"  σ_Mg(AZ31) = {sigma_mg:.2f} MPa")
        print(f"  σ_Al(99.5) = {sigma_al:.2f} MPa")
        print(f"  σ_average  = {sigma_avg:.2f} MPa")

        print("\n=== Rolling Force / Torque (THESIS formulas) ===")
        P_value = roll_force_P(Km, bm, R, delta_h, f1)
        T_value = roll_torque_T(Km, bm, R, h1, h2, f2)

        print(f"  Km = {Km:.3f} MPa")
        print(f"  P = 1.15*Km*Bm*sqrt(RΔh-(Δh^2)/4)*f1 = {P_value:.6f}")
        if PRINT_TORQUE_IN_NM_TOO:
            print(f"  T = (...) *1e-3 = {T_value:.6f} ({T_value*1000.0:.3f} N·m)")
        else:
            print(f"  T = (...) *1e-3 = {T_value:.6f}")

    except Exception as e:
        print(f"Error in stress calculation: {e}")
        sigma_mg = 0.0
        sigma_al = 0.0
        sigma_avg = 0.0
        Km = 0.0
        P_value = 0.0
        T_value = 0.0

    # Save all data
    input_data = {
        'temperature_C': theta,
        'reduction_ratio_percent': reduction_ratio,
        'h1_mm': h1,
        'rolling_speed_ms': v,
        'roller_radius_mm': R,
        'strip_width_mm': bm,
        'strain_epsilon': strain,
        'friction_coefficient': mu
    }

    results_data = {
        'h2_mm': h2,
        'delta_h_mm': delta_h,
        'deformation_length_mm': ld,
        'true_strain_phi': phi,
        'strain_rate_phidot': phi_dot,
        'engineering_strain': strain,
        'a_parameter_feb': a_val,
        'f1_coefficient_feb': float(f1),
        'f2_coefficient_feb': float(f2),
        'Km_MPa': float(Km),
        'P_roll_force_thesis': float(P_value),
        'T_roll_torque_thesis': float(T_value),
        'sigma_mg_MPa': float(sigma_mg),
        'sigma_al_MPa': float(sigma_al),
        'sigma_avg_MPa': float(sigma_avg),
    }

    save_calculation_data(input_data, results_data)

    print("\n=== Calculation Complete ===")
    print("All data saved relative to script folder")
    print(f"Script folder: {BASE_DIR}")
    print(f"Data folder:   {BASE_DIR / 'data'}")


# ---------------------------- Main Execution ----------------------------
if __name__ == "__main__":
    perform_rolling_calculation()
