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
            raise ValueError(
                f"{p.name}: φdot out of valid range {p.phidot_range} 1/s."
            )

    sigma = (
        p.A
        * math.exp(p.m1 * theta)
        * (e ** p.m2)
        * math.exp(p.m4 / e)
        * ((1.0 + e) ** p.m5)
        * math.exp(p.m7 * e)
        * (edot ** p.m8)
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

    # Contact length approximation (kept from your original code)
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
# FEB analytical force/torque functions (replaces image digitization)
# =============================================================================
def a_feb(friction_coeff: float, radius_mm: float, height_end_mm: float) -> float:
    """Dimensionless coefficient a (FEB Eq. 41)"""
    friction_coeff = float(friction_coeff)
    radius_mm = float(radius_mm)
    height_end_mm = float(height_end_mm)
    if height_end_mm <= 0.0:
        raise ValueError("height_end must be > 0 for a_feb.")
    return float(friction_coeff * np.sqrt(radius_mm / height_end_mm))


def r_feb(height_at_start_mm: float, height_end_mm: float) -> float:
    """Reduction r in [0, 1] (FEB Eq. 42)"""
    h0 = float(height_at_start_mm)
    h1 = float(height_end_mm)
    if h0 <= 0.0:
        raise ValueError("height_at_start must be > 0 for r_feb.")
    return float((h0 - h1) / h0)


def uppercase_phi_entry(coeff_a: float, reduction: float) -> float:
    """Angle at entry / friction coefficient (FEB Eq. 46)"""
    coeff_a = float(coeff_a)
    reduction = float(reduction)
    return float((1.0 / coeff_a) * np.sqrt(reduction / (1.0 - reduction)))


def uppercase_phi_neutral_point(coeff_a: float, reduction: float) -> float:
    """Angle at neutral point / friction coefficient (FEB Eq. 46)"""
    coeff_a = float(coeff_a)
    reduction = float(reduction)
    return float(
        (1.0 / coeff_a)
        * np.tan(
            0.5 * np.arctan(np.sqrt(reduction / (1.0 - reduction)))
            - (1.0 / (4.0 * coeff_a)) * np.log(1.0 / (1.0 - reduction))
        )
    )


def function_three(coeff_a: float, reduction: float) -> float:
    """Force function f3 (FEB Eq. 55)"""
    coeff_a = float(coeff_a)
    reduction = float(reduction)

    def function_one() -> float:
        """Part of the Force function (FEB Eq. 51)"""

        def core_equation(x: float, switch: int = 1) -> float:
            return float(
                (1.0 + (coeff_a**2) * (x**2))
                * np.exp(switch * 2.0 * coeff_a * np.arctan(coeff_a * x))
            )

        x_np = uppercase_phi_neutral_point(coeff_a, reduction)
        x_en = uppercase_phi_entry(coeff_a, reduction)

        first_part = integrate.quad(lambda x: core_equation(x, 1), 0.0, x_np)[0]
        second_part = integrate.quad(lambda x: core_equation(x, -1), x_np, x_en)[0]

        return float(
            first_part
            + (1.0 - reduction)
            * np.exp(2.0 * coeff_a * np.arctan(np.sqrt(reduction / (1.0 - reduction))))
            * second_part
        )

    return float(coeff_a * np.sqrt((1.0 - reduction) / reduction) * function_one())


def function_four(coeff_a: float, reduction: float) -> float:
    """Torque function f4 (FEB Eq. 56)"""
    coeff_a = float(coeff_a)
    reduction = float(reduction)

    def function_two() -> float:
        """Part of the Torque function (FEB Eq. 54)"""

        def core_equation(x: float, switch: int = 1) -> float:
            return float(
                (1.0 + (coeff_a**2) * (x**2))
                * np.exp(switch * 2.0 * coeff_a * np.arctan(coeff_a * x))
                * x
            )

        x_np = uppercase_phi_neutral_point(coeff_a, reduction)
        x_en = uppercase_phi_entry(coeff_a, reduction)

        first_part = integrate.quad(lambda x: core_equation(x, 1), 0.0, x_np)[0]
        second_part = integrate.quad(lambda x: core_equation(x, -1), x_np, x_en)[0]

        return float(
            first_part
            + (1.0 - reduction)
            * np.exp(2.0 * coeff_a * np.arctan(np.sqrt(reduction / (1.0 - reduction))))
            * second_part
        )

    return float((coeff_a**2) * ((1.0 - reduction) ** 2) * function_two())


def f1_value(reduction_percent: float, coeff_a: float) -> float:
    """
    Replacement for chart-based f1:
    Map FEB function_three (f3) to your f1 usage.
    Input: reduction in [%] -> converted to [0..1].
    """
    red = float(np.clip(reduction_percent / 100.0, 1e-9, 1.0 - 1e-9))
    return float(function_three(float(coeff_a), red))


def f2_value(reduction_percent: float, coeff_a: float) -> float:
    """
    Replacement for chart-based f2:
    Map FEB function_four (f4) to your f2 usage.
    """
    red = float(np.clip(reduction_percent / 100.0, 1e-9, 1.0 - 1e-9))
    return float(function_four(float(coeff_a), red))


# =============================================================================
# Roll force & torque calculations (kept as in your original script)
# =============================================================================
def roll_force_from_f1(f1, width_mm, h0_mm):
    """kN"""
    f1 = float(f1)
    width_mm = float(width_mm)
    h0_mm = float(h0_mm)
    return float((f1 * width_mm * h0_mm) / 1000.0)


def roll_torque_from_f2(f2, width_mm, h0_mm, R_eff_mm):
    """kN·m"""
    f2 = float(f2)
    width_mm = float(width_mm)
    h0_mm = float(h0_mm)
    R_eff_mm = float(R_eff_mm)
    return float((f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0)


def roll_force_basic(sigma_avg_MPa, width_mm, delta_h_mm):
    """kN"""
    sigma_avg_MPa = float(sigma_avg_MPa)
    width_mm = float(width_mm)
    delta_h_mm = float(delta_h_mm)
    return float((sigma_avg_MPa * width_mm * delta_h_mm) / 1000.0)


def roll_torque_basic(F_kN, roll_radius_mm):
    """kN·m"""
    F_kN = float(F_kN)
    roll_radius_mm = float(roll_radius_mm)
    return float(F_kN * (roll_radius_mm / 1000.0))


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

    # FEB a-parameter uses height_end (h2)
    a_val = a_feb(mu, R, max(h2, 1e-9))

    print("\n=== FEB analytical coefficients (no digitization) ===")
    print(f"  a = μ * √(R/h2) = {a_val:.6f}")

    # Get f1 and f2 coefficients from FEB equations
    try:
        f1 = f1_value(reduction_percent, a_val)
        f2 = f2_value(reduction_percent, a_val)

        print("\nCoefficients obtained (FEB):")
        print(f"  f1(r={reduction_percent:.2f}%, a={a_val:.6f}) = {f1:.6f}")
        print(f"  f2(r={reduction_percent:.2f}%, a={a_val:.6f}) = {f2:.6f}")

        print("\n=== Calculating Force and Torque ===")

        # Calculate force and torque from coefficients (kept from your code)
        F_coeff_kN = roll_force_from_f1(f1, bm, h1)
        T_coeff_kNm = roll_torque_from_f2(f2, bm, h1, R)

        print("From coefficients:")
        print(f"  Rolling force F = {F_coeff_kN:.3f} kN")
        if PRINT_TORQUE_IN_NM_TOO:
            print(
                f"  Rolling torque T = {T_coeff_kNm:.6f} kN·m "
                f"({T_coeff_kNm * 1000.0:.3f} N·m)"
            )
        else:
            print(f"  Rolling torque T = {T_coeff_kNm:.6f} kN·m")

    except Exception as e:
        print(f"Error in FEB coefficient calculation: {e}")
        print("Using fallback values for f1 and f2...")
        f1 = 3.5
        f2 = 0.15
        F_coeff_kN = roll_force_from_f1(f1, bm, h1)
        T_coeff_kNm = roll_torque_from_f2(f2, bm, h1, R)
        print(f"Using fallback: f1={f1}, f2={f2}")

    print("\n=== Calculating Deformation Stress ===")

    # Calculate deformation stress for both materials
    try:
        sigma_mg = float(sigma_LB(MATERIALS["mg"], theta, phi, phi_dot, strict=False))
        sigma_al = float(sigma_LB(MATERIALS["al"], theta, phi, phi_dot, strict=False))
        sigma_avg = float(weighted_avg_stress(sigma_mg, sigma_al))

        print("Deformation stress:")
        print(f"  σ_Mg(AZ31) = {sigma_mg:.2f} MPa")
        print(f"  σ_Al(99.5) = {sigma_al:.2f} MPa")
        print(f"  σ_average  = {sigma_avg:.2f} MPa")

        # Basic force/torque calculation for comparison
        F_basic_kN = roll_force_basic(sigma_avg, bm, delta_h)
        T_basic_kNm = roll_torque_basic(F_basic_kN, R)

        print("\nBasic calculation (for comparison):")
        print(f"  Rolling force F = {F_basic_kN:.3f} kN")
        if PRINT_TORQUE_IN_NM_TOO:
            print(
                f"  Rolling torque T = {T_basic_kNm:.6f} kN·m "
                f"({T_basic_kNm * 1000.0:.3f} N·m)"
            )
        else:
            print(f"  Rolling torque T = {T_basic_kNm:.6f} kN·m")

    except Exception as e:
        print(f"Error in stress calculation: {e}")
        sigma_mg = 0.0
        sigma_al = 0.0
        sigma_avg = 0.0
        F_basic_kN = 0.0
        T_basic_kNm = 0.0

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
        'force_coeff_kN': float(F_coeff_kN),
        'torque_coeff_kNm': float(T_coeff_kNm),
        'sigma_mg_MPa': float(sigma_mg),
        'sigma_al_MPa': float(sigma_al),
        'sigma_avg_MPa': float(sigma_avg),
        'force_basic_kN': float(F_basic_kN),
        'torque_basic_kNm': float(T_basic_kNm)
    }

    save_calculation_data(input_data, results_data)

    print("\n=== Calculation Complete ===")
    print("All data saved relative to script folder")
    print(f"Script folder: {BASE_DIR}")
    print(f"Data folder:   {BASE_DIR / 'data'}")


# ---------------------------- Main Execution ----------------------------
if __name__ == "__main__":
    perform_rolling_calculation()
