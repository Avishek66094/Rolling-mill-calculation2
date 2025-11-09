import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math
import pandas as pd
from datetime import datetime
import os

# ============================================================================
# Base directory: folder where THIS script is located
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent

# ---------------- tunables ----------------
MU_DEFAULT = 0.25               # friction μ
A_KEY_DECIMALS = 3              # rounding for 'a' cache keys
PHIDOT_DEFAULT = 1.0            # fallback strain rate [1/s]
PRINT_TORQUE_IN_NM_TOO = True   # also show torque in N·m

# ---------------- grids (x: reduction %) -------------------
_RPTS = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)

# =============================================================================
# Image Handling - relative to script location (no absolute / desktop path)
# =============================================================================
def get_image_path(filename: str) -> Path:
    """
    Find an image next to this script (or in ./images).
    This works no matter from where you run the script, as long as the
    PNG/JPG file travels in the same folder (or 'images' subfolder).
    """
    search_dirs = [BASE_DIR, BASE_DIR / "images"]

    possible_names = [
        filename,
        f"{filename}.png",
        f"{filename}.PNG",
        f"{filename}.jpg",
        f"{filename}.JPG",
        f"{filename}.jpeg",
        f"{filename}.JPEG",
    ]

    for folder in search_dirs:
        for name in possible_names:
            image_path = folder / name
            if image_path.exists():
                print(f"Found image: {image_path}")
                return image_path

    raise FileNotFoundError(
        f"Could not find image '{filename}' in {search_dirs}"
    )

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


def save_digitization_data(fig_name, a_value,
                           x_points, y_points,
                           calibrated_x, calibrated_y):
    """Save digitization data to CSV file (no Excel dependency)"""
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = data_dir / f"digitization_{fig_name.replace(' ', '_')}_{timestamp}.csv"

    points_data = []
    for i in range(len(x_points)):
        points_data.append({
            'point_index': i,
            'x_pixel': x_points[i],
            'y_pixel': y_points[i],
            'x_calibrated': calibrated_x[i],
            'y_calibrated': calibrated_y[i]
        })

    df = pd.DataFrame(points_data)
    df.to_csv(filename, index=False)
    print(f"Digitization data saved to: {filename}")
    return filename

# =============================================================================
# Kinematics & Calculations (All floats)
# =============================================================================
def calculate_geometry(h1, reduction_ratio, R):
    """
    Calculate rolling geometry parameters
    Returns: h2, delta_h, ld, reduction_percent
    """
    h1 = float(h1)
    reduction_ratio = float(reduction_ratio)
    R = float(R)

    reduction_percent = float(reduction_ratio)
    h2 = float((h1 * (100 - reduction_percent)) / 100)
    delta_h = float(h1 - h2)
    ld = float(math.sqrt(R * delta_h - (delta_h ** 2) / 4))

    return h2, delta_h, ld, reduction_percent


def calculate_strain_parameters(h1, h2, v, ld):
    """
    Calculate strain parameters
    Returns: phi, phi_dot, strain
    """
    h1 = float(h1)
    h2 = float(h2)
    v = float(v)
    ld = float(ld)

    phi = float(math.log(h1 / h2))
    phi_dot = float((v / ld) * phi) if ld > 0 else float(PHIDOT_DEFAULT)
    # engineering strain (as requested): ε = (h1 - h2) / h1
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

# ---------- calibration + single-curve digitizer ----------
def _calibrate_axis(ax, prompt, default_low, default_high, fig_name):
    print(f"\nCalibration for {prompt}: click first tick (low), then second tick (high).")
    plt.draw()
    p1 = plt.ginput(1, timeout=-1)
    if not p1:
        raise RuntimeError("No click detected for first tick")
    p2 = plt.ginput(1, timeout=-1)
    if not p2:
        raise RuntimeError("No click detected for second tick")

    (x0, y0) = p1[0]
    (x1, y1) = p2[0]
    v0 = _ask_float(f"  Value at first tick [default {default_low}]: ", default_low)
    v1 = _ask_float(f"  Value at second tick [default {default_high}]: ", default_high)

    # Save calibration plot
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cal_plot_path = plots_dir / f"calibration_{fig_name.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(cal_plot_path, dpi=300, bbox_inches='tight')
    print(f"Calibration plot saved to: {cal_plot_path}")

    if prompt.lower().startswith("y"):
        return (y0, y1), (v0, v1)
    return (x0, x1), (v0, v1)


def _px_to_val(px, px0, px1, v0, v1):
    px = float(px)
    px0 = float(px0)
    px1 = float(px1)
    v0 = float(v0)
    v1 = float(v1)
    return float(((px - px0) / (px1 - px0)) * (v1 - v0) + v0)


def _digitize_single_curve(image_path, label, reductions_grid,
                           x_defaults=(0.0, 100.0),
                           y_defaults=(0.0, 1.0),
                           a_value=0.0):
    """
    Digitize ONE curve (for the current 'a'): click 6-12 points left→right, press ENTER.
    Returns values sampled on 'reductions_grid', plus raw points for saving.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(
        f"{label}\nCalibrate axes, then click along THIS 'a' curve (ENTER to finish)\n"
        f"a = {a_value:.3f}"
    )
    plt.tight_layout()

    print("\n=== Axis calibration ===")
    (xpx0, xpx1), (xv0, xv1) = _calibrate_axis(
        ax, "x-axis (Reduction %)", x_defaults[0], x_defaults[1], label
    )
    (ypx0, ypx1), (yv0, yv1) = _calibrate_axis(
        ax, "y-axis (function value)", y_defaults[0], y_defaults[1], label
    )

    print("\n=== Curve digitization (single curve) ===")
    print("Click 6-12 points along the curve for YOUR 'a' value, then press ENTER.")
    print("Make sure to click points from left to right along the curve.")
    pts = []

    # Show the image again for digitization
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(
        f"{label}\nClick points along the curve for a = {a_value:.3f} (ENTER when done)"
    )
    plt.tight_layout()

    while True:
        clicks = plt.ginput(n=0, timeout=0)
        if not clicks:
            break
        pts.extend(clicks)
        plt.plot([p[0] for p in pts], [p[1] for p in pts],
                 "ro", ms=6, alpha=0.8)
        plt.draw()

    if len(pts) < 2:
        plt.close()
        raise RuntimeError(
            "Too few points captured; please retry and click more points along the curve."
        )

    # Convert points to calibrated values
    xs_px = [float(p[0]) for p in pts]
    ys_px = [float(p[1]) for p in pts]
    xs_val = [_px_to_val(x, xpx0, xpx1, xv0, xv1) for x in xs_px]
    ys_val = [_px_to_val(y, ypx0, ypx1, yv0, yv1) for y in ys_px]

    # Save final digitization plot
    plots_dir = BASE_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digit_plot_path = plots_dir / (
        f"digitization_{label.replace(' ', '_')}_a{a_value:.3f}_{timestamp}.png"
    )

    # Create final plot with interpolated curve
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title(
        f"{label}\nDigitized points and interpolated curve for a = {a_value:.3f}"
    )

    plt.plot(xs_px, ys_px, "ro", ms=6, alpha=0.8, label='Clicked points')

    # Add the interpolated curve to the plot
    if len(xs_val) >= 2:
        sorted_indices = np.argsort(xs_val)
        xs_val_sorted = [xs_val[i] for i in sorted_indices]
        ys_val_sorted = [ys_val[i] for i in sorted_indices]

        mask = [
            (x >= reductions_grid.min()) and (x <= reductions_grid.max())
            for x in xs_val_sorted
        ]
        xs_val_filtered = [x for i, x in enumerate(xs_val_sorted) if mask[i]]
        ys_val_filtered = [y for i, y in enumerate(ys_val_sorted) if mask[i]]

        if len(xs_val_filtered) >= 2:
            x_interp = np.linspace(
                float(reductions_grid.min()),
                float(reductions_grid.max()), 100
            )
            y_interp = np.interp(x_interp, xs_val_filtered, ys_val_filtered)
            # convert back to pixel coordinates for plotting
            x_px_interp = [
                _px_to_val(x, xv0, xv1, xpx0, xpx1) for x in x_interp
            ]
            y_px_interp = [
                _px_to_val(y, yv0, yv1, ypx0, ypx1) for y in y_interp
            ]
            plt.plot(x_px_interp, y_px_interp, 'b-', linewidth=3,
                     alpha=0.7, label='Interpolated curve')

    plt.legend()
    plt.tight_layout()
    plt.savefig(digit_plot_path, dpi=300, bbox_inches='tight')
    print(f"Digitization plot saved to: {digit_plot_path}")
    plt.close('all')

    # Save digitization data
    save_digitization_data(label, a_value, xs_px, ys_px, xs_val, ys_val)

    # Sort and filter points for interpolation
    sorted_indices = np.argsort(xs_val)
    xs_val_sorted = [xs_val[i] for i in sorted_indices]
    ys_val_sorted = [ys_val[i] for i in sorted_indices]

    mask = [
        (x >= reductions_grid.min()) and (x <= reductions_grid.max())
        for x in xs_val_sorted
    ]
    xs_val_filtered = [x for i, x in enumerate(xs_val_sorted) if mask[i]]
    ys_val_filtered = [y for i, y in enumerate(ys_val_sorted) if mask[i]]

    if len(xs_val_filtered) < 2:
        raise RuntimeError(
            "Not enough points within the x-range; "
            "try clicking across the whole curve."
        )

    row = np.interp(reductions_grid, xs_val_filtered, ys_val_filtered)
    return row, xs_px, ys_px

# ---------- per-figure caches ----------
def _load_cache(path):
    """
    Returns (TABLE, R_points, A_keys). TABLE shape = (n_a, n_r).
    If cache doesn't exist yet, returns empty TABLE and keys.
    """
    if not path.exists():
        return np.empty((0, _RPTS.size), dtype=float), _RPTS.copy(), []
    dat = np.load(str(path), allow_pickle=True)
    table = dat["TABLE"]
    rpts = dat["R_points"]
    akeys = dat["A_keys"].tolist()
    return table, rpts, akeys


def _save_cache(path, table, rpts, akeys):
    np.savez(str(path), TABLE=table, R_points=rpts,
             A_keys=np.array(akeys, dtype=float))


def _ensure_curve_value(fig_name, image_path, y_defaults,
                        a_value, r_query, cache_basename):
    """
    Ensures we can return f(r_query, a_value) for a given figure.
    """
    cache_path = BASE_DIR / cache_basename
    table, rpts, akeys = _load_cache(cache_path)
    a_key = round(float(a_value), A_KEY_DECIMALS)

    # Helper: value at reduction r for a row index i
    def val_at_r(i):
        return float(np.interp(float(r_query), rpts, table[i, :]))

    # 1) exact hit
    if a_key in akeys:
        i = akeys.index(a_key)
        return val_at_r(i)

    # 2) have neighbors → interpolate across a on the scalar value f(r)
    if len(akeys) >= 2:
        a_sorted_idx = np.argsort(akeys)
        a_sorted = [akeys[i] for i in a_sorted_idx]
        j = np.searchsorted(a_sorted, a_key)
        if 0 < j < len(a_sorted):
            il = a_sorted_idx[j - 1]
            ih = a_sorted_idx[j]
            al, ah = float(akeys[il]), float(akeys[ih])
            fl, fh = val_at_r(il), val_at_r(ih)
            t = (a_key - al) / (ah - al)
            return float((1.0 - t) * fl + t * fh)

    # 3) need to digitize THIS a only
    print(f"\nNo cached curve for a={a_key:.{A_KEY_DECIMALS}f} in {fig_name}.")
    print("Please click points for THIS curve (once).")
    row, raw_points_x, raw_points_y = _digitize_single_curve(
        image_path, fig_name, _RPTS,
        x_defaults=(0.0, 100.0), y_defaults=y_defaults,
        a_value=a_key
    )

    # append to cache
    if table.size == 0:
        table = row[None, :]
        akeys = [a_key]
    else:
        table = np.vstack([table, row])
        akeys.append(a_key)
    _save_cache(cache_path, table, _RPTS, akeys)

    return float(np.interp(float(r_query), _RPTS, row))

# ---------- public API: f1, f2 for given (r, a) ----------
def f1_value(r_percent, a_value):
    """Get f1 value from Roll Pressure Function chart"""
    f1_img = get_image_path("Roll pressure function")
    return _ensure_curve_value(
        fig_name="Fig. 42 — Roll Pressure Function f3",
        image_path=f1_img,
        y_defaults=(0.0, 90.0),  # Based on chart range
        a_value=a_value,
        r_query=float(np.clip(r_percent, _RPTS.min(), _RPTS.max())),
        cache_basename="f1_cache.npz"
    )


def f2_value(r_percent, a_value):
    """Get f2 value from Roll Torque Function chart"""
    f2_img = get_image_path("Roll Torque function")
    return _ensure_curve_value(
        fig_name="Fig. 43 — Torque Function f4",
        image_path=f2_img,
        y_defaults=(0.0, 0.7),   # Based on chart range
        a_value=a_value,
        r_query=float(np.clip(r_percent, _RPTS.min(), _RPTS.max())),
        cache_basename="f2_cache.npz"
    )

# =============================================================================
# Roll force & torque calculations (All floats)
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

    print("=== Rolling Mill Calculation ===")
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

    print("\n=== Calculating Geometry Parameters ===")

    # Calculate geometry
    h2, delta_h, ld, reduction_percent = calculate_geometry(h1, reduction_ratio, R)

    # Calculate strain parameters (φ, φ̇, ε)
    phi, phi_dot, strain = calculate_strain_parameters(h1, h2, v, ld)

    print("Calculated parameters:")
    print(f"  Exit thickness h2 = {h2:.4f} mm")
    print(f"  Height difference Δh = {delta_h:.4f} mm")
    print(f"  Deformation length ld = {ld:.4f} mm")
    print(f"  True strain φ = {phi:.4f}")
    print(f"  Strain rate φ̇ = {phi_dot:.4f} 1/s")
    print(f"  Engineering strain ε = {strain:.4f}")

    # Calculate 'a' parameter
    a = mu * math.sqrt(R / max(h1, 1e-9))
    a_key = round(a, A_KEY_DECIMALS)

    print("\n=== Getting f1 and f2 coefficients ===")
    print(f"Your calculated a value is: {a:.3f}")
    print("You will now digitize the curves for this specific 'a' value.")
    print("Look for the curve that matches or is closest to your 'a' value on the charts.")

    # Get f1 and f2 coefficients from digitized curves
    try:
        f1 = f1_value(reduction_percent, a)
        f2 = f2_value(reduction_percent, a)

        print("\nCoefficients obtained:")
        print(f"  a = μ * √(R/h1) = {mu:.3f} * √({R:.1f}/{h1:.3f}) = {a:.3f}")
        print(f"  f1(r={reduction_percent:.2f}%, a={a_key:.3f}) = {f1:.6f}")
        print(f"  f2(r={reduction_percent:.2f}%, a={a_key:.3f}) = {f2:.6f}")

        print("\n=== Calculating Force and Torque ===")

        # Calculate force and torque from coefficients
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
        print(f"Error in digitization: {e}")
        print("Using fallback values for f1 and f2...")
        # Fallback values based on typical curves
        f1 = 3.5   # Fallback value
        f2 = 0.15  # Fallback value
        F_coeff_kN = roll_force_from_f1(f1, bm, h1)
        T_coeff_kNm = roll_torque_from_f2(f2, bm, h1, R)
        print(f"Using fallback: f1={f1}, f2={f2}")

    print("\n=== Calculating Deformation Stress ===")

    # Calculate deformation stress for both materials
    try:
        sigma_mg = float(sigma_LB(MATERIALS["mg"], theta, phi, phi_dot,
                                  strict=False))
        sigma_al = float(sigma_LB(MATERIALS["al"], theta, phi, phi_dot,
                                  strict=False))
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
        F_basic_kN = 0.0
        T_basic_kNm = 0.0

    # Save all data (strain is the calculated ε)
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
        'a_parameter': a,
        'f1_coefficient': f1,
        'f2_coefficient': f2,
        'force_coeff_kN': F_coeff_kN,
        'torque_coeff_kNm': T_coeff_kNm,
        'sigma_mg_MPa': sigma_mg if 'sigma_mg' in locals() else 0,
        'sigma_al_MPa': sigma_al if 'sigma_al' in locals() else 0,
        'sigma_avg_MPa': sigma_avg if 'sigma_avg' in locals() else 0,
        'force_basic_kN': F_basic_kN,
        'torque_basic_kNm': T_basic_kNm
    }

    save_calculation_data(input_data, results_data)

    print("\n=== Calculation Complete ===")
    print("All data and plots saved relative to script folder")
    print(f"Script folder: {BASE_DIR}")
    print(f"Plots folder:  {BASE_DIR / 'plots'}")
    print(f"Data folder:   {BASE_DIR / 'data'}")

# ---------------------------- Main Execution ----------------------------
if __name__ == "__main__":
    perform_rolling_calculation()