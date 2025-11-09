from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math
import pandas as pd
from datetime import datetime
import os

Number = Union[float, int, np.ndarray]

# ---------------- tunables ----------------
MU_DEFAULT = 0.25            # friction μ
A_KEY_DECIMALS = 3           # rounding for 'a' cache keys
PHIDOT_DEFAULT = 1.0         # fallback strain rate [1/s]
PRINT_TORQUE_IN_NM_TOO = True  # also show torque in N·m

# ---------------- image handling (pwd method) ----------------
def get_image_path(filename: str) -> Path:
    """
    Get image path from current working directory
    This works anywhere - GitHub, Jupyter, local machine
    """
    current_dir = Path.cwd()
    possible_names = [
        filename,
        f"{filename}.png",
        f"{filename}.PNG",
        f"{filename}.jpg",
        f"{filename}.JPG",
        f"{filename}.jpeg",
        f"{filename}.JPEG"
    ]
    
    for name in possible_names:
        image_path = current_dir / name
        if image_path.exists():
            return image_path
    
    # If not found, create empty directories for output
    output_dirs = ['plots', 'data']
    for dir_name in output_dirs:
        (current_dir / dir_name).mkdir(exist_ok=True)
    
    raise FileNotFoundError(f"Could not find {filename} in current directory: {current_dir}")

# ---------------- grids (x: reduction %) -------------------
_RPTS = np.array([5,10,20,30,40,50,60,70,80,90], dtype=float)

# =============================================================================
# Landolt–Börnstein flow stress
# =============================================================================
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
    name: str = "material"

def sigma_LB(p: LBParams, theta_C: Number, phi: Number, phidot: Number, strict: bool = True) -> np.ndarray:
    theta = np.asarray(theta_C, dtype=float)
    e     = np.asarray(phi, dtype=float)
    edot  = np.asarray(phidot, dtype=float)

    if np.any(e <= 0.0):
        raise ValueError("phi (true strain) must be > 0 due to exp(m4/phi).")
    if np.any(edot <= 0.0):
        raise ValueError("phidot (strain rate) must be > 0.")

    if strict:
        if np.any((theta < p.theta_range[0]) | (theta > p.theta_range[1])):
            raise ValueError(f"{p.name}: θ out of valid range {p.theta_range} °C.")
        if np.any((e < p.phi_range[0]) | (e > p.phi_range[1])):
            raise ValueError(f"{p.name}: φ out of valid range {p.phi_range}.")
        if np.any((edot < p.phidot_range[0]) | (edot > p.phidot_range[1])):
            raise ValueError(f"{p.name}: φdot out of valid range {p.phidot_range} 1/s.")

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

# ---------------- Materials ----------------
AL995_HOT_DEF = LBParams(
    A=367.651, m1=-0.00463, m2=0.32911, m4=0.00167, m5=-0.00207, m7=0.16592, m8=0.000241,
    theta_range=(250.0, 550.0), phi_range=(0.03, 1.50), phidot_range=(0.01, 500.0),
    name="Al 99.5 (hot, deformed)"
)

AZ31_HOT_DIRECT = LBParams(
    A=961.667, m1=-0.00640, m2=0.04403, m4=-0.00718, m5=-0.00042, m7=-0.21096, m8=0.000435,
    theta_range=(280.0, 450.0), phi_range=(0.03, 0.75), phidot_range=(0.01, 100.0),
    name="AZ31 Mg (hot, direct)"
)

MATERIALS: Dict[str, LBParams] = {
    "al": AL995_HOT_DEF,
    "al995": AL995_HOT_DEF,
    "mg": AZ31_HOT_DIRECT,
    "az31": AZ31_HOT_DIRECT,
}

# =============================================================================
# Data Saving Functions
# =============================================================================
def save_calculation_data(input_data: dict, results: dict):
    """Save calculation data to CSV file"""
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = data_dir / f"roll_calculation_{timestamp}.csv"
    
    # Combine input and results data
    csv_data = {**input_data, **results}
    
    df = pd.DataFrame([csv_data])
    df.to_csv(filename, index=False)
    print(f"Calculation data saved to: {filename}")
    return filename

def save_digitization_data(fig_name: str, a_value: float, x_points: list, y_points: list, 
                          calibrated_x: list, calibrated_y: list):
    """Save digitization data to CSV file"""
    current_dir = Path.cwd()
    data_dir = current_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = data_dir / f"digitization_{fig_name.replace(' ', '_')}_{timestamp}.csv"
    
    digitization_data = {
        'figure_name': fig_name,
        'a_value': a_value,
        'timestamp': timestamp,
        'num_points': len(x_points)
    }
    
    # Points data
    points_data = []
    for i, (x_px, y_px) in enumerate(x_points):
        points_data.append({
            'point_index': i,
            'x_pixel': x_px,
            'y_pixel': y_px,
            'x_calibrated': calibrated_x[i] if i < len(calibrated_x) else 0,
            'y_calibrated': calibrated_y[i] if i < len(calibrated_y) else 0
        })
    
    points_df = pd.DataFrame(points_data)
    digitization_df = pd.DataFrame([digitization_data])
    
    # Save to Excel with multiple sheets
    excel_filename = filename.with_suffix('.xlsx')
    with pd.ExcelWriter(excel_filename) as writer:
        digitization_df.to_excel(writer, sheet_name='metadata', index=False)
        points_df.to_excel(writer, sheet_name='points', index=False)
    
    print(f"Digitization data saved to: {excel_filename}")
    return excel_filename

# =============================================================================
# Kinematics & Calculations
# =============================================================================
def calculate_geometry(h1: float, reduction_ratio: float, R: float) -> tuple:
    """
    Calculate rolling geometry parameters
    Returns: h2, delta_h, ld, reduction_percent
    """
    reduction_percent = reduction_ratio
    h2 = (h1 * (100 - reduction_percent)) / 100
    delta_h = h1 - h2
    ld = math.sqrt(R * delta_h - (delta_h ** 2) / 4)
    
    return h2, delta_h, ld, reduction_percent

def calculate_strain_parameters(h1: float, h2: float, v: float, ld: float) -> tuple:
    """
    Calculate strain parameters
    Returns: phi, phi_dot, strain
    """
    phi = math.log(h1 / h2)
    phi_dot = (v / ld) * phi if ld > 0 else PHIDOT_DEFAULT
    strain = (h1 - h2) / h1
    
    return phi, phi_dot, strain

def weighted_avg_stress(sigma_mg: Number, sigma_al: Number, w_mg: float = 1.0, w_al: float = 2.5) -> np.ndarray:
    return (w_mg * np.asarray(sigma_mg) + w_al * np.asarray(sigma_al)) / (w_mg + w_al)

def _ask_float(prompt: str, default: Optional[float] = None) -> float:
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("Not a number; try again.")

# ---------- calibration + single-curve digitizer ----------
def _calibrate_axis(ax, prompt: str, default_low: float, default_high: float, fig_name: str):
    print(f"\nCalibration for {prompt}: click first tick (low), then second tick (high).")
    p1 = plt.ginput(1, timeout=-1)
    if not p1:
        raise RuntimeError("No click detected for first tick")
    p2 = plt.ginput(1, timeout=-1)
    if not p2:
        raise RuntimeError("No click detected for second tick")
        
    (x0,y0) = p1[0]
    (x1,y1) = p2[0]
    v0 = _ask_float(f"  Value at first tick [default {default_low}]: ", default_low)
    v1 = _ask_float(f"  Value at second tick [default {default_high}]: ", default_high)
    
    # Save calibration plot
    current_dir = Path.cwd()
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cal_plot_path = plots_dir / f"calibration_{fig_name.replace(' ', '_')}_{timestamp}.png"
    plt.savefig(cal_plot_path, dpi=300, bbox_inches='tight')
    print(f"Calibration plot saved to: {cal_plot_path}")
    
    if prompt.lower().startswith("y"):
        return (y0,y1), (v0,v1)
    return (x0,x1), (v0,v1)

def _px_to_val(px, px0, px1, v0, v1):
    return ((px - px0) / (px1 - px0)) * (v1 - v0) + v0

def _digitize_single_curve(image_path: Path, label: str,
                           reductions_grid: np.ndarray,
                           x_defaults=(0.0, 100.0), y_defaults=(0.0, 1.0),
                           a_value: float = 0.0) -> tuple[np.ndarray, list, list]:
    """
    Digitize ONE curve (for the current 'a'): click 6–12 points left→right, press ENTER.
    Returns values sampled on 'reductions_grid', plus raw points for saving.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{label}\nCalibrate axes, then click along THIS 'a' curve (ENTER to finish)\na = {a_value:.3f}")

    print("\n=== Axis calibration ===")
    (xpx0,xpx1), (xv0,xv1) = _calibrate_axis(ax, "x-axis (Reduction %)", x_defaults[0], x_defaults[1], label)
    (ypx0,ypx1), (yv0,yv1) = _calibrate_axis(ax, "y-axis (function value)", y_defaults[0], y_defaults[1], label)

    print("\n=== Curve digitization (single curve) ===")
    print("Tip: Click 6–12 points along the curve (left→right), then press ENTER.")
    pts: List[tuple[float,float]] = []
    while True:
        clicks = plt.ginput(n=0, timeout=0)
        if not clicks:
            break
        pts.extend(clicks)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "ro", ms=4, alpha=0.7)
        plt.pause(0.01)

    # Convert points to calibrated values
    xs_px = np.array([p[0] for p in pts], float)
    ys_px = np.array([p[1] for p in pts], float)
    xs_val = _px_to_val(xs_px, xpx0, xpx1, xv0, xv1)
    ys_val = _px_to_val(ys_px, ypx0, ypx1, yv0, yv1)

    # Save final digitization plot
    current_dir = Path.cwd()
    plots_dir = current_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digit_plot_path = plots_dir / f"digitization_{label.replace(' ', '_')}_a{a_value:.3f}_{timestamp}.png"
    
    # Add the interpolated curve to the plot
    order = np.argsort(xs_val)
    xs_val_sorted, ys_val_sorted = xs_val[order], ys_val[order]
    mask = (xs_val_sorted >= reductions_grid.min()) & (xs_val_sorted <= reductions_grid.max())
    xs_val_sorted, ys_val_sorted = xs_val_sorted[mask], ys_val_sorted[mask]
    
    if len(xs_val_sorted) >= 2:
        # Plot the interpolated curve
        x_interp = np.linspace(reductions_grid.min(), reductions_grid.max(), 100)
        y_interp = np.interp(x_interp, xs_val_sorted, ys_val_sorted)
        ax.plot(x_interp, _px_to_val(y_interp, yv0, yv1, ypx0, ypx1), 'b-', linewidth=2, alpha=0.7, label='Interpolated')
        ax.legend()
    
    plt.savefig(digit_plot_path, dpi=300, bbox_inches='tight')
    print(f"Digitization plot saved to: {digit_plot_path}")
    plt.close(fig)

    if len(pts) < 2:
        raise RuntimeError("Too few points captured; please retry and click more points along the curve.")

    # Prepare data for saving
    raw_points = [(float(x), float(y)) for x, y in pts]
    calibrated_points = [(float(x), float(y)) for x, y in zip(xs_val, ys_val)]
    
    # Save digitization data
    save_digitization_data(label, a_value, raw_points, calibrated_points, xs_val.tolist(), ys_val.tolist())

    order = np.argsort(xs_val)
    xs_val, ys_val = xs_val[order], ys_val[order]
    mask = (xs_val >= reductions_grid.min()) & (xs_val <= reductions_grid.max())
    xs_val, ys_val = xs_val[mask], ys_val[mask]
    if len(xs_val) < 2:
        raise RuntimeError("Not enough points within the x-range; try clicking across the whole curve.")

    row = np.interp(reductions_grid, xs_val, ys_val)
    return row, raw_points, calibrated_points

# ---------- per-figure caches (rows keyed by rounded 'a') ----------
def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Returns (TABLE, R_points, A_keys). TABLE shape = (n_a, n_r).
    If cache doesn't exist yet, returns empty TABLE and keys.
    """
    if not path.exists():
        return np.empty((0, _RPTS.size), dtype=float), _RPTS.copy(), []
    dat = np.load(str(path), allow_pickle=True)
    table = dat["TABLE"]
    rpts  = dat["R_points"]
    akeys = dat["A_keys"].tolist()
    return table, rpts, akeys

def _save_cache(path: Path, table: np.ndarray, rpts: np.ndarray, akeys: list[float]) -> None:
    np.savez(str(path), TABLE=table, R_points=rpts, A_keys=np.array(akeys, dtype=float))

def _ensure_curve_value(fig_name: str,
                        image_path: Path,
                        y_defaults: tuple[float, float],
                        a_value: float,
                        r_query: float,
                        cache_basename: str) -> float:
    """
    Ensures we can return f(r_query, a_value) for a given figure.
    """
    current_dir = Path.cwd()
    cache_path = current_dir / cache_basename
    table, rpts, akeys = _load_cache(cache_path)
    a_key = round(float(a_value), A_KEY_DECIMALS)

    # Helper: value at reduction r for a row index i
    def val_at_r(i: int) -> float:
        return float(np.interp(r_query, rpts, table[i, :]))

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
            il = a_sorted_idx[j-1]
            ih = a_sorted_idx[j]
            al, ah = akeys[il], akeys[ih]
            fl, fh = val_at_r(il), val_at_r(ih)
            t = (a_key - al) / (ah - al)
            return (1.0 - t) * fl + t * fh

    # 3) need to digitize THIS a only
    print(f"\nNo cached curve for a={a_key:.{A_KEY_DECIMALS}f} in {fig_name}.")
    print("Please click points for THIS curve (once).")
    row, raw_points, calibrated_points = _digitize_single_curve(
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

    return float(np.interp(r_query, _RPTS, row))

# ---------- public API: f1, f2 for given (r, a) ----------
def f1_value(r_percent: float, a_value: float) -> float:
    f1_img = get_image_path("Roll pressure function")
    return _ensure_curve_value(
        fig_name="Fig. 42 — Roll Pressure Function f3",
        image_path=f1_img,
        y_defaults=(0.0, 90.0),
        a_value=a_value,
        r_query=float(np.clip(r_percent,  _RPTS.min(), _RPTS.max())),
        cache_basename="f1_cache.npz"
    )

def f2_value(r_percent: float, a_value: float) -> float:
    f2_img = get_image_path("Roll Torque function")
    return _ensure_curve_value(
        fig_name="Fig. 43 — Torque Function f4",
        image_path=f2_img,
        y_defaults=(0.0, 0.7),
        a_value=a_value,
        r_query=float(np.clip(r_percent,  _RPTS.min(), _RPTS.max())),
        cache_basename="f2_cache.npz"
    )

# =============================================================================
# Roll force & torque calculations
# =============================================================================
def roll_force_from_f1(f1: float, width_mm: float, h0_mm: float) -> float:
    """kN"""
    return (f1 * width_mm * h0_mm) / 1000.0

def roll_torque_from_f2(f2: float, width_mm: float, h0_mm: float, R_eff_mm: float) -> float:
    """kN·m"""
    return (f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0

def roll_force_basic(sigma_avg_MPa: float, width_mm: float, delta_h_mm: float) -> float:
    """kN"""
    return (sigma_avg_MPa * width_mm * delta_h_mm) / 1000.0

def roll_torque_basic(F_kN: float, roll_radius_mm: float) -> float:
    """kN·m"""
    return F_kN * (roll_radius_mm / 1000.0)

# =============================================================================
# Main Calculation Function
# =============================================================================
def perform_rolling_calculation():
    """Main function to perform rolling calculation with user inputs"""
    
    print("=== Rolling Mill Calculation ===")
    print("Please enter the following parameters:")
    
    # Get user inputs
    try:
        theta = float(input("Temperature θ [°C]: ").strip())
        reduction_ratio = float(input("Reduction ratio [%]: ").strip())
        h1 = float(input("Input thickness h1 [mm]: ").strip())
        v = float(input("Rolling speed v [m/s]: ").strip())
        R = float(input("Roller radius R [mm]: ").strip())
        bm = float(input("Strip width b [mm]: ").strip())
        epsilon = float(input("Strain ε: ").strip())
        mu = _ask_float("Friction coefficient μ [default 0.25]: ", 0.25)
    except Exception as e:
        raise SystemExit(f"Input error: {e}")

    print("\n=== Calculating Geometry Parameters ===")
    
    # Calculate geometry
    h2, delta_h, ld, reduction_percent = calculate_geometry(h1, reduction_ratio, R)
    
    # Calculate strain parameters
    phi, phi_dot, strain = calculate_strain_parameters(h1, h2, v, ld)
    
    print(f"Calculated parameters:")
    print(f"  Exit thickness h2 = {h2:.4f} mm")
    print(f"  Height difference Δh = {delta_h:.4f} mm")
    print(f"  Deformation length ld = {ld:.4f} mm")
    print(f"  True strain φ = {phi:.4f}")
    print(f"  Strain rate φ̇ = {phi_dot:.4f} 1/s")
    print(f"  Engineering strain ε = {strain:.4f}")

    # Calculate 'a' parameter
    a = mu * math.sqrt(R / max(h1, 1e-9))
    a_key = round(a, A_KEY_DECIMALS)

    print(f"\n=== Getting f1 and f2 coefficients ===")
    
    # Get f1 and f2 coefficients from digitized curves
    f1 = f1_value(reduction_percent, a)
    f2 = f2_value(reduction_percent, a)

    print(f"Coefficients:")
    print(f"  a = μ * √(R/h1) = {mu:.3f} * √({R:.1f}/{h1:.3f}) = {a:.3f}")
    print(f"  f1(r={reduction_percent:.2f}%, a={a_key:.3f}) = {f1:.6f}")
    print(f"  f2(r={reduction_percent:.2f}%, a={a_key:.3f}) = {f2:.6f}")

    print(f"\n=== Calculating Force and Torque ===")
    
    # Calculate force and torque from coefficients
    F_coeff_kN = roll_force_from_f1(f1, bm, h1)
    T_coeff_kNm = roll_torque_from_f2(f2, bm, h1, R)

    print(f"From coefficients:")
    print(f"  Rolling force F = {F_coeff_kN:.3f} kN")
    if PRINT_TORQUE_IN_NM_TOO:
        print(f"  Rolling torque T = {T_coeff_kNm:.6f} kN·m ({T_coeff_kNm*1000.0:.3f} N·m)")
    else:
        print(f"  Rolling torque T = {T_coeff_kNm:.6f} kN·m")

    print(f"\n=== Calculating Deformation Stress ===")
    
    # Calculate deformation stress for both materials
    sigma_mg = float(sigma_LB(MATERIALS["mg"], theta, phi, phi_dot, strict=False))
    sigma_al = float(sigma_LB(MATERIALS["al"], theta, phi, phi_dot, strict=False))
    sigma_avg = float(weighted_avg_stress(sigma_mg, sigma_al))

    print(f"Deformation stress:")
    print(f"  σ_Mg(AZ31) = {sigma_mg:.2f} MPa")
    print(f"  σ_Al(99.5) = {sigma_al:.2f} MPa")
    print(f"  σ_average  = {sigma_avg:.2f} MPa")

    # Basic force/torque calculation for comparison
    F_basic_kN = roll_force_basic(sigma_avg, bm, delta_h)
    T_basic_kNm = roll_torque_basic(F_basic_kN, R)

    print(f"\nBasic calculation (for comparison):")
    print(f"  Rolling force F = {F_basic_kN:.3f} kN")
    if PRINT_TORQUE_IN_NM_TOO:
        print(f"  Rolling torque T = {T_basic_kNm:.6f} kN·m ({T_basic_kNm*1000.0:.3f} N·m)")
    else:
        print(f"  Rolling torque T = {T_basic_kNm:.6f} kN·m")

    # Save all data
    input_data = {
        'temperature_C': theta,
        'reduction_ratio_percent': reduction_ratio,
        'h1_mm': h1,
        'rolling_speed_ms': v,
        'roller_radius_mm': R,
        'strip_width_mm': bm,
        'strain_epsilon': epsilon,
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
        'sigma_mg_MPa': sigma_mg,
        'sigma_al_MPa': sigma_al,
        'sigma_avg_MPa': sigma_avg,
        'force_basic_kN': F_basic_kN,
        'torque_basic_kNm': T_basic_kNm
    }
    
    save_calculation_data(input_data, results_data)
    
    print(f"\n=== Calculation Complete ===")
    print(f"All data and plots saved in current directory")
    print(f"Plots folder: {Path.cwd() / 'plots'}")
    print(f"Data folder: {Path.cwd() / 'data'}")

# ---------------------------- Main Execution ----------------------------
if __name__ == "__main__":
    perform_rolling_calculation()