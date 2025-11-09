# roll_models.py
# Unified LB flow-stress for Al & Mg + f1/f2 built from chart PNGs using matplotlib clicks.
# Dependencies: numpy, matplotlib

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List
import os
import numpy as np

# ---------------- matplotlib (for digitizing the images) ----------------
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Number = Union[float, int, np.ndarray]

# =============================================================================
# Landolt–Börnstein flow stress (same for Al and Mg; only parameters differ)
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
# Kinematics helpers
# =============================================================================
def exit_thickness_from_strain(h0_mm: Number, phi: Number) -> np.ndarray:
    h0 = np.asarray(h0_mm, dtype=float)
    e  = np.asarray(phi, dtype=float)
    return h0 * np.exp(-e)

def reduction_percent_from_strain(phi: Number) -> np.ndarray:
    e = np.asarray(phi, dtype=float)
    r = 1.0 - np.exp(-e)
    return 100.0 * r

# =============================================================================
# f1/f2 coefficient tables & interpolators
# This version can DIGITIZE the values directly from PNGs using matplotlib.
# =============================================================================

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CHART_DIR = Path(r"C:\Avishek\MSc\Master's Thesis\Photos")
F1_BASENAME = "Roll pressure function"
F2_BASENAME = "Roll Torque function"

def _resolve_image_path(folder: Path, basename: str) -> str:
    exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = folder / f"{basename}{ext}"
        if p.exists():
            return str(p)
    # also accept a file that already includes an extension
    p = folder / basename
    if p.exists():
        return str(p)
    raise FileNotFoundError(
        f"Could not find an image for '{basename}' in '{folder}'. "
        f"Tried: {', '.join(exts)}"
    )

F1_IMAGE_PATH = os.environ.get("F1_IMAGE_PATH", "Roll pressure function.PNG")
F2_IMAGE_PATH = os.environ.get("F2_IMAGE_PATH", "Roll Torque function.PNG")

# Target grids (columns = reductions, rows = a-values)
_RPTS = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)  # reduction %
_A_VALUES = np.array([0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)        # a = μ sqrt(R'/h0)

# Storage (filled either by loading npz or by digitizing once)
F1_TABLE = None  # shape: (len(_A_VALUES), len(_RPTS))
F2_TABLE = None

def _bilinear_interp(x: float, y: float, x_grid: np.ndarray, y_grid: np.ndarray, z: np.ndarray) -> float:
    x = float(np.clip(x, x_grid.min(), x_grid.max()))
    y = float(np.clip(y, y_grid.min(), y_grid.max()))
    ix = int(np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid) - 2))
    iy = int(np.clip(np.searchsorted(y_grid, y) - 1, 0, len(y_grid) - 2))
    x0, x1 = x_grid[ix], x_grid[ix+1]
    y0, y1 = y_grid[iy], y_grid[iy+1]
    z00 = z[iy, ix]; z10 = z[iy, ix+1]
    z01 = z[iy+1, ix]; z11 = z[iy+1, ix+1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
    z0 = (1 - tx) * z00 + tx * z10
    z1 = (1 - tx) * z01 + tx * z11
    return float((1 - ty) * z0 + ty * z1)

def f1_roll_pressure_coeff(reduction_percent: float, a: float) -> float:
    r = float(np.clip(reduction_percent, _RPTS.min(), _RPTS.max()))
    return _bilinear_interp(r, a, _RPTS, _A_VALUES, F1_TABLE)

def f2_torque_coeff(reduction_percent: float, a: float) -> float:
    r = float(np.clip(reduction_percent, _RPTS.min(), _RPTS.max()))
    return _bilinear_interp(r, a, _RPTS, _A_VALUES, F2_TABLE)

# --------- Matplotlib digitizer (simple, click-based, one-time) ---------
def _calibrate_axis(ax_img, prompt: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Ask user to click two points on the axis in image coordinates, then
    input their real data values. Returns ((x_px0, x_px1), (x_val0, x_val1)).
    """
    print(f"\nCalibration for {prompt}:")
    print(" - Click first point on axis (e.g., x-min tick).")
    pts = plt.ginput(1, timeout=-1)
    assert pts, "No point clicked."
    x0, y0 = pts[0]
    print(" - Click second point on axis (e.g., x-max tick).")
    pts = plt.ginput(1, timeout=-1)
    assert pts, "No point clicked."
    x1, y1 = pts[0]

    v0 = float(input("   Enter data value at first clicked point: ").strip())
    v1 = float(input("   Enter data value at second clicked point: ").strip())
    if prompt.lower().startswith("y"):
        # For y, we only care about pixel y-coordinates
        return (y0, y1), (v0, v1)
    else:
        return (x0, x1), (v0, v1)

def _px_to_val(px: np.ndarray, px0: float, px1: float, v0: float, v1: float) -> np.ndarray:
    return ( (px - px0) / (px1 - px0) ) * (v1 - v0) + v0

def _digitize_curves(image_path: str, label: str, reductions_grid: np.ndarray, a_values: np.ndarray) -> np.ndarray:
    """
    Interactively digitize multiple curves (each for a specific a).
    Returns a table of shape (len(a_values), len(reductions_grid)).
    Steps:
      1) Show image.
      2) Calibrate x-axis (Reduction %) with two clicks + numeric values.
      3) Calibrate y-axis (f1 or f2) with two clicks + numeric values.
      4) For each 'a' value, click along the corresponding curve from left to right.
         Press ENTER when done with that curve to proceed to the next 'a'.
    """
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_title(f"{label}: Click to calibrate axes, then digitize curves.\n"
                 f"Use zoom/pan if needed; right-click cancels last point.")

    # Calibrate x (Reduction %)
    print("\n=== Axis calibration ===")
    print("1) X-axis (Reduction in pass, %)")
    (xpx0, xpx1), (xv0, xv1) = _calibrate_axis(ax, "x-axis (Reduction %)")
    # Calibrate y (f1 or f2)
    print("\n2) Y-axis (function value)")
    (ypx0, ypx1), (yv0, yv1) = _calibrate_axis(ax, "y-axis (function value)")

    table_rows: List[np.ndarray] = []
    print("\n=== Curve digitization ===")
    print("Instructions:")
    print(" - For each 'a' prompted in the console, click points along that curve (left→right).")
    print(" - Press ENTER (no clicks) to finish the current curve and move to the next.")
    print(" - Close the figure when all curves are finished.")
    for a in a_values:
        ax.set_title(f"{label}: Digitize curve for a = {a:.2f}\n"
                     f"Click points, then press ENTER to finish this curve.")
        plt.draw()
        pts: List[Tuple[float, float]] = []
        # Collect clicks until ENTER
        while True:
            clicks = plt.ginput(n=0, timeout=0)  # returns [] on ENTER
            if not clicks:
                break
            pts.extend(clicks)
            # Optional: show as you click
            xs_tmp = [p[0] for p in pts]
            ys_tmp = [p[1] for p in pts]
            ax.plot(xs_tmp, ys_tmp, ".", ms=4, color="red")
            plt.pause(0.01)

        if len(pts) < 2:
            raise RuntimeError(f"Too few points clicked for a={a}. Need at least 2.")

        xs_px = np.array([p[0] for p in pts], dtype=float)
        ys_px = np.array([p[1] for p in pts], dtype=float)

        # Convert pixels to data
        xs_val = _px_to_val(xs_px, xpx0, xpx1, xv0, xv1)
        ys_val = _px_to_val(ys_px, ypx0, ypx1, yv0, yv1)  # caution: y grows downward in images;
        # if your y-axis increases upward, swap (ypx0, ypx1) accordingly during calibration.

        # Sort by x and interpolate to the common reductions grid
        order = np.argsort(xs_val)
        xs_val = xs_val[order]
        ys_val = ys_val[order]
        # restrict to grid range
        mask = (xs_val >= reductions_grid.min()) & (xs_val <= reductions_grid.max())
        xs_val = xs_val[mask]; ys_val = ys_val[mask]
        if len(xs_val) < 2:
            raise RuntimeError(f"After trimming to grid range, too few points for a={a}.")
        row = np.interp(reductions_grid, xs_val, ys_val)
        table_rows.append(row)

    plt.close(fig)
    table = np.vstack(table_rows)  # rows correspond to a_values order
    return table

def _load_or_build_tables() -> None:
    """
    Populate F1_TABLE / F2_TABLE from disk if present, otherwise guide the user
    through a one-time digitization using matplotlib clicks.
    Saves results to 'f_tables.npz'.
    """
    global F1_TABLE, F2_TABLE

    if os.path.exists("f_tables.npz"):
        dat = np.load("f_tables.npz")
        # allow reloading even if custom grids were used previously
        a_vals = dat["A_values"]; rpts = dat["R_points"]
        table1 = dat["F1"]; table2 = dat["F2"]
        # If grids differ, resample to our canonical grids:
        if not (np.allclose(a_vals, _A_VALUES) and np.allclose(rpts, _RPTS)):
            # resample both tables to our grids
            F1_TABLE = _resample_table(table1, rpts, a_vals, _RPTS, _A_VALUES)
            F2_TABLE = _resample_table(table2, rpts, a_vals, _RPTS, _A_VALUES)
        else:
            F1_TABLE, F2_TABLE = table1, table2
        return

    # Otherwise build interactively:
    print("\nNo 'f_tables.npz' found. We will digitize the two charts using matplotlib.")
    print("Ensure the images exist:\n"
          f"  f1 image: {F1_IMAGE_PATH}\n"
          f"  f2 image: {F2_IMAGE_PATH}\n")

    # f1 (roll pressure)
    if not os.path.exists(F1_IMAGE_PATH):
        raise FileNotFoundError(f"Cannot find f1 image at '{F1_IMAGE_PATH}'.")
    F1_TABLE_DIG = _digitize_curves(F1_IMAGE_PATH, "Fig. 42 — f1 (Roll Pressure Function)",
                                    _RPTS, _A_VALUES)

    # f2 (torque)
    if not os.path.exists(F2_IMAGE_PATH):
        raise FileNotFoundError(f"Cannot find f2 image at '{F2_IMAGE_PATH}'.")
    F2_TABLE_DIG = _digitize_curves(F2_IMAGE_PATH, "Fig. 43 — f2 (Torque Function)",
                                    _RPTS, _A_VALUES)

    # Save for future runs
    np.savez("f_tables.npz", A_values=_A_VALUES, R_points=_RPTS, F1=F1_TABLE_DIG, F2=F2_TABLE_DIG)
    F1_TABLE, F2_TABLE = F1_TABLE_DIG, F2_TABLE_DIG
    print("Saved digitized tables to 'f_tables.npz'.")

def _resample_table(table: np.ndarray,
                    r_src: np.ndarray, a_src: np.ndarray,
                    r_dst: np.ndarray, a_dst: np.ndarray) -> np.ndarray:
    """Bilinear resample from (a_src, r_src) grid to (a_dst, r_dst)."""
    # First along reduction axis
    tmp = np.vstack([np.interp(r_dst, r_src, row) for row in table])
    # Then along 'a' axis (operate per column)
    out = np.vstack([
        np.interp(a_dst, a_src, tmp[:, j]) for j in range(tmp.shape[1])
    ]).T
    return out

# Build/load tables on import
_load_or_build_tables()

# =============================================================================
# Roll force & torque formulas
# =============================================================================
def roll_force_basic(sigma_avg_MPa: float, width_mm: float, delta_h_mm: float) -> float:
    """Return roll force in kN."""
    return (sigma_avg_MPa * width_mm * delta_h_mm) / 1000.0

def roll_torque_basic(F_kN: float, roll_radius_mm: float) -> float:
    """Return roll torque in kN·m."""
    return F_kN * (roll_radius_mm / 1000.0)

def roll_force_from_f1(f1: float, width_mm: float, h0_mm: float) -> float:
    """Coefficient-based force (kN). Adjust if your source uses a different prefactor."""
    return (f1 * width_mm * h0_mm) / 1000.0

def roll_torque_from_f2(f2: float, width_mm: float, h0_mm: float, R_eff_mm: float) -> float:
    """Coefficient-based torque (kN·m)."""
    return (f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0

def weighted_avg_stress(sigma_mg: Number, sigma_al: Number, w_mg: float = 1.0, w_al: float = 2.5) -> np.ndarray:
    return (w_mg * np.asarray(sigma_mg) + w_al * np.asarray(sigma_al)) / (w_mg + w_al)

# =============================================================================
# Minimal manual-input runner (your 4 inputs)
# =============================================================================
if __name__ == "__main__":
    print("=== Manual input (only the 4 parameters you requested) ===")
    h1 = float(input("Initial thickness h1 [mm]: ").strip())
    theta = float(input("Temperature θ [°C]: ").strip())
    phi = float(input("True strain φ (e.g., 0.20): ").strip())
    phidot = float(input("Strain rate φdot [1/s]: ").strip())

    # Geometry / reduction
    h2 = exit_thickness_from_strain(h1, phi)
    red_pct = float(reduction_percent_from_strain(phi))
    print(f"h_out = {h2:.3f} mm   |   reduction = {red_pct:.2f} %")

    # Flow stresses
    p_mg = MATERIALS["mg"]; p_al = MATERIALS["al"]
    sigma_mg = float(sigma_LB(p_mg, theta, phi, phidot))
    sigma_al = float(sigma_LB(p_al, theta, phi, phidot))
    sigma_avg = float(weighted_avg_stress(sigma_mg, sigma_al))
    print(f"σ_Mg(AZ31) = {sigma_mg:.2f} MPa   |   σ_Al(99.5) = {sigma_al:.2f} MPa")
    print(f"Weighted average σ = {sigma_avg:.2f} MPa")

    # Basic F & T
    width_mm = 100.0
    R_mm = 180.0
    F_basic = roll_force_basic(sigma_avg, width_mm, h1 - h2)
    T_basic = roll_torque_basic(F_basic, R_mm)
    print(f"[Basic]  F ≈ {F_basic:.2f} kN,   T ≈ {T_basic:.2f} kN·m")

    # Coefficient-based (uses your digitized f1/f2)
    mu = float(input("Friction μ (e.g., 0.25): ").strip() or "0.25")
    R_eff_mm = float(input("Effective roll radius R′ [mm] (≈R initially): ").strip() or str(R_mm))
    a = mu * np.sqrt(R_eff_mm / max(h1, 1e-9))
    f1 = f1_roll_pressure_coeff(red_pct, a)
    f2 = f2_torque_coeff(red_pct, a)
    F_c = roll_force_from_f1(f1, width_mm, h1)
    T_c = roll_torque_from_f2(f2, width_mm, h1, R_eff_mm)
    print(f"a = {a:.3f}  |  f1 = {f1:.3f}  |  f2 = {f2:.3f}")
    print(f"[Coeff] F ≈ {F_c:.2f} kN,   T ≈ {T_c:.2f} kN·m")
