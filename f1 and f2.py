# roll_models.py
# CLI: r[%], phi, h1[mm], theta[°C]  → computes a = μ * sqrt(R'/h1) automatically,
# then interpolates f1 (Fig.42) and f2 (Fig.43) from digitized charts.
# One-time digitization is cached as f_tables.npz in the same image folder.

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

Number = Union[float, int, np.ndarray]

# -------------------- tunables (used for a = μ * sqrt(R'/h1)) --------------------
MU_DEFAULT = 0.25         # friction coefficient (edit if you have a better estimate)
R_EFF_DEFAULT_MM = 180.0  # effective roll radius in mm (≈ R)

# =============================================================================
# Landolt–Börnstein flow stress (kept for completeness)
# =============================================================================
@dataclass(frozen=True)
class LBParams:
    A: float; m1: float; m2: float; m4: float; m5: float; m7: float; m8: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    phidot_range: Tuple[float, float]
    name: str = "material"

def sigma_LB(p: LBParams, theta_C: Number, phi: Number, phidot: Number, strict: bool = True) -> np.ndarray:
    theta = np.asarray(theta_C, dtype=float)
    e     = np.asarray(phi, dtype=float)
    edot  = np.asarray(phidot, dtype=float)
    if np.any(e <= 0.0):    raise ValueError("phi (true strain) must be > 0 due to exp(m4/phi).")
    if np.any(edot <= 0.0): raise ValueError("phidot (strain rate) must be > 0.")
    if strict:
        if np.any((theta < p.theta_range[0]) | (theta > p.theta_range[1])):
            raise ValueError(f"{p.name}: θ out of valid range {p.theta_range} °C.")
        if np.any((e < p.phi_range[0]) | (e > p.phi_range[1])):
            raise ValueError(f"{p.name}: φ out of valid range {p.phi_range}.")
        if np.any((edot < p.phidot_range[0]) | (edot > p.phidot_range[1])):
            raise ValueError(f"{p.name}: φdot out of valid range {p.phidot_range} 1/s.")
    return (
        p.A * np.exp(p.m1 * theta) * (e ** p.m2) * np.exp(p.m4 / e)
        * ((1.0 + e) ** p.m5) * np.exp(p.m7 * e) * (edot ** p.m8)
    )

# ---------------- Materials (not used directly in CLI below, kept for later) -----
AL995_HOT_DEF = LBParams(367.651, -0.00463, 0.32911,  0.00167, -0.00207,  0.16592, 0.000241,
                         (250.0, 550.0), (0.03, 1.50), (0.01, 500.0), "Al 99.5 (hot, deformed)")
AZ31_HOT_DIRECT = LBParams(961.667, -0.00640, 0.04403, -0.00718, -0.00042, -0.21096, 0.000435,
                           (280.0, 450.0), (0.03, 0.75), (0.01, 100.0), "AZ31 Mg (hot, direct)")

MATERIALS: Dict[str, LBParams] = {
    "al": AL995_HOT_DEF, "al995": AL995_HOT_DEF,
    "mg": AZ31_HOT_DIRECT, "az31": AZ31_HOT_DIRECT,
}

# =============================================================================
# Kinematics helpers
# =============================================================================
def exit_thickness_from_strain(h0_mm: Number, phi: Number) -> np.ndarray:
    h0 = np.asarray(h0_mm, dtype=float); e = np.asarray(phi, dtype=float)
    return h0 * np.exp(-e)

def reduction_percent_from_strain(phi: Number) -> np.ndarray:
    e = np.asarray(phi, dtype=float)
    return 100.0 * (1.0 - np.exp(-e))

# =============================================================================
# f1/f2 via one-time digitization from your local images
# =============================================================================
CHART_DIR = Path(r"C:\Avishek\MSc\Master's Thesis\Photos")
F1_BASENAME = "Roll pressure function"
F2_BASENAME = "Roll Torque function"

def _resolve_image_path(folder: Path, basename: str) -> Path:
    exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG", ".bmp", ".tif", ".tiff"]
    for ext in exts:
        p = folder / f"{basename}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find an image for '{basename}' in '{folder}'. "
        f"Tried extensions: {', '.join(exts)}"
    )

F1_IMAGE_PATH = _resolve_image_path(CHART_DIR, F1_BASENAME)
F2_IMAGE_PATH = _resolve_image_path(CHART_DIR, F2_BASENAME)

# Target grids (x: reduction %, y: a)
_RPTS = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
_A_VALUES = np.array([0.10, 0.20, 0.30, 0.40, 0.50], dtype=float)

F1_TABLE: np.ndarray | None = None
F2_TABLE: np.ndarray | None = None

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
    if F1_TABLE is None:
        raise RuntimeError("F1_TABLE not loaded.")
    r = float(np.clip(reduction_percent, _RPTS.min(), _RPTS.max()))
    a_clamped = float(np.clip(a, _A_VALUES.min(), _A_VALUES.max()))
    return _bilinear_interp(r, a_clamped, _RPTS, _A_VALUES, F1_TABLE)

def f2_torque_coeff(reduction_percent: float, a: float) -> float:
    if F2_TABLE is None:
        raise RuntimeError("F2_TABLE not loaded.")
    r = float(np.clip(reduction_percent, _RPTS.min(), _RPTS.max()))
    a_clamped = float(np.clip(a, _A_VALUES.min(), _A_VALUES.max()))
    return _bilinear_interp(r, a_clamped, _RPTS, _A_VALUES, F2_TABLE)

# --------- Matplotlib click-digitizer ---------
def _calibrate_axis(ax, prompt: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    print(f"\nCalibration for {prompt}:")
    print(" - Click first tick mark on this axis.")
    p1 = plt.ginput(1, timeout=-1); assert p1
    print(" - Click second tick mark on this axis.")
    p2 = plt.ginput(1, timeout=-1); assert p2
    (x0, y0) = p1[0]; (x1, y1) = p2[0]
    v0 = float(input("   Enter data value at the FIRST clicked point: ").strip())
    v1 = float(input("   Enter data value at the SECOND clicked point: ").strip())
    if prompt.lower().startswith("y"):
        return (y0, y1), (v0, v1)
    else:
        return (x0, x1), (v0, v1)

def _px_to_val(px: np.ndarray, px0: float, px1: float, v0: float, v1: float) -> np.ndarray:
    return ((px - px0) / (px1 - px0)) * (v1 - v0) + v0

def _digitize_curves(image_path: Path, label: str,
                     reductions_grid: np.ndarray, a_values: np.ndarray) -> np.ndarray:
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f"{label}: Calibrate axes, then digitize curves.\n"
                 f"Click along each 'a' curve (left→right). Press ENTER to finish a curve.")
    print("\n=== Axis calibration ===")
    print("1) X-axis (Reduction in pass, %)")
    (xpx0, xpx1), (xv0, xv1) = _calibrate_axis(ax, "x-axis (Reduction %)")
    print("\n2) Y-axis (function value)")
    (ypx0, ypx1), (yv0, yv1) = _calibrate_axis(ax, "y-axis (function value)")

    rows: List[np.ndarray] = []
    print("\n=== Curve digitization ===")
    for a in a_values:
        ax.set_title(f"{label}: Digitize curve for a = {a:.2f}  (click points, ENTER to finish)")
        plt.draw()
        pts: List[Tuple[float, float]] = []
        while True:
            clicks = plt.ginput(n=0, timeout=0)
            if not clicks:   # ENTER
                break
            pts.extend(clicks)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=4, color="red")
            plt.pause(0.01)

        if len(pts) < 2:
            raise RuntimeError(f"Too few points for a={a}.")
        xs_px = np.array([p[0] for p in pts], dtype=float)
        ys_px = np.array([p[1] for p in pts], dtype=float)

        xs_val = _px_to_val(xs_px, xpx0, xpx1, xv0, xv1)
        ys_val = _px_to_val(ys_px, ypx0, ypx1, yv0, yv1)  # watch axis orientation

        order = np.argsort(xs_val)
        xs_val = xs_val[order]; ys_val = ys_val[order]
        mask = (xs_val >= reductions_grid.min()) & (xs_val <= reductions_grid.max())
        xs_val = xs_val[mask]; ys_val = ys_val[mask]
        row = np.interp(reductions_grid, xs_val, ys_val)
        rows.append(row)

    plt.close(fig)
    return np.vstack(rows)

def _resample_table(table: np.ndarray,
                    r_src: np.ndarray, a_src: np.ndarray,
                    r_dst: np.ndarray, a_dst: np.ndarray) -> np.ndarray:
    tmp = np.vstack([np.interp(r_dst, r_src, row) for row in table])
    out = np.vstack([np.interp(a_dst, a_src, tmp[:, j]) for j in range(tmp.shape[1])]).T
    return out

def _load_or_build_tables() -> None:
    """Loads f1 & f2 from cache or runs a one-time digitization."""
    global F1_TABLE, F2_TABLE
    cache = CHART_DIR / "f_tables.npz"
    if cache.exists():
        dat = np.load(str(cache))
        a_vals = dat["A_values"]; rpts = dat["R_points"]
        t1 = dat["F1"]; t2 = dat["F2"]
        if not (np.allclose(a_vals, _A_VALUES) and np.allclose(rpts, _RPTS)):
            F1_TABLE = _resample_table(t1, rpts, a_vals, _RPTS, _A_VALUES)
            F2_TABLE = _resample_table(t2, rpts, a_vals, _RPTS, _A_VALUES)
        else:
            F1_TABLE, F2_TABLE = t1, t2
        return

    print("\nNo cached 'f_tables.npz' found in your chart folder. We'll digitize once.")
    print(f"f1 image: {F1_IMAGE_PATH}\nf2 image: {F2_IMAGE_PATH}")

    F1_TABLE_DIG = _digitize_curves(F1_IMAGE_PATH, "Fig. 42 — f1 (Roll Pressure Function)", _RPTS, _A_VALUES)
    F2_TABLE_DIG = _digitize_curves(F2_IMAGE_PATH, "Fig. 43 — f2 (Torque Function)", _RPTS, _A_VALUES)

    np.savez(str(cache), A_values=_A_VALUES, R_points=_RPTS, F1=F1_TABLE_DIG, F2=F2_TABLE_DIG)
    F1_TABLE, F2_TABLE = F1_TABLE_DIG, F2_TABLE_DIG
    print(f"Saved digitized tables to '{cache}'.")

# Build/load tables on import
_load_or_build_tables()

# =============================================================================
# Simple roll force/torque helpers (optional)
# =============================================================================
def roll_force_from_f1(f1: float, width_mm: float, h0_mm: float) -> float:
    return (f1 * width_mm * h0_mm) / 1000.0  # kN

def roll_torque_from_f2(f2: float, width_mm: float, h0_mm: float, R_eff_mm: float) -> float:
    return (f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0  # kN·m

# =============================================================================
# CLI — YOUR 4 INPUTS
# =============================================================================
if __name__ == "__main__":
    print("=== Rolling coefficients lookup (f1 & f2 from charts) ===")
    try:
        r_str = input("Reduction in pass r [%] (leave blank to compute from φ): ").strip()
        phi   = float(input("True strain φ (e.g., 0.20): ").strip())
        h1    = float(input("Initial thickness h1 [mm]: ").strip())
        theta = float(input("Temperature θ [°C]: ").strip())  # kept for completeness
    except Exception as e:
        raise SystemExit(f"Input error: {e}")

    # If r not given, compute from φ
    if r_str:
        r_pct = float(r_str)
    else:
        r_pct = float(reduction_percent_from_strain(phi))
        print(f"Computed reduction from φ: r ≈ {r_pct:.2f} %")

    # Automatic 'a' using current inputs and defaults (no extra prompts)
    mu = MU_DEFAULT
    R_eff_mm = R_EFF_DEFAULT_MM
    a = mu * np.sqrt(R_eff_mm / max(h1, 1e-9))

    # Interpolate f1 and f2 from digitized tables
    f1 = f1_roll_pressure_coeff(r_pct, a)
    f2 = f2_torque_coeff(r_pct, a)

    print("\n--- Results ---")
    print(f"h1 = {h1:.3f} mm | φ = {phi:.4f} | r = {r_pct:.2f} % | θ = {theta:.1f} °C")
    print(f"a = μ * sqrt(R′/h1) = {mu:.3f} * sqrt({R_eff_mm:.1f}/{h1:.3f}) = {a:.3f}")
    print(f"f1(reduction={r_pct:.2f} %, a={a:.3f}) ≈ {f1:.4f}")
    print(f"f2(reduction={r_pct:.2f} %, a={a:.3f}) ≈ {f2:.4f}")

    # Optional sanity note on valid ranges:
    print("\nGrid clamps:")
    print(f"  reduction grid: {float(_RPTS.min())}–{float(_RPTS.max())} %")
    print(f"  a grid:         {float(_A_VALUES.min())}–{float(_A_VALUES.max())}")
