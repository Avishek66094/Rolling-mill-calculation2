# roll_models.py — prompt first, then load/digitize charts only if needed

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

Number = Union[float, int, np.ndarray]

# ---------------- tunables for a = μ * sqrt(R'/h1) ----------------
MU_DEFAULT = 0.25          # friction
R_EFF_DEFAULT_MM = 180.0   # effective roll radius (m

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

# ---------------- image folders & basenames -----------------------
# 1) Your Windows path  2) Fallback to container copies
SEARCH_DIRS = [
    Path(r"C:\Avishek\MSc\Master's Thesis\Photos"),
    Path("/mnt/data"),
]
F1_BASENAME = "Roll pressure function"
F2_BASENAME = "Roll Torque function"

# ---------------- grids (x: reduction %, y: a) --------------------
_RPTS     = np.array([5,10,20,30,40,50,60,70,80,90], dtype=float)
_A_VALUES = np.array([0.10,0.20,0.30,0.40,0.50],     dtype=float)

# (kept for future use)
@dataclass(frozen=True)
class LBParams:
    A: float; m1: float; m2: float; m4: float; m5: float; m7: float; m8: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    phidot_range: Tuple[float, float]
    name: str = "material"

# ---------------- helpers ----------------
def reduction_percent_from_strain(phi: Number) -> np.ndarray:
    e = np.asarray(phi, dtype=float)
    return 100.0 * (1.0 - np.exp(-e))

def _resolve_image_path(basename: str) -> Path:
    """Try all SEARCH_DIRS and common extensions; return the first match."""
    exts = [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".bmp",".tif",".tiff"]
    tried = []
    for folder in SEARCH_DIRS:
        for ext in exts:
            p = folder / f"{basename}{ext}"
            tried.append(str(p))
            if p.exists():
                return p
    raise FileNotFoundError("Image not found. Tried:\n" + "\n".join(tried))

def _bilinear_interp(x: float, y: float, x_grid: np.ndarray, y_grid: np.ndarray, z: np.ndarray) -> float:
    x = float(np.clip(x, x_grid.min(), x_grid.max()))
    y = float(np.clip(y, y_grid.min(), y_grid.max()))
    ix = int(np.clip(np.searchsorted(x_grid, x) - 1, 0, len(x_grid)-2))
    iy = int(np.clip(np.searchsorted(y_grid, y) - 1, 0, len(y_grid)-2))
    x0,x1 = x_grid[ix], x_grid[ix+1]
    y0,y1 = y_grid[iy], y_grid[iy+1]
    z00 = z[iy,ix]; z10 = z[iy,ix+1]; z01 = z[iy+1,ix]; z11 = z[iy+1,ix+1]
    tx = 0.0 if x1==x0 else (x-x0)/(x1-x0)
    ty = 0.0 if y1==y0 else (y-y0)/(y1-y0)
    z0 = (1-tx)*z00 + tx*z10
    z1 = (1-tx)*z01 + tx*z11
    return float((1-ty)*z0 + ty*z1)

# ---------- digitizer (only called if no cache exists) ----------
def _calibrate_axis(ax, prompt: str):
    print(f"\nCalibration for {prompt}: click first tick, then second tick.")
    p1 = plt.ginput(1, timeout=-1); assert p1
    p2 = plt.ginput(1, timeout=-1); assert p2
    (x0,y0) = p1[0]; (x1,y1) = p2[0]
    v0 = float(input("  Value at first tick: ").strip())
    v1 = float(input("  Value at second tick: ").strip())
    if prompt.lower().startswith("y"):
        return (y0,y1), (v0,v1)
    return (x0,x1), (v0,v1)

def _px_to_val(px, px0, px1, v0, v1):
    return ((px - px0) / (px1 - px0)) * (v1 - v0) + v0

def _digitize_curves(image_path: Path, label: str,
                     reductions_grid: np.ndarray, a_values: np.ndarray) -> np.ndarray:
    """
    Interactive digitization:
      - For each a, click along the curve from left→right (at least ~6 points recommended).
      - Press ENTER when finished with that curve.
      - If you pressed ENTER too early, you'll be prompted to Retry (r), Skip (s), or Quit (q).
    Skipped curves are filled afterwards by interpolation across 'a'.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f"{label}\nCalibrate axes, then digitize each a-curve (ENTER to finish)")

    print("\n=== Axis calibration ===")
    (xpx0, xpx1), (xv0, xv1) = _calibrate_axis(ax, "x-axis (Reduction %)")
    (ypx0, ypx1), (yv0, yv1) = _calibrate_axis(ax, "y-axis (function value)")

    rows: List[np.ndarray] = []
    MIN_CLICKS = 2  # mathematically minimum; ~6+ recommended for a smooth fit

    print("\n=== Curve digitization ===")
    print("Tip: Click 6–12 points along each curve (left→right), then press ENTER.")
    for idx, a in enumerate(a_values):
        while True:
            # clear any previous red points for a clean retry
            [l.remove() for l in ax.lines[:]]
            ax.set_title(f"{label} — curve a={a:.2f}: click points; ENTER to finish "
                         f"(r=retry, s=skip, q=quit if prompted)")
            plt.draw()

            pts: List[tuple[float, float]] = []
            while True:
                clicks = plt.ginput(n=0, timeout=0)  # ENTER returns []
                if not clicks:
                    break
                pts.extend(clicks)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=4)
                plt.pause(0.01)

            if len(pts) < MIN_CLICKS:
                print(f"\nToo few points for a={a:.2f}. Retry (r), Skip (s), or Quit (q)?")
                ans = (input("[r/s/q]: ").strip().lower() or "r")
                if ans == "r":
                    continue
                if ans == "s":
                    # store NaNs to fill later across the 'a' dimension
                    rows.append(np.full_like(reductions_grid, np.nan, dtype=float))
                    break
                if ans == "q":
                    plt.close(fig)
                    raise SystemExit("Digitization aborted by user.")
                # default: retry
                continue

            # Convert pixel → data values
            xs_px = np.array([p[0] for p in pts], dtype=float)
            ys_px = np.array([p[1] for p in pts], dtype=float)
            xs_val = _px_to_val(xs_px, xpx0, xpx1, xv0, xv1)
            ys_val = _px_to_val(ys_px, ypx0, ypx1, yv0, yv1)

            # Sort & clip to grid range, then resample along reductions_grid
            order = np.argsort(xs_val)
            xs_val, ys_val = xs_val[order], ys_val[order]
            mask = (xs_val >= reductions_grid.min()) & (xs_val <= reductions_grid.max())
            xs_val, ys_val = xs_val[mask], ys_val[mask]
            if len(xs_val) < 2:
                print(f"\nAfter clipping to x-grid, still too few points for a={a:.2f}. Retry or Skip?")
                ans = (input("[r/s]: ").strip().lower() or "r")
                if ans == "s":
                    rows.append(np.full_like(reductions_grid, np.nan, dtype=float))
                    break
                else:
                    continue

            row = np.interp(reductions_grid, xs_val, ys_val)
            rows.append(row)
            break  # next a

    plt.close(fig)
    table = np.vstack(rows)

    # Fill any skipped (NaN) rows by interpolation along 'a'
    if np.isnan(table).any():
        A_idx = np.arange(len(a_values))
        for j in range(table.shape[1]):  # for each reduction column
            col = table[:, j]
            good = ~np.isnan(col)
            # guard: if first/last are NaN, extend with nearest good value
            if not good.any():
                raise RuntimeError("All curves skipped; cannot interpolate.")
            # use 1D interpolation across a; extrapolate ends with nearest valid
            xg, yg = A_idx[good], col[good]
            interp = np.interp(A_idx, xg, yg)
            # For strict nearest at the ends, overwrite extremes explicitly
            # (np.interp already does edge holding)
            table[:, j] = interp

    return table

def _load_or_build_tables() -> tuple[np.ndarray, np.ndarray]:
    """
    Lazy loader: tries cache next to the f1 image; if missing, digitize once.
    Called ONLY after the CLI inputs have been read.
    """
    f1_img = _resolve_image_path(F1_BASENAME)
    f2_img = _resolve_image_path(F2_BASENAME)

    # store cache beside the image actually used (Windows folder if present)
    cache = f1_img.parent / "f_tables.npz"

    if cache.exists():
        dat = np.load(str(cache))
        a_vals = dat["A_values"]; rpts = dat["R_points"]
        t1 = dat["F1"]; t2 = dat["F2"]
        if not (np.allclose(a_vals, _A_VALUES) and np.allclose(rpts, _RPTS)):
            t1 = _resample_table(t1, rpts, a_vals, _RPTS, _A_VALUES)
            t2 = _resample_table(t2, rpts, a_vals, _RPTS, _A_VALUES)
        return t1, t2

    print("\nNo cached 'f_tables.npz' found. Digitizing the two figures once.")
    print(f"Using images:\n  f1 → {f1_img}\n  f2 → {f2_img}")
    f1_tab = _digitize_curves(f1_img, "Fig. 42 — f1 (Roll Pressure Function)", _RPTS, _A_VALUES)
    f2_tab = _digitize_curves(f2_img, "Fig. 43 — f2 (Torque Function)",        _RPTS, _A_VALUES)
    np.savez(str(cache), A_values=_A_VALUES, R_points=_RPTS, F1=f1_tab, F2=f2_tab)
    print(f"Saved digitized tables to '{cache}'.")
    return f1_tab, f2_tab

# --- Force/Torque from coefficients ---
def roll_force_from_f1(f1: float, width_mm: float, h0_mm: float) -> float:
    """kN"""
    return (f1 * width_mm * h0_mm) / 1000.0

def roll_torque_from_f2(f2: float, width_mm: float, h0_mm: float, R_eff_mm: float) -> float:
    """kN·m"""
    return (f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0

# --- (optional) flow-stress based estimate like before ---
def roll_force_basic(sigma_avg_MPa: float, width_mm: float, delta_h_mm: float) -> float:
    """kN"""
    return (sigma_avg_MPa * width_mm * delta_h_mm) / 1000.0

def roll_torque_basic(F_kN: float, roll_radius_mm: float) -> float:
    """kN·m"""
    return F_kN * (roll_radius_mm / 1000.0)

def weighted_avg_stress(sigma_mg: float, sigma_al: float, w_mg: float = 1.0, w_al: float = 2.5) -> float:
    return (w_mg * sigma_mg + w_al * sigma_al) / (w_mg + w_al)


# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    print("=== Input: h1, reduction r, strain φ ===")
    try:
        h1  = float(input("Initial thickness h1 [mm]: ").strip())
        r   = float(input("Reduction in pass r [%]: ").strip())
        phi = float(input("True strain φ (e.g., 0.20): ").strip())
    except Exception as e:
        raise SystemExit(f"Input error: {e}")

    # Compute 'a' automatically (no prompts)
    mu = MU_DEFAULT
    R_eff_mm = R_EFF_DEFAULT_MM
    a = mu * np.sqrt(R_eff_mm / max(h1, 1e-9))

    # Now load/digitize after inputs
    F1_TABLE, F2_TABLE = _load_or_build_tables()

    # Clamp and interpolate
    a_c = float(np.clip(a, _A_VALUES.min(), _A_VALUES.max()))
    r_c = float(np.clip(r,  _RPTS.min(),     _RPTS.max()))

    def _interp(z: np.ndarray) -> float:
        return _bilinear_interp(r_c, a_c, _RPTS, _A_VALUES, z)

    f1 = _interp(F1_TABLE)
    f2 = _interp(F2_TABLE)

    print("\n--- Results ---")
    print(f"h1 = {h1:.3f} mm | r = {r:.2f} % | φ = {phi:.4f}")
    print(f"a = μ * sqrt(R′/h1) = {mu:.3f} * sqrt({R_eff_mm:.1f}/{h1:.3f}) = {a:.3f} (clamped → {a_c:.3f})")
    print(f"f1(reduction={r_c:.2f} %, a={a_c:.3f}) ≈ {f1:.4f}")
    print(f"f2(reduction={r_c:.2f} %, a={a_c:.3f}) ≈ {f2:.4f}")

    # --- Force & Torque from f1/f2 ---
    # Width and R′ can be changed here, otherwise defaults are used.
    width_input = input("Strip width b [mm] (default 100): ").strip()
    b_mm = float(width_input) if width_input else 100.0

    Reff_input = input(f"Effective roll radius R′ [mm] (default {R_eff_mm:.1f}): ").strip()
    R_eff_mm = float(Reff_input) if Reff_input else R_eff_mm

    F_coeff = roll_force_from_f1(f1, b_mm, h1)                   # kN
    T_coeff = roll_torque_from_f2(f2, b_mm, h1, R_eff_mm)        # kN·m

    print(f"[Coeff]  F ≈ {F_coeff:.2f} kN   |   T ≈ {T_coeff:.2f} kN·m")

    # --- Now ask for Temperature (as you requested) ---
    theta = float(input("Temperature θ [°C]: ").strip())

    # (Optional) If you also want a flow-stress-based estimate, enter φdot (or leave blank to skip)
    phidot_str = input("Strain rate φdot [1/s] (optional, Enter to skip): ").strip()
    if phidot_str:
        phidot = float(phidot_str)

        # Example: use your two LB materials defined above
        p_mg = MATERIALS["mg"]; p_al = MATERIALS["al"]
        sigma_mg = float(sigma_LB(p_mg, theta, phi, phidot, strict=False))
        sigma_al = float(sigma_LB(p_al, theta, phi, phidot, strict=False))
        sigma_avg = float(weighted_avg_stress(sigma_mg, sigma_al))

        # basic F/T using average flow stress and an assumed barrel radius (reuse R_eff_mm)
        h_out = h1 * np.exp(-phi)
        F_basic = roll_force_basic(sigma_avg, b_mm, h1 - h_out)
        T_basic = roll_torque_basic(F_basic, R_eff_mm)

        print(f"σ_Mg(AZ31) = {sigma_mg:.1f} MPa | σ_Al(99.5) = {sigma_al:.1f} MPa | σ̄ = {sigma_avg:.1f} MPa")
        print(f"[Basic]  F ≈ {F_basic:.2f} kN   |   T ≈ {T_basic:.2f} kN·m")
