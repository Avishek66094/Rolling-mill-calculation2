# roll_models.py — prompt first; digitize only the needed 'a' curve once; cache per-figure

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import math

Number = Union[float, int, np.ndarray]

# ---------------- tunables for a = μ * sqrt(R'/h1) ----------------
MU_DEFAULT = 0.25           # friction
R_EFF_DEFAULT_MM = 180.0    # effective roll radius (mm)
A_KEY_DECIMALS = 3         # how we round 'a' when naming cache rows (0.243 -> 0.243)

# ---------------- defaults for force/torque -----------------------
DEFAULT_WIDTH_MM = 100.0    # strip width (mm)
PRINT_TORQUE_IN_NM_TOO = True

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
SEARCH_DIRS = [
    Path(r"C:\Avishek\MSc\Master's Thesis\Photos"),  # your Windows folder
    Path("/mnt/data"),                               # fallback copies
]
F1_BASENAME = "Roll pressure function"
F2_BASENAME = "Roll Torque function"

# ---------------- grids (x: reduction %) --------------------------
_RPTS = np.array([5,10,20,30,40,50,60,70,80,90], dtype=float)

# (kept for future use)
@dataclass(frozen=True)
class LBParams:
    A: float; m1: float; m2: float; m4: float; m5: float; m7: float; m8: float
    theta_range: Tuple[float, float]
    phi_range: Tuple[float, float]
    phidot_range: Tuple[float, float]
    name: str = "material"

# ---------------- basic helpers ----------------
def reduction_percent_from_strain(phi: Number) -> np.ndarray:
    e = np.asarray(phi, dtype=float)
    return 100.0 * (1.0 - np.exp(-e))

def _resolve_image_path(basename: str) -> Path:
    exts = [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG",".bmp",".tif",".tiff"]
    for folder in SEARCH_DIRS:
        for ext in exts:
            p = folder / f"{basename}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find '{basename}.*' in any search dir.")

def _ask_float(prompt: str, default: Optional[float]=None) -> float:
    while True:
        s = input(prompt).strip()
        if s == "":
            if default is None:
                print("Please enter a number.")
                continue
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Not a number; try again.")

# ---------- calibration + single-curve digitizer ----------
def _calibrate_axis(ax, prompt: str, default_low: float, default_high: float):
    print(f"\nCalibration for {prompt}: click first tick (low), then second tick (high).")
    p1 = plt.ginput(1, timeout=-1); assert p1
    p2 = plt.ginput(1, timeout=-1); assert p2
    (x0,y0) = p1[0]; (x1,y1) = p2[0]
    v0 = _ask_float(f"  Value at first tick [default {default_low}]: ", default_low)
    v1 = _ask_float(f"  Value at second tick [default {default_high}]: ", default_high)
    if prompt.lower().startswith("y"):
        return (y0,y1), (v0,v1)
    return (x0,x1), (v0,v1)

def _px_to_val(px, px0, px1, v0, v1):
    return ((px - px0) / (px1 - px0)) * (v1 - v0) + v0

def _digitize_single_curve(image_path: Path, label: str,
                           reductions_grid: np.ndarray,
                           x_defaults=(0.0, 100.0), y_defaults=(0.0, 1.0)) -> np.ndarray:
    """
    Digitize ONE curve (for the current 'a'): click 6–12 points left→right, press ENTER.
    Returns values sampled on 'reductions_grid'.
    """
    img = mpimg.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f"{label}\nCalibrate axes, then click along THIS 'a' curve (ENTER to finish)")

    print("\n=== Axis calibration ===")
    (xpx0,xpx1), (xv0,xv1) = _calibrate_axis(ax, "x-axis (Reduction %)", x_defaults[0], x_defaults[1])
    (ypx0,ypx1), (yv0,yv1) = _calibrate_axis(ax, "y-axis (function value)", y_defaults[0], y_defaults[1])

    print("\n=== Curve digitization (single curve) ===")
    print("Tip: Click 6–12 points along the curve (left→right), then press ENTER.")
    pts: List[tuple[float,float]] = []
    while True:
        clicks = plt.ginput(n=0, timeout=0)
        if not clicks:
            break
        pts.extend(clicks)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ".", ms=4)
        plt.pause(0.01)

    plt.close(fig)

    if len(pts) < 2:
        raise RuntimeError("Too few points captured; please retry and click more points along the curve.")

    xs_px = np.array([p[0] for p in pts], float)
    ys_px = np.array([p[1] for p in pts], float)
    xs_val = _px_to_val(xs_px, xpx0, xpx1, xv0, xv1)
    ys_val = _px_to_val(ys_px, ypx0, ypx1, yv0, yv1)

    order = np.argsort(xs_val)
    xs_val, ys_val = xs_val[order], ys_val[order]
    mask = (xs_val >= reductions_grid.min()) & (xs_val <= reductions_grid.max())
    xs_val, ys_val = xs_val[mask], ys_val[mask]
    if len(xs_val) < 2:
        raise RuntimeError("Not enough points within the x-range; try clicking across the whole curve.")

    row = np.interp(reductions_grid, xs_val, ys_val)
    return row

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
    Strategy:
      - Look for exact a_key in cache → interpolate along r.
      - Else, if two cached a_keys bracket a_value → interpolate across a (no new clicking).
      - Else → ask user to digitize THIS a only; append to cache and use it.
    """
    cache_path = image_path.parent / cache_basename
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
        # find insertion point
        j = np.searchsorted(a_sorted, a_key)
        if 0 < j < len(a_sorted):
            il = a_sorted_idx[j-1]; ih = a_sorted_idx[j]
            al, ah = akeys[il], akeys[ih]
            fl, fh = val_at_r(il), val_at_r(ih)
            t = (a_key - al) / (ah - al)
            return (1.0 - t) * fl + t * fh

    # 3) need to digitize THIS a only
    print(f"\nNo cached curve for a={a_key:.{A_KEY_DECIMALS}f} in {fig_name}.")
    print("Please click points for THIS curve (once).")
    # Defaults: x 0–100 always; y depends on figure
    row = _digitize_single_curve(
        image_path, fig_name, _RPTS,
        x_defaults=(0.0, 100.0), y_defaults=y_defaults
    )

    # append to cache
    if table.size == 0:
        table = row[None, :]
        akeys = [a_key]
    else:
        table = np.vstack([table, row])
        akeys.append(a_key)
    _save_cache(cache_path, table, _RPTS, akeys)

    # return value at r
    return float(np.interp(r_query, _RPTS, row))

# ---------- public API: f1, f2 for given (r, a) ----------
def f1_value(r_percent: float, a_value: float) -> float:
    f1_img = _resolve_image_path(F1_BASENAME)
    return _ensure_curve_value(
        fig_name="Fig. 42 — f1 (Roll Pressure Function)",
        image_path=f1_img,
        y_defaults=(0.0, 90.0),
        a_value=a_value,
        r_query=float(np.clip(r_percent,  _RPTS.min(), _RPTS.max())),
        cache_basename="f1_cache.npz"
    )

def f2_value(r_percent: float, a_value: float) -> float:
    f2_img = _resolve_image_path(F2_BASENAME)
    return _ensure_curve_value(
        fig_name="Fig. 43 — f2 (Torque Function)",
        image_path=f2_img,
        y_defaults=(0.0, 0.7),
        a_value=a_value,
        r_query=float(np.clip(r_percent,  _RPTS.min(), _RPTS.max())),

        cache_basename="f2_cache.npz"
    )

# ---------- roll force & torque from coefficients ----------



# ---------------------------- CLI ----------------------------
if __name__ == "__main__":
    print("=== Input: h1, φ, reduction r, θ ===")
    try:
        h1  = float(input("Initial thickness h1 [mm]: ").strip())
        phi = float(input("True strain φ (e.g., 0.20): ").strip())
        r   = float(input("Reduction in pass r [%]: ").strip())
        theta = float(input("Temperature θ [°C]: ").strip())
    except Exception as e:
        raise SystemExit(f"Input error: {e}")

    # Compute 'a' automatically (no extra prompts)
    mu = MU_DEFAULT
    R_eff_mm = R_EFF_DEFAULT_MM
    a = mu * math.sqrt(R_eff_mm / max(h1, 1e-9))
    a_key = round(a, A_KEY_DECIMALS)

    # f1 & f2 — digitize only this 'a' curve once; cache and reuse thereafter
    f1 = f1_value(r, a)
    f2 = f2_value(r, a)

       # --- Coefficients already computed above: f1, f2, a, etc. ---
    print("\n--- Coefficients ---")
    print(f"h1 = {h1:.3f} mm | φ = {phi:.4f} | r = {r:.2f} % | θ = {theta:.1f} °C")
    print(f"a = μ * sqrt(R′/h1) = {mu:.3f} * sqrt({R_EFF_DEFAULT_MM:.1f}/{h1:.3f}) = {a:.3f}  (cache key: {a_key:.3f})")
    print(f"f1(r={r:.2f} %, a≈{a_key:.3f}) ≈ {f1:.6f}")
    print(f"f2(r={r:.2f} %, a≈{a_key:.3f}) ≈ {f2:.6f}")

    # --- Force & Torque from f1/f2, using defaults (no prompts) ---
    b_mm = DEFAULT_WIDTH_MM
    R_eff_mm = R_EFF_DEFAULT_MM

   # =============================================================================
# Roll force & torque
# =============================================================================
def roll_force_basic(sigma_avg_MPa: float, width_mm: float, delta_h_mm: float) -> float:
    return (sigma_avg_MPa * width_mm * delta_h_mm) / 1000.0  # kN

def roll_torque_basic(F_kN: float, roll_radius_mm: float) -> float:
    return F_kN * (roll_radius_mm / 1000.0)  # kN·m

def roll_force_from_f1(f1: float, width_mm: float, h0_mm: float) -> float:
    return (f1 * width_mm * h0_mm) / 1000.0  # kN

def roll_torque_from_f2(f2: float, width_mm: float, h0_mm: float, R_eff_mm: float) -> float:
    return (f2 * width_mm * h0_mm * (R_eff_mm / 1000.0)) / 1000.0  # kN·m

def weighted_avg_stress(sigma_mg: Number, sigma_al: Number, w_mg: float = 1.0, w_al: float = 2.5) -> np.ndarray:
    return (w_mg * np.asarray(sigma_mg) + w_al * np.asarray(sigma_al)) / (w_mg + w_al)

    # also show N·m to avoid "0.00" in kN·m for very small torques
    if PRINT_TORQUE_IN_NM_TOO:
        T_coeff_Nm = T_coeff_kNm * 1000.0
        print(f"[From f1/f2]  F ≈ {F_coeff_kN:.4f} kN   |   T ≈ {T_coeff_kNm:.6f} kN·m  ({T_coeff_Nm:.3f} N·m)")
    else:
        print(f"[From f1/f2]  F ≈ {F_coeff_kN:.4f} kN   |   T ≈ {T_coeff_kNm:.6f} kN·m")
