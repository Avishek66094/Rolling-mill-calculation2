def weighted_avg_stress(sigma_Mg, sigma_Al):
    """
    Weighted average deformation stress:
    (1*σ_Mg + 2.5*σ_Al) / 3.5
    """
    return (1.0 * sigma_Mg + 2.5 * sigma_Al) / 3.5


def roll_force_torque(sigma_avg, width_mm, delta_h_mm, roll_radius_mm):
    """
    Compute roll force [kN] and torque [kNm].

    Parameters
    ----------
    sigma_avg : float
        Average deformation stress [MPa]
    width_mm : float
        Strip width [mm]
    delta_h_mm : float
        Thickness reduction [mm]
    roll_radius_mm : float
        Roll radius [mm]

    Returns
    -------
    F_kN : float
        Roll force [kN]
    T_kNm : float
        Roll torque [kNm]
    """
    # Roll force [kN]
    F_kN = sigma_avg * width_mm * delta_h_mm / 1000.0

    # Roll torque [kNm]
    R_m = roll_radius_mm / 1000.0
    T_kNm = F_kN * R_m

    return F_kN, T_kNm


# ------------------- Example -------------------

# Suppose we already computed:
sigma_mg = 150.0   # MPa
sigma_al = 100.0   # MPa
sigma_avg = weighted_avg_stress(sigma_mg, sigma_al)

# Rolling parameters
width_mm = 100.0        # mm
delta_h_mm = 2.0        # mm thickness reduction
roll_radius_mm = 180.0  # mm

F, T = roll_force_torque(sigma_avg, width_mm, delta_h_mm, roll_radius_mm)
print(f"Average deformation stress = {sigma_avg:.2f} MPa")
print(f"Roll force = {F:.2f} kN")
print(f"Roll torque = {T:.2f} kNm")

