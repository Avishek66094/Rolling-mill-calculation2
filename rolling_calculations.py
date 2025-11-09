import math

def deformation_resistance(A, m1, m2, m4, m5, m7, m8, phi, theta, phi_dot_mean):
    """
    σ_F [MPa]
    Excel equivalence (AE):
      = C * EXP(D*B) * POWER(O,E) * EXP(G/O) * POWER(1+O, H*B) * EXP(I*O) * POWER(AD, J*B)
    """
    return (
        A
        * math.exp(m1 * theta)
        * (phi ** m2)
        * math.exp(m4 / phi)
        * ((1.0 + phi) ** (m5 * theta))
        * math.exp(m7 * phi)
        * (phi_dot_mean ** (m8 * theta))
    )

def roll_force_kN(sigmaF_avg, width_bm_mm, pressed_length_mm, f1):
    """
    Roll Force F [kN]
    Excel equivalence (AI):
      = (1.15 * AH * T * W * Z) / 10^3
    Mapping:
      AH -> sigmaF_avg [MPa]
      T  -> width_bm_mm [mm]
      W  -> pressed_length_mm [mm]
      Z  -> f1 [-]
    """
    return (1.15 * sigmaF_avg * width_bm_mm * pressed_length_mm * f1) / (10**3)

def roll_torque_kNm(sigmaF_avg, width_bm_mm, R_mm, h1_mm, h2_mm, f2):
    """
    Roll Torque τ [kN·m]
    Excel equivalence (AJ):
      = ((2*T*AH*U*(L^2 * AA)) / (M*1000)) / 1000
      = (2 * width * sigmaF_avg * R * (h1^2 * f2) / (h2 * 1000)) / 1000
    Mapping:
      AH -> sigmaF_avg [MPa]
      T  -> width_bm_mm [mm]
      U  -> R_mm [mm]
      L  -> h1_mm [mm]
      M  -> h2_mm [mm]
      AA -> f2 [-]
    """
    return (2.0 * width_bm_mm * sigmaF_avg * R_mm * ((h1_mm**2) * f2) / (h2_mm * 1000.0)) / 1000.0

if __name__ == "__main__":
    # Example values from your sheet (row 4) to validate:
    sigmaF_avg = 506.509643447995  # AH4 [MPa]
    width_bm_mm = 50               # T4 [mm]
    pressed_length_mm = 10.473657431862092  # W4 [mm]
    f1 = 2.11                      # Z4 [-]
    R_mm = 100                     # U4 [mm]
    h1_mm = 2.0                    # L4 [mm]
    h2_mm = 0.9                    # M4 [mm]
    f2 = 0.23                      # AA4 [-]

    F_kN = roll_force_kN(sigmaF_avg, width_bm_mm, pressed_length_mm, f1)
    tau_kNm = roll_torque_kNm(sigmaF_avg, width_bm_mm, R_mm, h1_mm, h2_mm, f2)

    print(f"Roll Force F [kN] = {F_kN:.9f}")
    print(f"Roll Torque τ [kN·m] = {tau_kNm:.9f}")
