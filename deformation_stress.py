import math

def deformation_resistance(A, m1, m2, m4, m5, m7, m8, phi, theta, phi_dot_mean):
    """
    Compute Deformation Resistance σ_F [MPa]

    This implements the Excel formula:
        = C * EXP(D*B) * POWER(O,E) * EXP(G/O) * POWER(1+O, H*B) * EXP(I*O) * POWER(AD, J*B)

    Parameter mapping (Excel -> Python):
        A  -> C (constant A)
        m1 -> D
        m2 -> E
        m4 -> G
        m5 -> H
        m7 -> I
        m8 -> J
        phi -> O (strain φ)
        theta -> B (ϑ [°C])
        phi_dot_mean -> AD (Mean Strain Rate φ'_m [s^-1])

    Returns:
        σ_F in MPa
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


if __name__ == "__main__":
    # Example using the values from your Excel row 4 to verify a match:
    example = {
        "A": 3518.6,
        "m1": -0.00327,
        "m2": 0.68022,
        "m4": 0.03841,
        "m5": -0.00121,
        "m7": -0.43501,
        "m8": 0.000186,
        "phi": 0.7985076962177716,       # strain (φ)
        "theta": 480,                     # temperature ϑ [°C]
        "phi_dot_mean": 7.982288549906957 # mean strain rate φ'_m [s^-1]
    }
    sigma_F = deformation_resistance(**example)
    print(f"σ_F (MPa) = {sigma_F:.6f}")
