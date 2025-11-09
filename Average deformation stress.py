def weighted_avg_stress(sigma_Mg, sigma_Al):
    """
    Compute the weighted average deformation stress:
    (1*σ_Mg + 2.5*σ_Al) / 3.5

    Parameters
    ----------
    sigma_Mg : float or array
        Deformation stress of Mg [MPa]
    sigma_Al : float or array
        Deformation stress of Al [MPa]

    Returns
    -------
    sigma_avg : float or array
        Weighted average deformation stress [MPa]
    """
    return (1.0 * sigma_Mg + 2.5 * sigma_Al) / 3.5


# Example:
sigma_mg = 150.0  # MPa
sigma_al = 100.0  # MPa

sigma_avg = weighted_avg_stress(sigma_mg, sigma_al)
print(f"Weighted average stress = {sigma_avg:.2f} MPa")
