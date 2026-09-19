"""
Single-Phase Incompressible (SPI) liquid flow model.

Reference: Waxman Section 2.2.1, Eq. 2.17.

The simplest of the four injector models and the baseline against
which everything else in the package is compared. SPI combines the
continuity equation (Eqs. 2.1-2.5), Bernoulli's equation (Eqs. 2.8-2.9),
and an empirical discharge coefficient Cd (Eq. 2.15) to give

    m_dot = Cd * A2 * sqrt( 2 * rho * dP )

where rho is the upstream liquid density and dP = P1 - P2. The model
treats the fluid as incompressible single-phase liquid: no flashing,
no phase change, no compressibility. It is valid when the chamber
pressure stays above the upstream vapour pressure (P2 >= Pv) so that
no cavitation occurs inside the orifice. Once P2 drops below Pv the
fluid flashes and one of the two-phase models (HEM, Burnell, Dyer)
becomes appropriate instead.

Typical Cd values fall between 0.6 and 0.9 depending on inlet
geometry; see the validation script for representative values from
Waxman's Table 3.1.
"""

import numpy as np


def mass_flow_rate(injector, rho_upstream, delta_P):
    """
    SPI mass flow rate through an injector orifice.

    Uses the finite-area form (Waxman Eq. 2.16) which retains the
    (1 - (A2/A1)^2) correction term in the denominator. For typical
    injector geometries A1 >> A2 and this collapses to the textbook
    simplified form (Eq. 2.17), but keeping the correction lets the
    same code handle injectors where the upstream area is not vastly
    larger than the orifice.

    """
    delta_P = np.asarray(delta_P, dtype=float)
    delta_P = np.maximum(delta_P, 0.0)

    denominator_correction = 1.0 - injector.area_ratio_squared()

    return (injector.Cd
            * injector.A2
            * np.sqrt(2.0 * rho_upstream * delta_P / denominator_correction))


def discharge_coefficient_from_measurement(injector_A2, rho_upstream,
                                           delta_P, measured_m_dot):
    """
    Back out an empirical discharge coefficient from a flow measurement.

    Rearranges m_dot = Cd * A2 * sqrt(2 * rho * dP) for Cd, giving

        Cd = m_dot / ( A2 * sqrt(2 * rho * dP) )

     measure m_dot at known rho and dP, divide out
    the theoretical SPI ceiling, and report whatever's left as Cd.

    """
    theoretical_max = injector_A2 * np.sqrt(2.0 * rho_upstream * delta_P)
    return measured_m_dot / theoretical_max


def supercharge_pressure(P1_pa, vapor_pressure_pa):
    """
    Supercharge pressure: how far the upstream sits above its boiling point.

    Waxman Eq. 2.18:

        P_super = P1 - Pv

    Positive values mean the upstream fluid has liquid margin; zero
    means it is exactly at saturation. Negative values are unphysical
    for a self-pressurised tank.

    """
    return P1_pa - vapor_pressure_pa