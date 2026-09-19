"""
Burnell frozen non-equilibrium two-phase flow model.

Reference: Waxman Section 2.2.2.1.1, Eq. 2.52.

Burnell sits at the opposite extreme from HEM. Where HEM assumes the
fluid is in continuous thermodynamic equilibrium and flashes the
moment its pressure drops below saturation, Burnell assumes the
fluid stays single-phase liquid through the entire orifice and only
flashes at the exit plane when it suddenly meets the low chamber
pressure P₂. Because the fluid remains liquid inside, the flow
never goes sonic inside, and no choking occurs internally — the
fundamental structural difference from HEM.

To acknowledge that exit-plane flashing actually does happen,
Burnell tacks an empirical correction factor onto the SPI form:

    C = sqrt( P₂ / Pv )    when P₂ <  Pv   (cavitating exit)
    C = 1                  when P₂ >= Pv   (no cavitation, Burnell = SPI)
    m_burnell = C * Cd * A₂ * sqrt( 2 * p_liquid * (P₁ - P₂) )

Burnell is therefore SPI with one prefactor. No enthalpy, no entropy,
no optimiser — just an algebraic expression evaluated once. It is
vectorised across P₂ and runs at SPI speed.

Burnell tends to do best on short orifices (low L/D) where the fluid
exits before it has time to flash internally, matching the frozen
assumption. Long orifices give the fluid more residence time and
push the real behaviour toward HEM, where Burnell underperforms.
"""

import numpy as np
from CoolProp.CoolProp import PropsSI

from fluids import get_vapor_pressure


def correction_factor(P2_pa, Pv_pa):
    """
    Burnell's empirical exit-flashing correction factor C.

        C = sqrt( P₂ / Pv )  when P₂ <  Pv
        C = 1.0              when P₂ >= Pv

    Vectorised across P2_pa.

    """
    P2_pa = np.asarray(P2_pa, dtype=float)

    return np.where(
        P2_pa >= Pv_pa,
        1.0,
        np.sqrt(np.clip(P2_pa / Pv_pa, 0.0, 1.0)),
    )


def mass_flow_rate(injector, fluid, P1_pa, T1_k, P2_pa):
    """

        m_burnell = C * Cd * A₂ * sqrt( 2 * p_liquid * (P1 - P2) )

    Liquid density is evaluated at upstream conditions, with P1 inset
    slightly above its nominal value to avoid the saturation boundary
    where CoolProp's (P, T) lookup is ambiguous between phases.

    """
    Pv_pa      = get_vapor_pressure(fluid, T1_k)
    C          = correction_factor(P2_pa, Pv_pa)
    rho_liquid = PropsSI('D', 'P', P1_pa * 1.0001, 'T', T1_k, fluid)
    delta_P    = np.maximum(P1_pa - P2_pa, 0.0)

    return (C * injector.Cd * injector.A2
            * np.sqrt(2.0 * rho_liquid * delta_P))


def mass_flow_rate_sweep(injector, fluid, P1_pa, T1_k, P2_array_pa):
    """
    Burnell mass flow rate over an array of chamber pressures.
    Burnell is already vectorised across P2_pa, so this just calls 
    mass_flow_rate once with the whole array.
    """
    return mass_flow_rate(injector, fluid, P1_pa, T1_k, P2_array_pa)