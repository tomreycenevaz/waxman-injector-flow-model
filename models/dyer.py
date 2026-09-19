"""
Dyer non-equilibrium two-phase flow model.

Reference: Waxman Section 2.2.2.2.1, Figs. 2.43–2.46, Eq. 2.57.

Dyer sits between HEM and SPI rather than committing to either
extreme. HEM assumes the fluid is in perfect thermodynamic
equilibrium and flashes instantly; SPI assumes the fluid never
flashes. Real injectors do something in between, with the balance
controlled by how much liquid margin the upstream condition has.
Dyer captures this with a single dimensionless parameter k that
linearly weights the two limit cases:

    k = sqrt( (P1 - Pv) / (P1 - P2) )         (Waxman Eq. 2.57)
    m_dyer = k * m_SPI + (1 - k) * m_HEM

The numerator (P1 - Pv) is the supercharge — how far the upstream
sits above its boiling point. The denominator (P1 - P2) is the
total pressure drop across the injector. Their ratio gives k a
clean physical reading:

    k = 0  when P1 = Pv (saturated upstream): Dyer = HEM, because a
           saturated fluid flashes aggressively and equilibrium is
           the right assumption.
    k = 1  when P2 = Pv (no cavitation at the exit): Dyer = SPI,
           because the fluid never sees subsaturated pressure and
           stays liquid all the way through.

The form here is the Waxman-corrected version of Dyer's original
(2007) formulation. The original used cavitation number directly
and gave less clean limiting behaviour; Eq. 2.57 is what the
validation script and main.py both use.

"""

import numpy as np
from CoolProp.CoolProp import PropsSI

from fluids import get_vapor_pressure
from models.spi import mass_flow_rate as spi_mass_flow_rate
from models.hem import mass_flow_rate as hem_mass_flow_rate


def non_equilibrium_parameter(P1_pa, P2_pa, Pv_pa):
    """
    Dyer non-equilibrium parameter k (Waxman Eq. 2.57).

        k = sqrt( (P1 - Pv) / (P1 - P2) )

    Vectorised across P2_pa: any of the inputs may be a scalar or
    array as long as they broadcast. The result is clipped to [0, 1]
    so that physically nonsensical inputs (negative supercharge,
    P2 ≥ P1) give a defined value rather than a NaN.

    """
    numerator   = np.clip(P1_pa - Pv_pa, 0.0, None)
    denominator = np.clip(P1_pa - P2_pa, 1e-6, None)  # avoid divide by zero
    return np.sqrt(np.clip(numerator / denominator, 0.0, 1.0))


def mass_flow_rate(injector, fluid, P1_pa, T1_k, P2_pa):
    """
    Dyer mass flow rate for a single operating point.

        m_dyer = k * m_SPI + (1 - k) * m_HEM

    Calls the SPI and HEM models internally
    and blends their results with the non-equilibrium parameter k.
    """
    Pv_pa = get_vapor_pressure(fluid, T1_k)
    k     = non_equilibrium_parameter(P1_pa, P2_pa, Pv_pa)

    rho_upstream = PropsSI('D', 'P', P1_pa * 1.0001, 'T', T1_k, fluid)
    delta_P      = P1_pa - P2_pa
    m_spi        = spi_mass_flow_rate(injector, rho_upstream, delta_P)
    m_hem        = hem_mass_flow_rate(injector, fluid, P1_pa, T1_k, P2_pa)

    return k * m_spi + (1.0 - k) * m_hem


def mass_flow_rate_sweep(injector, fluid, P1_pa, T1_k, P2_array_pa):
    """
    Dyer mass flow rate over an array of chamber pressures.
    Loops over P2_array_pa and calls mass_flow_rate for each point.
    """
    m_dot_array = np.zeros(len(P2_array_pa))

    for i, P2 in enumerate(P2_array_pa):
        if P2 >= P1_pa:
            m_dot_array[i] = 0.0  # no flow against an APG
        else:
            m_dot_array[i] = mass_flow_rate(injector, fluid, P1_pa, T1_k, P2)

        if (i + 1) % 20 == 0:
            print(f"  Dyer progress: {i + 1}/{len(P2_array_pa)} points done...")

    return m_dot_array