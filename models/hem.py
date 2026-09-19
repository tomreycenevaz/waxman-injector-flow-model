"""
Homogeneous Equilibrium Model (HEM) for two-phase injector flow.

Reference: Waxman Section 2.2.1.1.1, Figs. 2.19-2.26.


When chamber pressure P2 drops below the upstream vapour pressure Pv,
the propellant flashes inside the orifice and the flow becomes a
liquid/vapour mixture. HEM models that mixture under two assumptions:

  1. Homogeneous — liquid and vapour move at the same velocity.
  2. Equilibrium — liquid and vapour are always in thermodynamic
     equilibrium (no metastable liquid, no superheat).

From the adiabatic energy equation (Waxman Eq. 2.36) the mass flux at
any throat pressure is

    G = sqrt( 2 * (h0 - h) ) / v

where h0 is the upstream stagnation enthalpy, and h and v are the
enthalpy and specific volume evaluated at the throat. The flow is
assumed isentropic from inlet to throat, so the throat state is fixed
by (P_throat, s0) where s0 is the upstream entropy.


G(P_throat) is non-monotonic: as P_throat drops from P1, the enthalpy
drop grows (raising velocity) but the fluid flashes (raising specific
volume). The two effects fight, producing a maximum at some interior
pressure. That maximum is the choked mass flux. We find it numerically
with `scipy.optimize.minimize_scalar` on `-G`. If the optimiser's
answer falls at or below P2, the flow is not choked and we evaluate G
at P2 directly instead.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from CoolProp.CoolProp import PropsSI


def _get_upstream_properties(fluid, P1_pa, T1_k):
    """
    Return the stagnation enthalpy h0 and entropy s0 at the upstream state.

    These are the reference values held constant during the throat search:
    s0 because the flow is isentropic, h0 because it appears in the energy
    equation for G.

    """
    # Inset slightly above P1 to avoid the saturation boundary, where
    # CoolProp's (P, T) lookup is ambiguous between liquid and vapour.
    P1_safe = P1_pa * 1.0001
    h0 = PropsSI('H', 'P', P1_safe, 'T', T1_k, fluid)
    s0 = PropsSI('S', 'P', P1_safe, 'T', T1_k, fluid)
    return h0, s0


def _get_throat_properties(fluid, P_throat_pa, s0):
    """
    Return enthalpy and specific volume at the throat under isentropic flow.

    Throat state is fixed by (P_throat, s0). Returns (None, None) if
    CoolProp can't evaluate the lookup

    """
    try:
        h = PropsSI('H', 'P', P_throat_pa, 'S', s0, fluid)
        rho = PropsSI('D', 'P', P_throat_pa, 'S', s0, fluid)
        return h, 1.0 / rho
    except Exception:
        return None, None


def _mass_flux_at_pressure(fluid, P_throat_pa, h0, s0):
    """
    Evaluate G = sqrt(2 * (h0 - h)) / v at a given throat pressure.

    This is the objective the optimiser maximises over P_throat. Returns
    0.0 for unreachable states (CoolProp failure or non-positive
    enthalpy drop) so those points are simply rejected by the search.

    """
    h, v = _get_throat_properties(fluid, P_throat_pa, s0)
    if h is None:
        return 0.0

    enthalpy_drop = h0 - h
    if enthalpy_drop <= 0:
        return 0.0

    return np.sqrt(2.0 * enthalpy_drop) / v


def critical_mass_flux(fluid, P1_pa, T1_k, P2_pa):
    """
    Compute the critical (choked) HEM mass flux for one operating point.

    """
    h0, s0 = _get_upstream_properties(fluid, P1_pa, T1_k)

    # minimize_scalar finds minima, so negate G to find its maximum.
    def neg_mass_flux(P_throat):
        return -_mass_flux_at_pressure(fluid, P_throat, h0, s0)

    result = minimize_scalar(
        neg_mass_flux,
        bounds=(P2_pa * 1.001, P1_pa * 0.999),  # inset to avoid the bounds
        method='bounded',
        options={'xatol': P1_pa * 1e-4},        # converge to 0.01% of P1
    )

    G_crit = _mass_flux_at_pressure(fluid, result.x, h0, s0)
    G_at_P2 = _mass_flux_at_pressure(fluid, P2_pa, h0, s0)

    # If the unchoked value at P2 is larger, the flow isn't actually choked.
    return max(G_crit, G_at_P2)


def mass_flow_rate(injector, fluid, P1_pa, T1_k, P2_pa):
    """
    HEM mass flow rate through the injector for one operating point.

        m_dot = Cd * A2 * G_crit

    """
    G_crit = critical_mass_flux(fluid, P1_pa, T1_k, P2_pa)
    return injector.Cd * injector.A2 * G_crit


def mass_flow_rate_sweep(injector, fluid, P1_pa, T1_k, P2_array_pa):
    """
    HEM mass flow rate over an array of chamber pressures.

    Loops because critical_mass_flux runs an optimiser per point — this
    can't be vectorised. Prints progress
    every 20 points

    """
    m_dot_array = np.zeros(len(P2_array_pa))

    for i, P2 in enumerate(P2_array_pa):
        if P2 >= P1_pa:
            m_dot_array[i] = 0.0  # no flow against an APG
        else:
            m_dot_array[i] = mass_flow_rate(injector, fluid, P1_pa, T1_k, P2)

        if (i + 1) % 20 == 0:
            print(f"  HEM progress: {i + 1}/{len(P2_array_pa)} points done...")

    return m_dot_array