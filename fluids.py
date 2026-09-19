"""
Fluid property lookups for the Waxman injector models.

Wraps CoolProp's PropsSI for the two fluids this project uses:
N2O (the live propellant exercised in main.py) and CO2 (the
surrogate Waxman used for the cold-flow validation in his Chapter 3
experiments). Waxman's dissertation uses NIST REFPROP for property
lookups; CoolProp is the open-source equivalent and uses the same
underlying equations of state, so results match REFPROP to within
the EOS uncertainty.

This module also exposes the unit-conversion helpers (psia ↔ Pa,
°C → K, mm → m) so the rest of the codebase can speak in Waxman's
preferred units at the boundary while CoolProp runs internally in SI.
"""

from CoolProp.CoolProp import PropsSI


# =============================================================================
# Unit conversions
# =============================================================================
# Waxman reports in psia and °C. CoolProp speaks SI. These four helpers do
# the boundary translation so callers can write expressions in Waxman's
# units and let the conversion happen at function entry / exit.

def psia_to_pa(psia):
    """Convert pressure from psia to Pa. 1 psia = 6894.76 Pa."""
    return psia * 6894.76


def pa_to_psia(pa):
    """Convert pressure from Pa to psia."""
    return pa / 6894.76


def celsius_to_kelvin(temp_c):
    """Convert temperature from Celsius to Kelvin."""
    return temp_c + 273.15


def mm_to_m(mm):
    """Convert millimetres to metres."""
    return mm / 1000.0


# =============================================================================
# Fluid property lookups
# =============================================================================
# Thin wrappers around CoolProp's PropsSI. The general PropsSI signature is
#
#     PropsSI(output, input1_name, input1_value, input2_name, input2_value, fluid)
#
# Common property keys used here:
#     'D'  density (kg/m³)
#     'P'  pressure (Pa)
#     'T'  temperature (K)
#     'H'  specific enthalpy (J/kg)
#     'S'  specific entropy (J/kg/K)
#     'Q'  vapour quality (0 = saturated liquid, 1 = saturated vapour)

def get_density(fluid, pressure_pa, temperature_k):
    """
    Density of a fluid at given pressure and temperature.

    """
    return PropsSI('D', 'P', pressure_pa, 'T', temperature_k, fluid)


def get_vapor_pressure(fluid, temperature_k):
    """
    Saturation (vapour) pressure of a fluid at a given temperature.

    This is the pressure at which the fluid begins to boil or cavitate;
    in Waxman's notation it is Pv (Eq. 2.18). Computed by querying
    CoolProp on the saturated-liquid line (Q = 0).

    """
    return PropsSI('P', 'T', temperature_k, 'Q', 0, fluid)


def get_liquid_density_at_saturation(fluid, temperature_k):
    """
    Density of the liquid phase at saturation conditions (Q = 0).

    Use this as the upstream density for the SPI and Burnell models
    when the upstream is saturated. For supercharged conditions
    (P1 > Pv) you want the subcooled liquid density at (P2, T2) instead,
    which goes through ``get_density`` or PropsSI directly.

    """
    return PropsSI('D', 'T', temperature_k, 'Q', 0, fluid)


# =============================================================================
# Fluid name constants
# =============================================================================
# CoolProp expects exact string names. Defining them once here avoids
# scattering the exact spellings throughout the rest of the codebase.

N2O = 'NitrousOxide'
CO2 = 'CarbonDioxide'