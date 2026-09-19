"""
Injector geometry.

Reference: Waxman Section 2.1, Fig. 2.1.

Defines a simple straight-hole orifice injector parameterised by
orifice diameter D2, discharge coefficient Cd, optional upstream
manifold area A1, and optional length-to-diameter ratio L/D. The
class stores these as SI values (metres, m^2) and computes the
orifice area A2 = π(D2/2)^2 once at construction time so the model
files can read it cheaply.

The discharge coefficient is treated as a fixed input here. In
reality Cd depends on inlet shape, L/D, Reynolds number, and
cavitation state; this code does not attempt to predict it. Use
``models.spi.discharge_coefficient_from_measurement`` to extract
Cd from cold-flow data.
"""

import numpy as np
from fluids import mm_to_m


class Injector:
    """
    Single straight-hole orifice injector.

    Attributes:
        diameter_mm (float): Orifice diameter in mm, as supplied.
        Cd (float):          Discharge coefficient.
        L_over_D (float or None): Length-to-diameter ratio, or None.
        D2 (float):          Orifice diameter in m.
        A2 (float):          Orifice area in m².
        A1 (float or None):  Upstream manifold area in m², or None
                             if the A1 >> A2 simplification is in use.
    """

    def __init__(self, diameter_mm, Cd, upstream_area_m2=None, L_over_D=None):
        """
        Construct an injector from its geometry.

        Args:
            diameter_mm (float): Orifice diameter D₂ in mm. Waxman's
                standard example uses 1.5 mm.
            Cd (float): Single-phase liquid discharge coefficient. Waxman
                uses 0.75 in Section 2.1.1.1; values from 0.6 to 0.9
                are typical depending on inlet geometry.
            upstream_area_m2 (float, optional): Manifold cross-sectional
                area A₁ in m². If omitted, the model assumes A₁ >> A₂,
                which is Waxman's Eq. 2.17 simplification of the full
                Eq. 2.16. Most hybrid feed systems satisfy this.
            L_over_D (float, optional): Orifice length-to-diameter ratio.
                Stored for reference only — Cd is not derived from it.
        """
        self.diameter_mm = diameter_mm
        self.Cd          = Cd
        self.L_over_D    = L_over_D

        self.D2 = mm_to_m(diameter_mm)
        self.A2 = np.pi * (self.D2 / 2.0) ** 2

        
        self.A1 = upstream_area_m2

    def area_ratio_squared(self):
        """
        Return (A2/A1)² for use in the SPI denominator (Waxman Eq. 2.16):

            m_dot = Cd * A2 * sqrt( 2*ρ*ΔP / (1 - (A2/A1)²) )

        Returns 0.0 when A1 is unset, which collapses the equation to
        the simplified Eq. 2.17 form.

        Returns:
            float: (A2/A1)², or 0.0 under the large-manifold assumption.
        """
        if self.A1 is None:
            return 0.0
        return (self.A2 / self.A1) ** 2

    def __repr__(self):
        """Compact one-line representation for prints and logs."""
        return (f"Injector(D2={self.diameter_mm} mm, "
                f"Cd={self.Cd}, "
                f"A2={self.A2 * 1e6:.4f} mm^2, "
                f"L/D={self.L_over_D})")