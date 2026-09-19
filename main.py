"""
Reproduce Waxman's saturated-flow figures for N₂O and explore the
non-equilibrium behaviour of the Dyer model under varying supercharge.

Reference: Waxman (2014) Chapter 2.

Produces six figures:

    1, 2  SPI mass flow rate vs ΔP and vs P₂ at six temperatures.
          Reproduces Waxman Figs. 2.2 and 2.3.

    3, 4  HEM vs SPI for saturated N₂O across three temperatures
          (Fig. 3 reproduces Waxman Fig. 2.20), and an all-models
          comparison at T = 0°C showing that Dyer collapses onto
          HEM under saturated conditions because k = 0.

    5, 6  Dyer mass flow and the non-equilibrium parameter k as
          functions of ΔP for several supercharge values at T = 0°C.
          This is an additional cut not in the dissertation: Waxman
          fixes supercharge and varies temperature; here we fix
          temperature and vary supercharge to show directly how the
          Dyer prediction migrates from HEM toward SPI as P_super
          grows.
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from fluids import N2O, get_liquid_density_at_saturation, get_vapor_pressure
from fluids import psia_to_pa, pa_to_psia, celsius_to_kelvin
from injector import Injector
from models.spi import mass_flow_rate as spi_mass_flow_rate
from models.hem import mass_flow_rate_sweep as hem_mass_flow_rate_sweep
from models.dyer import mass_flow_rate_sweep as dyer_mass_flow_rate_sweep
from models.dyer import non_equilibrium_parameter
from models.burnell import mass_flow_rate_sweep as burnell_mass_flow_rate_sweep

injector = Injector(diameter_mm=1.5, Cd=0.75)
print(f"Injector: {injector}\n")

# =============================================================================
# FIGURES 1 & 2 — SPI baseline
# Mass flow rate vs ΔP and vs P₂ for saturated N₂O across six temperatures.
# =============================================================================
temperatures_C = [-20, -10, 0, 10, 20, 30]
delta_P_psia   = np.linspace(1, 600, 300)
delta_P_pa     = psia_to_pa(delta_P_psia)

fig1, ax1 = plt.subplots(figsize=(9, 6))
fig2, ax2 = plt.subplots(figsize=(9, 6))

for T_C in temperatures_C:
    T_K     = celsius_to_kelvin(T_C)
    Pv_pa   = get_vapor_pressure(N2O, T_K)
    Pv_psia = pa_to_psia(Pv_pa)
    rho     = get_liquid_density_at_saturation(N2O, T_K)

    m_dot    = spi_mass_flow_rate(injector, rho, delta_P_pa)
    m_dot_gs = m_dot * 1000.0
    P2_psia  = Pv_psia - delta_P_psia
    label    = f"T = {T_C}°C  (Pv = {Pv_psia:.0f} psia)"

    ax1.plot(delta_P_psia, m_dot_gs, linewidth=2, label=label)
    ax2.plot(P2_psia, m_dot_gs, linewidth=2, label=label)

ax1.set_xlabel("Injector Pressure Drop  ΔP  (psia)", fontsize=13)
ax1.set_ylabel("Mass Flow Rate  ṁ  (g/s)", fontsize=13)
ax1.set_title("Figure 2.2 (reproduced): SPI Mass Flow Rate vs. ΔP\n"
              "Saturated N₂O  |  D₂ = 1.5 mm, Cd = 0.75", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 600)
ax1.set_ylim(0, None)

ax2.set_xlabel("Chamber (Back) Pressure  P₂  (psia)", fontsize=13)
ax2.set_ylabel("Mass Flow Rate  ṁ  (g/s)", fontsize=13)
ax2.set_title("Figure 2.3 (reproduced): SPI Mass Flow Rate vs. P₂\n"
              "Saturated N₂O  |  D₂ = 1.5 mm, Cd = 0.75", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()

# =============================================================================
# FIGURES 3 & 4 — Saturated sweep
# P1 = Pv, so k = 0 throughout and Dyer collapses onto HEM. We compute SPI,
# HEM, and Burnell explicitly; Dyer is plotted as the HEM result with the
# correct label since they are mathematically equal here.
# =============================================================================
print("=" * 60)
print("Starting saturated sweep (HEM and Burnell) ...")
print("=" * 60)

sweep_temperatures_C = [-10, 0, 20]
n_points             = 60
colors               = {-10: 'steelblue', 0: 'forestgreen', 20: 'crimson'}
saturated_results    = {}

for T_C in sweep_temperatures_C:
    T_K     = celsius_to_kelvin(T_C)
    Pv_pa   = get_vapor_pressure(N2O, T_K)
    Pv_psia = pa_to_psia(Pv_pa)
    P1_pa   = Pv_pa  # saturated upstream

    P2_psia_array = np.linspace(10, Pv_psia - 10, n_points)
    P2_pa_array   = psia_to_pa(P2_psia_array)
    dP_psia_array = Pv_psia - P2_psia_array

    rho   = get_liquid_density_at_saturation(N2O, T_K)
    dP_pa = psia_to_pa(dP_psia_array)
    m_spi = spi_mass_flow_rate(injector, rho, dP_pa) * 1000.0

    print(f"\nRunning HEM (saturated) for T = {T_C}°C ...")
    m_hem = hem_mass_flow_rate_sweep(
                injector, N2O, P1_pa, T_K, P2_pa_array) * 1000.0

    print(f"Running Burnell (saturated) for T = {T_C}°C ...")
    m_burnell = burnell_mass_flow_rate_sweep(
                    injector, N2O, P1_pa, T_K, P2_pa_array) * 1000.0

    saturated_results[T_C] = {
        'Pv_psia':   Pv_psia,
        'dP_psia':   dP_psia_array,
        'm_spi':     m_spi,
        'm_hem':     m_hem,
        'm_burnell': m_burnell,
    }

# -----------------------------------------------------------------------------
# Figure 3 — HEM vs SPI, saturated, three temperatures
# -----------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(10, 7))

for T_C in sweep_temperatures_C:
    r     = saturated_results[T_C]
    color = colors[T_C]
    ax3.plot(r['dP_psia'], r['m_spi'], color=color, linewidth=2,
             linestyle='--', label=f"SPI  T={T_C}°C (Pv={r['Pv_psia']:.0f} psia)")
    ax3.plot(r['dP_psia'], r['m_hem'], color=color, linewidth=2.5,
             linestyle='-',  label=f"HEM  T={T_C}°C")

ax3.set_xlabel("Injector Pressure Drop  ΔP  (psia)", fontsize=13)
ax3.set_ylabel("Mass Flow Rate  ṁ  (g/s)", fontsize=13)
ax3.set_title("Figure 2.20 (reproduced): HEM vs. SPI\n"
              "Saturated N₂O  |  Solid=HEM, Dashed=SPI  |  D₂=1.5mm, Cd=0.75",
              fontsize=12)
ax3.legend(fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, None)
ax3.set_ylim(0, None)
plt.tight_layout()

# -----------------------------------------------------------------------------
# Figure 4 — All four models at T = 0°C, saturated
# Dyer is plotted using the HEM array because k = 0 makes them mathematically
# identical here; running Dyer separately would just call HEM internally.
# -----------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(10, 7))

T_C = 0
r   = saturated_results[T_C]

ax4.plot(r['dP_psia'], r['m_spi'],     color='gray',       linewidth=2,
         linestyle='--', label="SPI      (incompressible baseline)")
ax4.plot(r['dP_psia'], r['m_hem'],     color='steelblue',  linewidth=2.5,
         linestyle=':',  label="HEM      (equilibrium, choked)")
ax4.plot(r['dP_psia'], r['m_burnell'], color='darkorange', linewidth=2.5,
         linestyle='-.', label="Burnell  (frozen, exit flash)")
ax4.plot(r['dP_psia'], r['m_hem'],     color='crimson',    linewidth=3,
         linestyle='-',  label="Dyer     (= HEM for saturated, k=0)")

ax4.set_xlabel("Injector Pressure Drop  ΔP  (psia)", fontsize=13)
ax4.set_ylabel("Mass Flow Rate  ṁ  (g/s)", fontsize=13)
ax4.set_title(f"All Models — Saturated N₂O at T={T_C}°C  (Pv={r['Pv_psia']:.0f} psia)\n"
              "D₂=1.5mm, Cd=0.75  |  Dyer collapses to HEM when P1=Pv",
              fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, None)
ax4.set_ylim(0, None)
plt.tight_layout()

# =============================================================================
# FIGURES 5 & 6 — Supercharge-value sweep (additional analysis cut)
# Fix T at 0°C, sweep P_super through {0, 50, 100, 200, 400} psia. This is
# the orthogonal cut to Waxman's figures, which fix supercharge and vary T.
# =============================================================================
print("\n" + "=" * 60)
print("Starting supercharge-value sweep at T = 0°C ...")
print("=" * 60)

T_C_sweep     = 0
T_K_sweep     = celsius_to_kelvin(T_C_sweep)
Pv_pa_sweep   = get_vapor_pressure(N2O, T_K_sweep)
Pv_psia_sweep = pa_to_psia(Pv_pa_sweep)

P_super_sweep_psia = [0, 50, 100, 200, 400]   # 0 → saturated (Dyer = HEM)
super_colors       = plt.cm.viridis(np.linspace(0.15, 0.85,
                                                len(P_super_sweep_psia)))

super_sweep_results = {}

for P_super in P_super_sweep_psia:
    P_super_pa_i = psia_to_pa(P_super)
    P1_pa_i      = Pv_pa_sweep + P_super_pa_i
    P1_psia_i    = Pv_psia_sweep + P_super

    P2_psia_array = np.linspace(10, P1_psia_i - 10, n_points)
    P2_pa_array   = psia_to_pa(P2_psia_array)
    dP_psia_array = P1_psia_i - P2_psia_array

    # SPI uses liquid density at P1: saturated for P_super = 0, otherwise
    # subcooled liquid at the elevated pressure.
    if P_super > 0:
        rho = PropsSI('D', 'P', P1_pa_i * 1.0001, 'T', T_K_sweep, N2O)
    else:
        rho = get_liquid_density_at_saturation(N2O, T_K_sweep)
    dP_pa = psia_to_pa(dP_psia_array)
    m_spi = spi_mass_flow_rate(injector, rho, dP_pa) * 1000.0

    print(f"\n--- P_super = {P_super} psia (P1 = {P1_psia_i:.0f} psia) ---")
    print("Running Dyer ...")
    m_dyer = dyer_mass_flow_rate_sweep(
                injector, N2O, P1_pa_i, T_K_sweep, P2_pa_array) * 1000.0

    print("Running HEM  ...")
    m_hem = hem_mass_flow_rate_sweep(
                injector, N2O, P1_pa_i, T_K_sweep, P2_pa_array) * 1000.0

    k_array = non_equilibrium_parameter(P1_pa_i, P2_pa_array, Pv_pa_sweep)

    super_sweep_results[P_super] = {
        'P1_psia': P1_psia_i,
        'dP_psia': dP_psia_array,
        'm_spi':   m_spi,
        'm_hem':   m_hem,
        'm_dyer':  m_dyer,
        'k':       k_array,
    }

# -----------------------------------------------------------------------------
# Figure 5 — Dyer mass flow at varying supercharge, with HEM and SPI bounds
# -----------------------------------------------------------------------------
fig5, ax5 = plt.subplots(figsize=(10, 7))

for P_super, color in zip(P_super_sweep_psia, super_colors):
    r = super_sweep_results[P_super]
    ax5.plot(r['dP_psia'], r['m_dyer'], color=color, linewidth=2.5,
             label=f"Dyer  P_super = {P_super:>3} psia  (P1 = {r['P1_psia']:.0f} psia)")

# Reference bounds: HEM at P_super = 0, SPI at the highest P_super.
r_lo = super_sweep_results[P_super_sweep_psia[0]]
r_hi = super_sweep_results[P_super_sweep_psia[-1]]
ax5.plot(r_lo['dP_psia'], r_lo['m_hem'], color='black', linewidth=1.2,
         linestyle=':', alpha=0.6, label="HEM bound (P_super = 0)")
ax5.plot(r_hi['dP_psia'], r_hi['m_spi'], color='black', linewidth=1.2,
         linestyle='--', alpha=0.6,
         label=f"SPI bound (P_super = {P_super_sweep_psia[-1]} psia)")

ax5.set_xlabel("Injector Pressure Drop  ΔP  (psia)", fontsize=13)
ax5.set_ylabel("Mass Flow Rate  ṁ  (g/s)", fontsize=13)
ax5.set_title(f"Dyer Mass Flow vs ΔP at Varying Supercharge — N₂O at T={T_C_sweep}°C\n"
              "As P_super ↑, k ↑ and Dyer migrates from HEM toward SPI",
              fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(0, None)
ax5.set_ylim(0, None)
plt.tight_layout()

# -----------------------------------------------------------------------------
# Figure 6 — k parameter vs ΔP at varying supercharge
# -----------------------------------------------------------------------------
fig6, ax6 = plt.subplots(figsize=(9, 5))

for P_super, color in zip(P_super_sweep_psia, super_colors):
    r = super_sweep_results[P_super]
    ax6.plot(r['dP_psia'], r['k'], color=color, linewidth=2,
             label=f"P_super = {P_super:>3} psia")
ax6.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='k=1 → SPI')
ax6.axhline(y=0.0, color='gray', linestyle=':',  linewidth=1, label='k=0 → HEM')

ax6.set_xlabel("Injector Pressure Drop  ΔP  (psia)", fontsize=13)
ax6.set_ylabel("Non-equilibrium parameter  k", fontsize=13)
ax6.set_title(f"Dyer k Parameter vs ΔP at Varying Supercharge — T={T_C_sweep}°C",
              fontsize=12)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, None)
ax6.set_ylim(-0.05, 1.1)
plt.tight_layout()

print("\nShowing plots ...")
plt.show()