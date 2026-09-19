"""
Validation of HEM, Dyer, and Burnell against Waxman's cold-flow data.

Reference: Waxman Chapter 3, Figs. 3.17, 3.21, 3.22, Table 3.1.

Reproduces Waxman's validation methodology: predict the choked mass
flow rate for each of his four characterised injectors, compare
against his cold-flow CO₂ measurements, and quantify the agreement
with RMS error in g/s.

Outputs five figures:

    1-4  One panel per injector, showing HEM / Dyer / Burnell
         predictions overlaid on Waxman's experimental points,
         with each model's RMS error in the legend.
    5    Bar chart summarising RMS error across all four injectors
         and three models.

"""

import numpy as np
import matplotlib.pyplot as plt

from fluids import psia_to_pa, celsius_to_kelvin, get_vapor_pressure
from injector import Injector
from models.hem     import mass_flow_rate as hem_mass_flow_rate
from models.burnell import mass_flow_rate as burnell_mass_flow_rate
from models.dyer    import mass_flow_rate as dyer_mass_flow_rate


CO2 = 'CarbonDioxide'

# =============================================================================
# Injector geometries (Waxman Table 3.1)
# Cd values from Fig. 3.21's single-phase plateau, except Injector 6
# =============================================================================
injectors = {
    'Injector 2 (Square Edge, L/D=12.3)': Injector(diameter_mm=1.50, Cd=0.68,
                                                   L_over_D=12.3),
    'Injector 3 (Rounded, L/D=12.3)':     Injector(diameter_mm=1.50, Cd=0.80,
                                                   L_over_D=12.3),
    'Injector 4 (Chamfered, L/D=12.3)':   Injector(diameter_mm=1.50, Cd=0.75,
                                                   L_over_D=12.3),
    'Injector 6 (Square Edge, L/D=2.1)':  Injector(diameter_mm=1.50, Cd=0.65,
                                                   L_over_D=2.1),
}

# =============================================================================
# Experimental data — digitised by eye from Waxman Figs. 3.17 and 3.22
# =============================================================================
exp_data = {
    'Injector 2 (Square Edge, L/D=12.3)': {
        'P_super_psi': [100,   130,   150,   200,   250,   300],
        'm_crit_kgs':  [0.053, 0.060, 0.060, 0.063, 0.076, 0.081],
    },
    'Injector 3 (Rounded, L/D=12.3)': {
        'P_super_psi': [60,    100,   150,   200,   250,   300,   350],
        'm_crit_kgs':  [0.055, 0.061, 0.071, 0.080, 0.086, 0.091, 0.095],
    },
    'Injector 4 (Chamfered, L/D=12.3)': {
        'P_super_psi': [60,    100,   150,   200,   250,   300,   350],
        'm_crit_kgs':  [0.060, 0.063, 0.073, 0.081, 0.088, 0.093, 0.100],
    },
    'Injector 6 (Square Edge, L/D=2.1)': {
        'P_super_psi': [55,    100,   200,   250,   300,   350,   400],
        'm_crit_kgs':  [0.072, 0.074, 0.077, 0.083, 0.083, 0.089, 0.093],
    },
}

T_test_C          = 20.0
T_test_K          = celsius_to_kelvin(T_test_C)
P_super_model_psi = np.linspace(10, 410, 40)


# =============================================================================
# Per-model critical-flow sweeps
#
# Each sweep walks over P_super values, computes P₁ = P_super + Pv, and
# delegates the actual mass-flow calculation to the corresponding model
# in the ``models/`` package. P₂ is fixed at 0.1·Pv to guarantee choked
# flow at every point.
# =============================================================================

def _critical_sweep(injector, fluid, T_k, P_super_psi_array, model_fn):
    """
    Generic supercharge sweep that delegates per-point computation to model_fn.
        model_fn is one of the mass_flow_rate functions from the models package.
        This helper just computes P₁ from P_super and Pv, sets P₂ to 0.1·Pv,
        and loops over the supercharge array to build up the model curve.
        Prints progress every 20 points.
    """
    Pv_pa        = get_vapor_pressure(fluid, T_k)
    P2_pa        = Pv_pa * 0.1   # well below Pv to guarantee choked flow
    m_crit_array = np.zeros(len(P_super_psi_array))

    for i, P_super_psi in enumerate(P_super_psi_array):
        P1_pa           = Pv_pa + psia_to_pa(P_super_psi)
        m_crit_array[i] = model_fn(injector, fluid, P1_pa, T_k, P2_pa)

    return m_crit_array


def hem_critical_sweep(injector, fluid, T_k, P_super_psi_array):
    """HEM critical mass flow rate over a supercharge sweep."""
    return _critical_sweep(injector, fluid, T_k, P_super_psi_array,
                           hem_mass_flow_rate)


def burnell_critical_sweep(injector, fluid, T_k, P_super_psi_array):
    """Burnell critical mass flow rate over a supercharge sweep."""
    return _critical_sweep(injector, fluid, T_k, P_super_psi_array,
                           burnell_mass_flow_rate)


def dyer_critical_sweep(injector, fluid, T_k, P_super_psi_array):
    """
    Dyer critical mass flow rate over a supercharge sweep.

    Uses the corrected Dyer parameter k = √((P₁ - Pv) / (P₁ - P₂))
    from Waxman Eq. 2.57, as implemented in ``models.dyer``. With
    supercharge P₁ > Pv, k > 0 and the Dyer prediction shifts toward
    SPI; at saturation k = 0 and Dyer collapses onto HEM.
    """
    return _critical_sweep(injector, fluid, T_k, P_super_psi_array,
                           dyer_mass_flow_rate)


# =============================================================================
# Plotting and error helpers
# =============================================================================

def compute_rms(model_array, model_x, exp_x, exp_y_kgs):
    """
    Interpolate a model curve at experimental x-points and return RMS error.

    """
    model_at_data = np.interp(exp_x, model_x, model_array * 1000.0)
    exp_y_gs      = np.array(exp_y_kgs) * 1000.0
    return np.sqrt(np.mean((model_at_data - exp_y_gs) ** 2))


def plot_injector_validation(ax, inj_name, injector,
                             m_hem, m_dyer, m_burnell,
                             P_super_model_psi, rms_dict):
    """
    Plot HEM / Dyer / Burnell predictions and Waxman's data on one axis.

    """
    exp   = exp_data[inj_name]
    exp_x = np.array(exp['P_super_psi'])
    exp_y = np.array(exp['m_crit_kgs'])

    ax.plot(P_super_model_psi, m_hem * 1000.0, color='steelblue',
            linewidth=2.5, linestyle=':',
            label=f"HEM      (RMS={rms_dict['HEM']:.1f} g/s)")
    ax.plot(P_super_model_psi, m_burnell * 1000.0, color='darkorange',
            linewidth=2.5, linestyle='-.',
            label=f"Burnell  (RMS={rms_dict['Burnell']:.1f} g/s)")
    ax.plot(P_super_model_psi, m_dyer * 1000.0, color='crimson',
            linewidth=2.5, linestyle='-',
            label=f"Dyer     (RMS={rms_dict['Dyer']:.1f} g/s)")

    ax.scatter(exp_x, exp_y * 1000.0,
               color='black', s=80, zorder=5, marker='o',
               label="Waxman experimental data")

    ax.set_xlabel("Supercharge Pressure  P_super  (psi)", fontsize=12)
    ax.set_ylabel("Critical Mass Flow Rate  ṁ_crit  (g/s)", fontsize=12)
    ax.set_title(f"{inj_name}\nCO₂ at T={T_test_C}°C  |  "
                 f"Cd={injector.Cd},  D=1.50mm,  L/D={injector.L_over_D}",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 420)
    ax.set_ylim(0, None)


# =============================================================================
# Main execution
# =============================================================================

def main():
    """Run all four sweeps, draw all five figures, and print the summary."""
    print("=" * 60)
    print("Running all model sweeps for 4 injectors ...")
    print("=" * 60)

    model_results = {}
    all_rms       = {}

    for inj_name, inj in injectors.items():
        print(f"\n--- {inj_name} ---")

        print("  HEM ...")
        m_hem = hem_critical_sweep(inj, CO2, T_test_K, P_super_model_psi)

        print("  Dyer ...")
        m_dyer = dyer_critical_sweep(inj, CO2, T_test_K, P_super_model_psi)

        print("  Burnell ...")
        m_burnell = burnell_critical_sweep(inj, CO2, T_test_K, P_super_model_psi)

        exp   = exp_data[inj_name]
        exp_x = np.array(exp['P_super_psi'])
        exp_y = np.array(exp['m_crit_kgs'])

        rms = {
            'HEM':     compute_rms(m_hem,     P_super_model_psi, exp_x, exp_y),
            'Dyer':    compute_rms(m_dyer,    P_super_model_psi, exp_x, exp_y),
            'Burnell': compute_rms(m_burnell, P_super_model_psi, exp_x, exp_y),
        }

        model_results[inj_name] = {
            'm_hem':     m_hem,
            'm_dyer':    m_dyer,
            'm_burnell': m_burnell,
        }
        all_rms[inj_name] = rms

    # -------------------------------------------------------------------------
    # Figures 1–4: one panel per injector
    # -------------------------------------------------------------------------
    inj_names = list(injectors.keys())

    for i, inj_name in enumerate(inj_names):
        fig, ax = plt.subplots(figsize=(10, 7))
        r       = model_results[inj_name]

        plot_injector_validation(
            ax, inj_name, injectors[inj_name],
            r['m_hem'], r['m_dyer'], r['m_burnell'],
            P_super_model_psi, all_rms[inj_name],
        )

        fig.suptitle(f"Figure {i + 1}: Model Validation — {inj_name}",
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()

    # -------------------------------------------------------------------------
    # Figure 5: RMS bar chart summary across all injectors and models
    # -------------------------------------------------------------------------
    fig5, ax5 = plt.subplots(figsize=(12, 6))

    model_names = ['HEM', 'Dyer', 'Burnell']
    bar_colors  = {'HEM': 'steelblue', 'Dyer': 'crimson', 'Burnell': 'darkorange'}
    x           = np.arange(len(inj_names))
    width       = 0.25

    for j, model in enumerate(model_names):
        rms_values = [all_rms[inj][model] for inj in inj_names]
        bars       = ax5.bar(x + j * width, rms_values, width,
                             label=model, color=bar_colors[model], alpha=0.85)

        for bar, val in zip(bars, rms_values):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax5.set_xlabel("Injector", fontsize=13)
    ax5.set_ylabel("RMS Error  (g/s)", fontsize=13)
    ax5.set_title("Figure 5: Model Accuracy Summary\n"
                  "RMS Error vs Waxman Experimental Data — Lower is Better",
                  fontsize=12)
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([name.replace(' (', '\n(') for name in inj_names],
                        fontsize=9)
    ax5.legend(fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_ylim(0, None)
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Console summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY — RMS Error (g/s) vs Waxman Data")
    print("=" * 65)
    print(f"{'Injector':<40} {'HEM':>8} {'Dyer':>8} {'Burnell':>8}  Best")
    print("-" * 65)
    for inj_name in inj_names:
        rms  = all_rms[inj_name]
        best = min(rms, key=rms.get)
        print(f"{inj_name:<40} {rms['HEM']:>8.1f} "
              f"{rms['Dyer']:>8.1f} {rms['Burnell']:>8.1f}  {best}")
    print("=" * 65)

    print("\nShowing plots ...")
    plt.show()


if __name__ == "__main__":
    main()