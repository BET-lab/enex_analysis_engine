import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Setup paths to import enex_analysis and dartwork_mpl
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.dirname(script_dir)
src_dir = os.path.abspath(os.path.join(docs_dir, "..", "src"))
sys.path.insert(0, src_dir)

import dartwork_mpl as dm
from enex_analysis.refrigerant import calc_ref_state
from enex_analysis.visualization import plot_th_diagram, plot_ph_diagram, plot_ts_diagram
import enex_analysis.calc_util as cu

plt.switch_backend("Agg")
dm.style.use('scientific')

def main():
    refrigerants = ["R410A", "R134a", "R32", "R290"]
    T_sources_C = np.arange(-15, 20, 5)  # -15 to 15
    T_sinks_C = np.arange(25, 60, 5)   # 25 to 55
    
    # Create output directory
    out_dir = os.path.join(docs_dir, "source", "_static", "interactive_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    eta_cmp_isen = 0.7
    dT_superheat = 3.0
    dT_subcool = 3.0
    
    total_iters = len(refrigerants) * len(T_sources_C) * len(T_sinks_C)
    
    with tqdm(total=total_iters, desc="Generating Cycle Plots") as pbar:
        for ref in refrigerants:
            for T_source in T_sources_C:
                for T_sink in T_sinks_C:
                    Te = T_source - 5
                    Tc = T_sink + 5
                    Te_K = cu.C2K(Te)
                    Tc_K = cu.C2K(Tc)
                    
                    # Ensure Tc > Te
                    if Tc <= Te:
                        pbar.update(1)
                        continue
                        
                    result = calc_ref_state(
                        T_evap_K=Te_K,
                        T_cond_K=Tc_K,
                        refrigerant=ref,
                        eta_cmp_isen=eta_cmp_isen,
                        mode="heating",
                        dT_superheat=dT_superheat,
                        dT_subcool=dT_subcool,
                        is_active=True
                    )
                    
                    # We add 'is_on' to result so that lines are drawn solid instead of grayed out
                    result["is_on"] = True
                    
                    FIG_W = dm.cm2in(7.0)
                    FIG_H = dm.cm2in(6.0)

                    # Generate T-h
                    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    plot_th_diagram(
                        ax, result, ref, 
                        T_evap_bound={"val": float(T_source), "label": "Source"}, 
                        T_cond_bound={"val": float(T_sink), "label": "Sink"}
                    )
                    fig.savefig(os.path.join(out_dir, f"th_{ref}_{T_source}_{T_sink}.png"), dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    
                    # Generate P-h
                    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    plot_ph_diagram(ax, result, ref)
                    fig.savefig(os.path.join(out_dir, f"ph_{ref}_{T_source}_{T_sink}.png"), dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    
                    # Generate T-s
                    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
                    plot_ts_diagram(
                        ax, result, ref, 
                        T_evap_bound={"val": float(T_source), "label": "Source"}, 
                        T_cond_bound={"val": float(T_sink), "label": "Sink"}
                    )
                    fig.savefig(os.path.join(out_dir, f"ts_{ref}_{T_source}_{T_sink}.png"), dpi=600, bbox_inches="tight")
                    plt.close(fig)
                    
                    pbar.update(1)

if __name__ == "__main__":
    main()
