import os
import sys
import numpy as np

# Setup paths to import enex_analysis
script_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.dirname(script_dir)
src_dir = os.path.abspath(os.path.join(docs_dir, "..", "src"))
sys.path.insert(0, src_dir)

from enex_analysis.heat_pumps.refrigerant import calc_ref_state
import enex_analysis.heat_pumps.calc_util as cu

def find_limits():
    refrigerants = ["R410A", "R134a", "R32", "R290"]
    T_evaps_C = np.arange(-20, 15, 5)
    T_conds_C = np.arange(30, 65, 5)
    
    eta_cmp_isen = 0.7
    dT_superheat = 3.0
    dT_subcool = 3.0
    
    h_min, h_max = float('inf'), float('-inf')
    P_min, P_max = float('inf'), float('-inf')
    s_min, s_max = float('inf'), float('-inf')
    T_min, T_max = float('inf'), float('-inf')
    
    for ref in refrigerants:
        for Te in T_evaps_C:
            for Tc in T_conds_C:
                if Tc <= Te:
                    continue
                
                Te_K = cu.C2K(Te)
                Tc_K = cu.C2K(Tc)
                
                res = calc_ref_state(
                    T_evap_K=Te_K,
                    T_cond_K=Tc_K,
                    refrigerant=ref,
                    eta_cmp_isen=eta_cmp_isen,
                    mode="heating",
                    dT_superheat=dT_superheat,
                    dT_subcool=dT_subcool,
                    is_active=True
                )
                
                # Extract values from res dict. The values might be lists or arrays or scalars.
                # Actually, calc_ref_state returns scalars for each key like 'T_ref_cmp_out [K]'
                keys = list(res.keys())
                h_keys = [k for k in keys if k.startswith('h_ref')]
                P_keys = [k for k in keys if k.startswith('P_ref')]
                s_keys = [k for k in keys if k.startswith('s_ref')]
                T_keys = [k for k in keys if k.startswith('T_ref')]
                
                for k in h_keys:
                    val = res[k]
                    h_min = min(h_min, val)
                    h_max = max(h_max, val)
                
                for k in P_keys:
                    val = res[k]
                    P_min = min(P_min, val)
                    P_max = max(P_max, val)
                
                for k in s_keys:
                    val = res[k]
                    s_min = min(s_min, val)
                    s_max = max(s_max, val)
                    
                for k in T_keys:
                    val = cu.K2C(res[k]) # Convert to C
                    T_min = min(T_min, val)
                    T_max = max(T_max, val)
                    
    print(f"Global Enthalpy (h) limits: min={h_min:.2f}, max={h_max:.2f}")
    print(f"Global Pressure (P) limits: min={P_min:.2f}, max={P_max:.2f}")
    print(f"Global Entropy (s) limits: min={s_min:.2f}, max={s_max:.2f}")
    print(f"Global Temperature (T) limits: min={T_min:.2f}, max={T_max:.2f}")

if __name__ == "__main__":
    find_limits()
