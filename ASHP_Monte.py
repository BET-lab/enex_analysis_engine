#%% 
import sys
sys.path.append('src')
import en_system_ex_analysis as enex
from SALib.sample import saltelli
from SALib.analyze import sobol
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dartwork_mpl as dm
import seaborn as sns
from scipy.stats import norm, uniform, triang, gamma
import warnings
from pprint import pprint

#%% 

# 시뮬레이션 반복 횟수
num_simulations = 10000

# 문제 정의
problem = {
    'num_vars': 4,
    'names'   : ['Insulation thick [m]',  'Outdoor air temp [°C]', "Supply water temp [°C]", "COP [-]"],
    'bounds'  : [
        [0.01, 0.1],    # x_ins      : 단열 두께 (m)
        [enex.C2K(-10), enex.C2K(10)],    # T_oa: 실외 온도 (°C)
        [enex.C2K(4), enex.C2K(10)],    # T_sw: 상수도 온도 (°C)
        [1.5, 2.5],    # COP: COP
        [10000, 100000]  # Q (W)
    ],
    'dists': ['unif', 'unif', 'unif', 'unif']  # 분포 형상 (균등, 삼각, 정규)
}

# Saltelli 방법을 사용하여 샘플 생성 -> 최종 샘플 개수는 num_simulations * (num_vars + 2) = 10000 * (5 + 2) = 70000
param_values = saltelli.sample(problem, num_simulations, calc_second_order=True)

# 엑서지 효율 저장 리스트
ex_eff_HPB = []
cop_list = []
T_r_ext_list = []

# 몬테카를로 시뮬레이션 및 엑서지 효율 계산
for i in range(param_values.shape[0]):  
    # enex.ElectricASHP 객체 생성
    ASHP = enex.HeatPumpASHP()

    # 확률 변수에서 추출한 변수 값들 할당
    ASHP.x_ins              = param_values[i, 0]  # 단열 두께
    ASHP.T0                 = param_values[i, 1]  # 대류 열전달 계수
    ASHP.T_w_sup            = param_values[i, 2]  # 상수도 온도
    ASHP.Q_w_serv          = param_values[i, 3]  # 상수도 온도
    
    # COP calculation
    dT = ASHP.T_w_tank - ASHP.T0 # 온수 탱크 온도와 실외 온도 차이
    ASHP.COP_hp             = 5.06-0.05*dT +0.00006*(dT**2) # https://www.notion.so/betlab/Model-based-flexibility-assessment-of-a-residential-heat-pump-pool-1af6947d125d8037a108e85bb6985dee?pvs=4
    
    
    cop_list.append(ASHP.COP_hp)
    

    # 시스템 업데이트 수행
    ASHP.system_update()
    T_r_ext_list.append(ASHP.T_r_ext)

    # 엑서지 효율 계산
    X_w_serv = ASHP.Xout 
    E_fan    = ASHP.exergy_balance["external unit"]["in"]["E_fan"]
    E_cmp    = ASHP.exergy_balance["refrigerant loop"]["in"]["E_cmp"]
    
    eta_exergy = X_w_serv / (E_fan + E_cmp)
    ex_eff_HPB.append(eta_exergy)

#%%
# 데이터 시각화
nrows = 1
ncols = 1

xmin, xmax, xint, xmar = 0, 40, 10, 0
ymin, ymax, yint, ymar = 0, 0.15,  0.05,  0

fig, ax = plt.subplots(nrows, ncols, figsize=(dm.cm2in(10), dm.cm2in(5)), dpi=600)
for ridx in range(nrows): 
    for cidx in range(ncols): 
        idx = ridx * ncols + cidx

        # Plot histogram
        sns.kdeplot(np.array(ex_eff_HPB) * 100, ax=ax, color='tw.indigo:700', fill=True, linewidth=0.75)

        # tick limits 
        ax.set_xlim(xmin - xmar, xmax + xmar)
        ax.set_ylim(ymin - ymar, ymax + ymar)
        
        # set ticks
        ax.set_xticks(np.arange(xmin, xmax + xint*0.9, xint))
        ax.set_yticks(np.arange(ymin, ymax + yint, yint))
        
        # print x, y range
        print(f"Exergy efficiency range: {min(ex_eff_HPB)*100:.2f} ~ {max(ex_eff_HPB)*100:.2f}")
        # print(f"y max: {max_frequency}")
        

# set tick parameters
plt.xlabel("Exergy efficiency [%]", fontsize=dm.fs(0), fontweight=300)
plt.ylabel("Probability density", fontsize=dm.fs(0), fontweight=300)

# layout
dm.simple_layout(fig, bbox=(0, 1, 0.01, 1), verbose=False)

# save
fig_name = 'ex_eff_distribution_HPB'
folder_path = r'figure'
fig.savefig(f'{folder_path}/{fig_name}.png', dpi=600)

# show
dm.util.save_and_show(fig)
plt.close()

# print
print(f"Exergy efficiency range: {min(ex_eff_HPB)*100:.2f} ~ {max(ex_eff_HPB)*100:.2f}")
print(f"cop range: {min(cop_list):.2f} ~ {max(cop_list):.2f}")
print(f"\n\n{'='*10} Problem setting {'='*10}\n")



#%% 
# Sobol sensitivity analysis
Si = sobol.analyze(problem, np.array(ex_eff_EB), print_to_console=False)

# Visualize first-order sensitivity indices
fig, ax = plt.subplots(figsize=(dm.cm2in(10), dm.cm2in(5)), dpi=600)

# Bar plot for first-order sensitivity indices
ax.bar(problem['names'], Si['S1'], color='tw.indigo:700', alpha=0.7)

# Add labels and title
ax.set_ylabel('First-order Sensitivity Index', fontsize=dm.fs(0), fontweight=300)
ax.set_xlabel('Input Variables', fontsize=dm.fs(0), fontweight=300)
ax.set_title('Sobol Sensitivity Analysis', fontsize=dm.fs(1), fontweight=400)

# Layout adjustments
dm.simple_layout(fig, bbox=(0, 1, 0.01, 1), verbose=False)

# Save and show the figure
fig_name = 'sobol_sensitivity_EB'
fig.savefig(f'{folder_path}/{fig_name}.png', dpi=600)
dm.util.save_and_show(fig)
plt.close()