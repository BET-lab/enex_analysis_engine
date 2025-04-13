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
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
#%% 

# 시뮬레이션 반복 횟수
num_simulations = 10000

# 문제 정의
problem = {
    'num_vars': 6,
    'names'   : ['Load [W]', "dT_a_int [°C]", "dT_a_ext [°C]", "dT_r_int [°C]", "dT_r_ext [°C]", "T_a_room [°C]"],
    'bounds'  : [
        [1000,10000],
        [9,12], # https://georgebrazilhvac.com/blog/what-temperature-should-my-central-air-conditioner-be-putting-out#:~:text=Your%20personal%20comfort%20should%20always,when%20you%20set%20your%20thermostat
        [9,12],
        [15,20],
        [15,20],    
        [20,25]
    ],
    'dists': ['unif', 'unif', 'unif', 'unif', 'unif', 'unif']  # 분포 형상 (균등, 삼각, 정규)
}

# Saltelli 방법을 사용하여 샘플 생성 -> 최종 샘플 개수는 num_simulations * (num_vars + 2) = 10000 * (5 + 2) = 70000
param_values = saltelli.sample(problem, num_simulations, calc_second_order=True)

# 엑서지 효율 저장 리스트
ex_eff_ASHP = []

# 몬테카를로 시뮬레이션 및 엑서지 효율 계산
for i in range(param_values.shape[0]):  
    # enex.ElectricASHP 객체 생성
    ASHP = enex.AirSourceHeatPump_heating()

    # 확률 변수에서 추출한 변수 값들 할당
    ASHP.Q_r_int  = param_values[i, 0]  
    ASHP.dT_a_int = param_values[i, 1]  
    ASHP.dT_a_ext = param_values[i, 2]  
    ASHP.dT_r_int = param_values[i, 3] 
    ASHP.dT_r_ext = param_values[i, 4]  
    
    # 시스템 업데이트 수행
    ASHP.system_update()

    # 엑서지 효율 계산
    X_eff  = (ASHP.X_a_int_out - ASHP.X_a_ext_out)/(ASHP.E_cmp + ASHP.E_fan_ext + ASHP.E_fan_int)
    ex_eff_ASHP.append(X_eff)
    

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
        sns.kdeplot(np.array(ex_eff_ASHP) * 100, ax=ax, color='tw.indigo:700', fill=True, linewidth=0.75)

        # tick limits 
        ax.set_xlim(xmin - xmar, xmax + xmar)
        ax.set_ylim(ymin - ymar, ymax + ymar)
        
        # set ticks
        ax.set_xticks(np.arange(xmin, xmax + xint*0.9, xint))
        ax.set_yticks(np.arange(ymin, ymax + yint, yint))
        
        # print x, y range
        print(f"Exergy efficiency range: {min(ex_eff_ASHP)*100:.2f} ~ {max(ex_eff_ASHP)*100:.2f}")
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
print(f"Exergy efficiency range: {min(ex_eff_ASHP)*100:.2f} ~ {max(ex_eff_ASHP)*100:.2f}")
print(f"\n\n{'='*10} Problem setting {'='*10}\n")



#%% 
# Sobol sensitivity analysis
Si = sobol.analyze(problem, np.array(ex_eff_ASHP), print_to_console=False)

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