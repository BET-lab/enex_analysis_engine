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
# 난방모드 

# 시뮬레이션 반복 횟수
num_simulations = 10000

# 문제 정의
problem = {
    'num_vars': 4,
    # 'names'   : ["Q_r_int","dT_a_int [°C]", "dT_a_ext [°C]", "dT_r_int [°C]", "dT_r_ext [°C]", "T_a_room [°C]"],
    'names'   : ["T_a_int_out [°C]", "T_a_ext_out [°C]", "T_r_int [°C]", "T_r_ext [°C]"],
    'bounds'  : [
        [enex.C2K(20)+9, enex.C2K(20)+12], # https://georgebrazilhvac.com/blog/what-temperature-should-my-central-air-conditioner-be-putting-out#:~:text=Your%20personal%20comfort%20should%20always,when%20you%20set%20your%20thermostat
        [enex.C2K(0)-12, enex.C2K(0)-9],
        [enex.C2K(20)+15,enex.C2K(20)+20],
        [enex.C2K(0)-20, enex.C2K(0)-15],
    ],
    'dists': ['unif', 'unif', 'unif', 'unif']  # 분포 형상 (균등, 삼각, 정규)
}

# Saltelli 방법을 사용하여 샘플 생성 -> 최종 샘플 개수는 num_simulations * (num_vars + 2) = 10000 * (5 + 2) = 70000
param_values = saltelli.sample(problem, num_simulations, calc_second_order=True)

# 엑서지 효율 저장 리스트
ex_eff_ASHP_heating = []

# 몬테카를로 시뮬레이션 및 엑서지 효율 계산
ASHP = enex.AirSourceHeatPump_heating()
ASHP.T_0 = enex.C2K(0)
ASHP.Q_r_int  = 7000
for i in range(param_values.shape[0]):  
    # enex.ElectricASHP 객체 생성

    # 확률 변수에서 추출한 변수 값들 할당
    ASHP.T_a_int_out = param_values[i, 0]  
    ASHP.T_a_ext_out = param_values[i, 1]  
    ASHP.T_r_int = param_values[i, 2] 
    ASHP.T_r_ext = param_values[i, 3]  

    # 시스템 업데이트 수행
    ASHP.system_update()

    # 엑서지 효율 계산
    X_eff  = (ASHP.X_a_int_out - ASHP.X_a_int_in)/(ASHP.E_cmp + ASHP.E_fan_ext + ASHP.E_fan_int)
    ex_eff_ASHP_heating.append(X_eff)
    
#%% 
# 냉방모드

# 시뮬레이션 반복 횟수
num_simulations = 10000

# 문제 정의
problem = {
    'num_vars': 4,
    # 'names'   : ["Q_r_int","dT_a_int [°C]", "dT_a_ext [°C]", "dT_r_int [°C]", "dT_r_ext [°C]", "T_a_room [°C]"],
    'names'   : ["T_a_int_out [°C]", "T_a_ext_out [°C]", "T_r_int [°C]", "T_r_ext [°C]"],
    'bounds'  : [
        [enex.C2K(20)-12, enex.C2K(20)-9], # https://georgebrazilhvac.com/blog/what-temperature-should-my-central-air-conditioner-be-putting-out#:~:text=Your%20personal%20comfort%20should%20always,when%20you%20set%20your%20thermostat
        [enex.C2K(30)+9,enex.C2K(30)+12],
        [enex.C2K(20)-20, enex.C2K(20)-15],
        [enex.C2K(30)+15,enex.C2K(30)+20],
    ],
    'dists': ['unif', 'unif', 'unif', 'unif']  # 분포 형상 (균등, 삼각, 정규)
}

# Saltelli 방법을 사용하여 샘플 생성 -> 최종 샘플 개수는 num_simulations * (num_vars + 2) = 10000 * (5 + 2) = 70000
param_values = saltelli.sample(problem, num_simulations, calc_second_order=True)

# 엑서지 효율 저장 리스트
ex_eff_ASHP_cooling = []

# 몬테카를로 시뮬레이션 및 엑서지 효율 계산
ASHP = enex.AirSourceHeatPump_cooling()
ASHP.T_0 = enex.C2K(30)
ASHP.Q_r_int  = 7000
for i in range(param_values.shape[0]): 
    # enex.ElectricASHP 객체 생성

    # 확률 변수에서 추출한 변수 값들 할당
    ASHP.T_a_int_out = param_values[i, 0]  
    ASHP.T_a_ext_out = param_values[i, 1]  
    ASHP.T_r_int = param_values[i, 2] 
    ASHP.T_r_ext = param_values[i, 3]  

    # 시스템 업데이트 수행
    ASHP.system_update()

    # 엑서지 효율 계산
    X_eff  = (ASHP.X_a_int_out - ASHP.X_a_int_in)/(ASHP.E_cmp + ASHP.E_fan_ext + ASHP.E_fan_int)
    ex_eff_ASHP_cooling.append(X_eff)
#%%
# 데이터 시각화
nrows = 1
ncols = 1

ex_eff_ASHP = [ex_eff_ASHP_heating, ex_eff_ASHP_cooling]

xmin, xmax, xint, xmar = 10, 30, 5, 0
ymin, ymax, yint, ymar = 0, 0.4,  0.1,  0

# colors = ['tw.blue:300', 'tw.blue:400', 'tw.blue:500', 'tw.blue:600', 'tw.blue:700']
colors = ['tw.red:300', 'tw.blue:400']

fig, ax = plt.subplots(nrows, ncols, figsize=(dm.cm2in(10), dm.cm2in(5)), dpi=600)
for ridx in range(nrows): 
    for cidx in range(ncols): 
        for tidx in range(len(ex_eff_ASHP)):
            
            # Plot histogram
            idx = ridx * ncols + cidx

            # Plot histogram
            sns.kdeplot(np.array(ex_eff_ASHP[tidx]) * 100, ax=ax, color=colors[tidx], fill=True, linewidth=0.75)

            # tick limits 
            ax.set_xlim(xmin - xmar, xmax + xmar)
            ax.set_ylim(ymin - ymar, ymax + ymar)
            
            # set ticks
            ax.set_xticks(np.arange(xmin, xmax + xint*0.9, xint))
            ax.set_yticks(np.arange(ymin, ymax*1.001, yint))
            
            ax.legend(["heating mode", "cooling mode"], loc='best', handletextpad=0.7)
            
            # print x, y range
            print(f"Exergy efficiency range: {min(ex_eff_ASHP[tidx])*100:.2f} ~ {max(ex_eff_ASHP[tidx])*100:.2f}")
            # print(f"y max: {max_frequency}")

# set tick parameters
plt.xlabel("Exergy efficiency [%]", fontsize=dm.fs(0), fontweight=300)
plt.ylabel("Probability density", fontsize=dm.fs(0), fontweight=300)

# layout
dm.simple_layout(fig, bbox=(0, 1, 0.01, 1), verbose=False)

# save
fig_name = 'ex_eff_distribution_ASHP2'
folder_path = r'figure'
fig.savefig(f'{folder_path}/{fig_name}.png', dpi=600)

# show
dm.util.save_and_show(fig)
plt.close()

# print
print(f"Exergy efficiency range: {min(ex_eff_ASHP[tidx])*100:.2f} ~ {max(ex_eff_ASHP[tidx])*100:.2f}")
print(f"\n\n{'='*10} Problem setting {'='*10}\n")



#%% 
Si_list = []
for i in range(2):
    Y = np.array(ex_eff_ASHP[i])
    Si = sobol.analyze(problem, Y, print_to_console=False)
    Si_list.append(Si['S1'])  # 1차 민감도만 저장

# 2D array (2 conditions × 4 variables)에서 내부, 외부 온도 변수만 선택 (첫 2개 변수)
S1_array = np.array(Si_list)[:, :2]

colors = ['tw.red:300', 'tw.blue:400']

# 시각화: 가로형 막대 그래프 (내부, 외부 온도만)
fig, ax = plt.subplots(figsize=(dm.cm2in(12), dm.cm2in(3)), dpi=600)

n_T0 = S1_array.shape[0]   # 조건 개수 = 2
n_vars = S1_array.shape[1] # 변수 개수 = 2
bar_height = 0.15          # 막대 높이
y = np.arange(n_vars)      # y축 위치 (각 변수 별)

# 외기온도 조건 레이블
T0_labels = ['0 °C', '30 °C']

# 막대 그래프 그리기 (수평 막대)
for i in range(n_T0):
    ax.barh(y + i * bar_height, S1_array[i] * 100, height=bar_height,
            label=['heating mode','cooling mode'][i], color=colors[i], alpha=0.8)

# x축 (값의 범위) 설정: 백분율이므로 0~100
ax.set_xlim(0, 100)

# y축 틱 설정 및 변수 라벨 지정 (내부 온도, 외부 온도)
ax.set_yticks(y + bar_height * (n_T0 - 1) / 2)
ax.set_yticklabels([
    'Internal unit\nexhaust air\ntemperature', 
    'External unit\nexhaust air\ntemperature'
], fontsize=dm.fs(-2), fontweight=300)

# x축 라벨 설정
ax.set_xlabel("Percentage [%]", fontsize=dm.fs(0), fontweight=300)
ax.tick_params(axis='y', labelsize=dm.fs(-1), which='major', length=0, pad=5)

# 범례
ax.legend(fontsize=dm.fs(-1), title_fontsize=dm.fs(-1), loc='upper right', bbox_to_anchor=(1, 1.2), ncol=2)

# 레이아웃 조정
dm.simple_layout(fig, bbox=(0, 1, 0.07, 0.99), verbose=False)

# 저장 및 출력
fig_name = 'sobol_sensitivity_comparison_selected'
fig.savefig(f'{folder_path}/{fig_name}.png', dpi=600)
dm.util.save_and_show(fig)
plt.close()
# %%

ASHP = enex.AirSourceHeatPump_cooling()
ASHP.Q_r_int  = 7000
ASHP.T_a_room = enex.C2K(20)
ASHP.T_0 = enex.C2K(30)
ASHP.system_update()
X_eff  = (ASHP.X_a_int_out - ASHP.X_a_int_in)/(ASHP.E_cmp + ASHP.E_fan_ext + ASHP.E_fan_int)
print(f"X_a_int_in: {ASHP.X_a_int_in} W")
print(f"X_a_int_out: {ASHP.X_a_int_out} W")
print(f"T_a_int_out: {enex.K2C(ASHP.T_a_int_out)} °C")
print(f"Exergy efficiency: {X_eff*100:.2f} %")

# %%
Si['S1']