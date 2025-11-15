#%%
# import libraries
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
from enex_analysis.plot_style import fs, pad
dm.use_style()

# ===============================
# 1. 기본 설정
# ===============================
time = np.arange(0, 25, 1)  # 0~24시간

# 최대값 (W/m2)
equipment_max = 10.76
light_max = 10.76
occupant_max = 28 * 120 / 511

# ===============================
# 2. Fraction 스케줄 정의
# ===============================
def step_schedule(times, values):
    """주어진 시각별 fraction을 1시간 단위로 확장"""
    sched = []
    prev_time = 0
    for t, val in zip(times, values):
        sched.extend([val] * (t - prev_time))
        prev_time = t
    return np.array(sched)

# Equipment 스케줄
eq_times = [8, 12, 13, 17, 18, 24]
eq_vals  = [0.40, 0.90, 0.80, 0.90, 0.50, 0.40]
equipment = step_schedule(eq_times, eq_vals) * equipment_max

# Light 스케줄
lt_times = [5, 7, 8, 17, 18, 20, 22, 23, 24]
lt_vals  = [0.05, 0.1, 0.3, 0.9, 0.5, 0.3, 0.2, 0.1, 0.05]
light = step_schedule(lt_times, lt_vals) * light_max

# Occupant 스케줄
oc_times = [6, 7, 8, 12, 13, 17, 18, 20, 24]
oc_vals  = [0.00, 0.1, 0.2, 0.95, 0.5, 0.95, 0.3, 0.1, 0.05]
occupant = step_schedule(oc_times, oc_vals) * occupant_max

# ===============================
# 3. 그래프 그리기
# ===============================
fig, ax = plt.subplots(1, 1, figsize=(dm.cm2in(9), dm.cm2in(6)))
plt.subplots_adjust(left=0.12, right=0.95, top=0.95, bottom=0.16)
ax.step(time, np.append(equipment, equipment[-1]), where='post', label='Equipment', color='dm.teal5', linewidth=0.8)
ax.step(time, np.append(light, light[-1]), where='post', label='Light', color='dm.blue5', linestyle='-.', linewidth=0.8)
ax.step(time, np.append(occupant, occupant[-1]), where='post', label='Occupant', color='dm.red5', linestyle=':', linewidth=0.8)


ax.set_xlabel('Time [h]', fontsize=fs['label'], labelpad=pad['label'])
ax.set_ylabel('Internal heat gain [W/m$^2$]', fontsize=fs['label'], labelpad=pad['label'])
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 24.1, 6))
ax.set_xticklabels(np.arange(0, 25, 6), fontsize=fs['tick'])
ax.set_ylim(-1, 12)
ax.set_yticks(np.arange(0, 13, 4))
ax.set_yticklabels(np.arange(0, 13, 4), fontsize=fs['tick'])
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))  # 1시간 단위 마이너틱
ax.grid(True, linestyle=':', linewidth=0.2, alpha=0.6, which='major')
legend = ax.legend(
    loc='upper right', ncol=1, frameon=False,
    fontsize=fs['legend'], fancybox=False,
    columnspacing=1.0, labelspacing=0.5,
    bbox_to_anchor=(1.0, 1.01),
    handlelength=1.8
)

plt.savefig('../figure/Fig.A1.png', dpi=600)
plt.savefig('../figure/Fig.A1.pdf', dpi=600)
dm.util.save_and_show(fig)
# %%
