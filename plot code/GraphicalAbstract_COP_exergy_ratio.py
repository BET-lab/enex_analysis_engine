#%%
# Graphical Abstract - Monthly Average COP and Exergy Output/Input Ratio (combined)
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
from data_setting import get_weekday_df
from enex_analysis.plot_style import fs, pad
dm.use_style()

# Data
weekday_df = get_weekday_df()
cooling_load_list = weekday_df['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
heating_load_list = weekday_df['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h

grouped = weekday_df.groupby('Month')

monthly_avg_COP = []
monthly_exergy_input = []
monthly_exergy_output = []

for month, group in grouped:
    total_cooling_COP = 0
    total_heating_COP = 0
    cooling_count = 0
    heating_count = 0
    
    input_exergy = 0
    output_exergy = 0

    for _, row in group.iterrows():
        Toa = row['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        cooling_load = row['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
        heating_load = row['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h

        if cooling_load > 0:
            ASHP_cooling = enex.AirSourceHeatPump_cooling()
            ASHP_cooling.T0 = Toa
            ASHP_cooling.T_a_room = 22
            ASHP_cooling.Q_r_int = cooling_load
            ASHP_cooling.Q_r_max = max(cooling_load_list)
            ASHP_cooling.system_update()
            if ASHP_cooling.X_eff > 0:
                total_cooling_COP += ASHP_cooling.COP_sys
                cooling_count += 1
                input_exergy += (ASHP_cooling.E_cmp + ASHP_cooling.E_fan_int + ASHP_cooling.E_fan_ext) * enex.h2s
                output_exergy += (ASHP_cooling.X_a_int_out - ASHP_cooling.X_a_int_in) * enex.h2s
                
        if heating_load > 0:
            ASHP_heating = enex.AirSourceHeatPump_heating()
            ASHP_heating.T0 = Toa
            ASHP_heating.T_a_room = 22
            ASHP_heating.Q_r_int = heating_load
            ASHP_heating.Q_r_max = max(heating_load_list)
            ASHP_heating.system_update()
            if ASHP_heating.X_eff > 0:
                total_heating_COP += ASHP_heating.COP_sys
                heating_count += 1
                input_exergy += (ASHP_heating.E_cmp + ASHP_heating.E_fan_int + ASHP_heating.E_fan_ext) * enex.h2s
                output_exergy += (ASHP_heating.X_a_int_out - ASHP_heating.X_a_int_in) * enex.h2s

    avg_COP = (total_cooling_COP + total_heating_COP) / (cooling_count + heating_count) if (cooling_count + heating_count) > 0 else None
    monthly_avg_COP.append(avg_COP if avg_COP is not None else 0)
    
    monthly_exergy_input.append(input_exergy * enex.W2GW)
    monthly_exergy_output.append(output_exergy * enex.W2GW)

# Calculate ratio as percentage
monthly_exergy_ratio = []
for i in range(len(monthly_exergy_input)):
    if monthly_exergy_input[i] > 0:
        ratio = (monthly_exergy_output[i] / monthly_exergy_input[i]) * 100.0
        monthly_exergy_ratio.append(ratio)
    else:
        monthly_exergy_ratio.append(0.0)

labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
labels = ['Jan', '', '', '', '', 'Jun', '', '', '', '', '', 'Dec']

x = np.arange(1, 13)

# Graphical Abstract Settings
avg_cop_color = 'dm.orange6'
ratio_color = 'dm.green5'
marker_size = 4.0  # Increased for graphical abstract
line_width = 2.5  # Increased for graphical abstract
spine_width = 1.5  # Increased for graphical abstract

# Font sizes - increased for graphical abstract
fs_ga = {
    'label': fs['label'],
    'tick': fs['tick'],
    'legend': fs['legend'],
}

# Figure size - half of original (17x11 -> 8.5x5.5)
FIG_W_CM, FIG_H_CM = 8.5, 5.5

fig, ax1 = plt.subplots(1, 1, figsize=(dm.cm2in(FIG_W_CM), dm.cm2in(FIG_H_CM)))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)

# Create second y-axis
ax2 = ax1.twinx()

# Plot monthly average COP on left y-axis
line1 = ax1.plot(x, monthly_avg_COP, color=avg_cop_color, linewidth=line_width, 
                 label='COP', marker='o', markersize=marker_size, zorder=2)

# Plot exergy output/input ratio on right y-axis
line2 = ax2.plot(x, monthly_exergy_ratio, color=ratio_color, linewidth=line_width, 
                 label='Exergy efficiency', marker='o', markersize=marker_size, zorder=2)

# ==== Variable declaration for axis limits and ticks ====
# X axis (Month)
XMIN = 0.5
XMAX = 12.5
XINT = 1
XMAR = x  # already defined: np.arange(1, 13)
# Y axis 1 (COP)
Y1MIN = 0.0
Y1MAX = 5.0
Y1TICK = 1.0
Y1MAR = np.arange(Y1MIN, Y1MAX + 1e-9, Y1TICK)
# Y axis 2 (Exergy ratio)
max_ratio = max(monthly_exergy_ratio) if monthly_exergy_ratio else 50
Y2MIN = 0.0
Y2MAX = max_ratio * 1.1 if max_ratio > 0 else 50
Y2TICK = 10
Y2MAR = np.arange(Y2MIN, Y2MAX * 1.09 + 1e-9, Y2TICK)

# ==== Set limits and ticks using variables ====
# Left y-axis settings (COP)
ax1.set_xlim(XMIN, XMAX)
ax1.set_ylim(Y1MIN, Y1MAX)
ax1.set_xticks(XMAR)
ax1.set_xticklabels(labels, fontsize=fs_ga['tick'])

# Set the number of minor ticks for x and y axes to 0
ax1.xaxis.set_minor_locator(plt.NullLocator())
ax1.yaxis.set_minor_locator(plt.NullLocator())

# x축 major tick length를 0으로 설정
ax1.tick_params(axis='x', which='major', length=0)

# y축 major/minor tick length를 0으로 설정, label도 숨기기
ax1.tick_params(axis='y', which='major', length=0, label1On=False)
ax1.tick_params(axis='y', which='minor', length=0, label1On=False)

# ax1.set_xlabel('Month', fontsize=fs_ga['label'])
# ylabel and y ticks removed

ax1.tick_params(axis='x', which='major', labelsize=fs_ga['tick'], pad=pad['tick'])

# Right y-axis settings (Exergy ratio)
ax2.set_ylim(Y2MIN, Y2MAX)
# y축 major/minor tick length를 0으로 설정, label도 숨기기
ax2.tick_params(axis='y', which='major', length=0)
ax2.tick_params(axis='y', which='minor', length=0)

# Set the number of minor ticks for right y-axis to 0
ax2.yaxis.set_minor_locator(plt.NullLocator())
ax2.xaxis.set_minor_locator(plt.NullLocator())

# y축 숫자(라벨)를 완전히 숨기기
ax2.set_yticklabels([])

# Spine - only bottom (x-axis) visible
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_visible(False)

ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['top'].set_visible(False)

# for spine in ax1.spines.values():
#     spine.set_linewidth(spine_width)
# for spine in ax2.spines.values():
#     spine.set_linewidth(spine_width)


# Legend - combine both axes, increase handle and label spacing
lines = line1 + line2
labels_combined = [l.get_label() for l in lines]
ax1.legend(
    lines,
    labels_combined,
    loc='upper right',
    fontsize=fs_ga['legend'],
    ncols=2,
    handletextpad=1.0,   # handle과 label 사이 간격(default=0.8)
    columnspacing=1.5    # 각 컬럼 간 간격(default=2.0)
)

plt.savefig('../figure/GraphicalAbstract_COP_exergy_ratio.png', dpi=600)
plt.savefig('../figure/GraphicalAbstract_COP_exergy_ratio.svg', dpi=600, transparent=True)
dm.util.save_and_show(fig)
