#%%
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
heating_load_list = weekday_df['DistrictHeatingWater:Facility [J](TimeStep)'] * enex.s2h

# 월별 그룹화
grouped = weekday_df.groupby('Month')

# 월별 리스트 생성
monthly_exergy_input = []
monthly_exergy_consumption = []
monthly_exergy_output = []
monthly_avg_COP = []
monthly_exergy_efficiency = []

# 월별 온도 리스트 생성
monthly_avg_temp = grouped['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].mean().tolist()
monthly_min_temp = grouped['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].min().tolist()
monthly_max_temp = grouped['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'].max().tolist()

for month, group in grouped:
    input_exergy = 0
    consumption_exergy = 0
    output_exergy = 0
    
    total_cooling_exergy_efficiency = 0
    total_heating_exergy_efficiency = 0
    total_cooling_COP = 0
    total_heating_COP = 0
    cooling_count = 0
    heating_count = 0

    for _, row in group.iterrows():
        Toa = row['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
        Tia = row['CORE_ZN:Zone Air Temperature [C](TimeStep)']
        cooling_load = row['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
        heating_load = row['DistrictHeatingWater:Facility [J](TimeStep)'] * enex.s2h

        if cooling_load > 0:
            ASHP_cooling = enex.AirSourceHeatPump_cooling()
            ASHP_cooling.T0 = Toa
            ASHP_cooling.T_a_room = 22
            ASHP_cooling.Q_r_int = cooling_load
            ASHP_cooling.Q_r_max = max(cooling_load_list)
            ASHP_cooling.system_update()
            if ASHP_cooling.X_eff > 0:
                input_exergy += (ASHP_cooling.E_cmp + ASHP_cooling.E_fan_int + ASHP_cooling.E_fan_ext) * enex.h2s
                output_exergy += (ASHP_cooling.X_a_int_out - ASHP_cooling.X_a_int_in) * enex.h2s
                total_cooling_COP += ASHP_cooling.COP_sys
                total_cooling_exergy_efficiency += ASHP_cooling.X_eff
                cooling_count += 1
            else:
                cooling_count += 0
        else:
            cooling_count += 0
            
        if heating_load > 0:
            ASHP_heating = enex.AirSourceHeatPump_heating()
            ASHP_heating.T0 = Toa
            ASHP_heating.T_a_room = 22
            ASHP_heating.Q_r_int = heating_load
            ASHP_heating.Q_r_max = max(heating_load_list)
            ASHP_heating.system_update()
            if ASHP_heating.X_eff > 0:
                input_exergy += (ASHP_heating.E_cmp + ASHP_heating.E_fan_int + ASHP_heating.E_fan_ext) * enex.h2s
                output_exergy += (ASHP_heating.X_a_int_out - ASHP_heating.X_a_int_in) * enex.h2s
                total_heating_COP += ASHP_heating.COP_sys
                total_heating_exergy_efficiency += ASHP_heating.X_eff
                heating_count += 1
            else:
                heating_count += 0
        else:
            heating_count += 0

    consumption_exergy = input_exergy - output_exergy

    monthly_exergy_input.append(input_exergy * enex.W2GW)
    monthly_exergy_consumption.append(consumption_exergy * enex.W2GW)
    monthly_exergy_output.append(output_exergy * enex.W2GW)

    avg_exergy_efficiency = (total_cooling_exergy_efficiency + total_heating_exergy_efficiency) / (cooling_count + heating_count) if (cooling_count + heating_count) > 0 else None
    monthly_exergy_efficiency.append(avg_exergy_efficiency * 100 if avg_exergy_efficiency is not None else 0)
    avg_COP = (total_cooling_COP + total_heating_COP) / (cooling_count + heating_count) if (cooling_count + heating_count) > 0 else None
    monthly_avg_COP.append(avg_COP)

labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(1, 13)

# Create figure with 2 subplots vertically stacked
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(dm.cm2in(17), dm.cm2in(11)), gridspec_kw={'height_ratios': [1.0, 1.4]})
plt.subplots_adjust(left=0.07, right=0.94, top=0.97, bottom=0.04, wspace=0.2, hspace=0.2)

# Colors
fill_color = 'dm.gray1'
min_temp_color = 'dm.blue5'
max_temp_color = 'dm.red5'
avg_temp_color = 'dm.gray8'
percentage_color = 'dm.black'
avg_cop_color = 'dm.orange6'

# Common setting
marker_size = 1.5
line_width = 0.5

# --- FIRST SUBPLOT: TEMPERATURE ---
ax1.fill_between(x, monthly_min_temp, monthly_max_temp, color=fill_color, alpha=0.6)

# ax1 plot
ax1.plot(x, monthly_max_temp, color=max_temp_color, linewidth=line_width, label='Max', marker='o', markersize=marker_size)
ax1.plot(x, monthly_avg_temp, color=avg_temp_color, linewidth=line_width, label='Avg', marker='o', markersize=marker_size)
ax1.plot(x, monthly_min_temp, color=min_temp_color, linewidth=line_width, label='Min', marker='o', markersize=marker_size)

# Limit
ax1.set_xlim(0.5, 12.5)
ax1.set_ylim(-10, 40)

# Tick
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=fs['tick'])
ax1.set_yticks(np.arange(-10, 41, 10))
ax1.set_yticklabels(np.arange(-10, 41, 10), fontsize=fs['tick'])
ax1.tick_params(axis='both', which='major', labelsize=fs['tick'])
ax1.tick_params(axis='x', which='minor', bottom=False)
ax2.text(0.01, 0.97, '(a)', transform=ax1.transAxes, fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')

# Axis and title
ax1.set_ylabel('Environmental temp. [$^{\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(
    loc='upper right', ncol=3, frameon=False,
    fontsize=fs['legend'], fancybox=False,
    columnspacing=1.00, labelspacing=0.8,
    bbox_to_anchor=(0.99, 1.01),
    handlelength=1.8
)

ax1.axhline(y=22, color='dm.teal6', linestyle='--', linewidth=line_width, label='Setpoint')
ax1.text(0.6, 24, 'Setpoint', rotation=0, fontsize=fs['setpoint'], color='dm.teal6', ha='left', va='center')

# --- SECOND SUBPLOT: EXERGY STACKED BAR ---
# Exergy input
total_exergy = np.array(monthly_exergy_input)

# Common settings
bar_width = 0.57
spine_width = 0.5

# Colors
colors = ['dm.gray2', 'dm.green3', 'dm.orange7']
edge_color = 'dm.gray8'

# Stacked bar plot
bar_top = ax2.bar(x, monthly_exergy_consumption, bar_width, bottom=monthly_exergy_output, 
                 label='Exergy consumption', color=colors[0], alpha=1.0, zorder=3)
bar_bottom = ax2.bar(x, monthly_exergy_output, bar_width, 
                    label='Exergy output', color=colors[1], alpha=0.75, zorder=3)

# Add percentage labels
for i in range(len(labels)):
    x_pos = x[i]
    consumption = monthly_exergy_consumption[i]
    output = monthly_exergy_output[i]
    consumption_ratio = consumption * 100 / total_exergy[i]
    output_ratio = output * 100 / total_exergy[i]
    ax2.text(
        x_pos, output + consumption / 2,
        f'{consumption_ratio:.1f}%',
        ha='center', va='center',
        fontsize=fs['text'], color=percentage_color, zorder=4)
    ax2.text(
        x_pos, output / 2,
        f'{output_ratio:.1f}%',
        ha='center', va='center',
        fontsize=fs['text'], color=percentage_color, zorder=4)
    
# Add rectangle outlines
for i in range(len(labels)):
    total_height = monthly_exergy_consumption[i] + monthly_exergy_output[i]
    rect = plt.Rectangle(
        (x[i] - bar_width / 2, 0),
        bar_width,
        total_height,
        linewidth=line_width,
        edgecolor=edge_color,
        facecolor='none',
        alpha=0.7, zorder=3
    )
    ax2.add_patch(rect)

# Create custom legend element
from matplotlib.patches import Patch
input_legend = Patch(
    facecolor='none', edgecolor=edge_color,
    linewidth=line_width, alpha=0.7,
    label='Exergy input'
)

# Label
ax2.set_ylabel('Exergy [GWh]', fontsize=fs['label'], labelpad=pad['label'])

for ax in (ax1, ax2):
    ax.yaxis.set_label_coords(-0.05, 0.5)  # x는 좌우 위치, y는 세로 중앙

# Limit
ax2.set_xlim(0.5, 12.5)
ax2.set_ylim(0, 5)

# Tick
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=fs['tick'])
ax2.set_ylim(0.0, 5.0)
ax2.set_yticklabels(np.arange(0, 6, 1), fontsize=fs['tick'])
ax2.tick_params(axis='both', which='major', labelsize=fs['tick'])
ax2.tick_params(axis='x', which='major', bottom=False)
ax2.tick_params(axis='x', which='minor', bottom=False)
ax2.text(0.01, 0.97, '(b)', transform=ax2.transAxes, fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')
ax2.spines['right'].set_visible(False)

# Add secondary y-axis for monthly_avg_COP
ax2_right = ax2.twinx()
ax2_right.plot(x, monthly_avg_COP, color=avg_cop_color, linewidth=line_width, label='COP', marker='o', markersize=marker_size)
ax2_right.set_ylabel('Average COP$_{sys}$ [ - ]', fontsize=fs['label'], labelpad=pad['label'], color=avg_cop_color)
ax2_right.set_ylim(0.0, 5.0)
ax2_right.set_yticks(np.arange(0.0, 5.1, 1.0))
ax2_right.tick_params(axis='y', labelsize=fs['tick'], color=avg_cop_color, labelcolor=avg_cop_color)
ax2_right.tick_params(axis='y', which='minor', color=avg_cop_color)
ax2_right.spines['right'].set_color(avg_cop_color)
ax2_right.spines['right'].set_linewidth(spine_width)

# Grid
ax2.grid(True, linestyle='--', alpha=0.6, zorder=1)
ax2.xaxis.grid(False)

# Legend (combined for ax2 and ax2_right, 4 columns)
handles_left, labels_left = ax2.get_legend_handles_labels()
handles_left.insert(0, input_legend)
labels_left.insert(0, 'Exergy input')

handles_right, labels_right = ax2_right.get_legend_handles_labels()

# Combine handles and labels
handles_combined = handles_left + handles_right
labels_combined = labels_left + labels_right

legend = ax2.legend(
    handles_combined, labels_combined,
    loc='upper right', ncol=4, frameon=False,
    fontsize=fs['legend'], fancybox=False,
    columnspacing=1.0, labelspacing=1.0,
    bbox_to_anchor=(0.99, 1.01),
    handlelength=1.8
)

plt.savefig('../figure/Fig. 15.png', dpi=600)
plt.savefig('../figure/Fig. 15.svg', dpi=600, transparent=True)
plt.savefig('../figure/Fig. 15.pdf', dpi=600, transparent=True)
dm.util.save_and_show(fig)