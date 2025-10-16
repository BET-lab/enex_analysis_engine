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
df = get_weekday_df()
Toa_list = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
cooling_load_list = df['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
heating_load_list = df['DistrictHeatingWater:Facility [J](TimeStep)'] * enex.s2h

# Set up the subplots
fig, ax = plt.subplots(2, 1, figsize=(dm.cm2in(17), dm.cm2in(10)))
plt.subplots_adjust(left=0.08, right=0.975, top=0.95, bottom=0.1, hspace=0.25)

# Colors
color_temp = 'dm.green4'
color_cooling = 'dm.blue6'
color_heating = 'dm.red6'

# Common settings
line_width = 0.8

# Plot environmental temperature
ax[0].plot(Toa_list, color=color_temp, label='Outdoor temperature [$^{\circ}$C]', linewidth=line_width, alpha=0.7)
ax[0].set_ylabel('Outdoor temperature [$^{\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
ax[0].tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
ax[0].set_xticks(np.linspace(0, 8760, 7))
ax[0].set_yticks(np.arange(-10, 41, 10))
ax[0].grid(True, linestyle='--', alpha=0.6)
ax[0].text(0.01, 0.97, '(a)', transform=ax[0].transAxes, fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')
ax[0].yaxis.set_label_coords(-0.055, 0.5) 
ax[0].set_xlim(0, 8760)  
ax[0].set_ylim(-10, 40)

# Plot loads
cooling_load_list = cooling_load_list * enex.W2kW
heating_load_list = heating_load_list * enex.W2kW
cooling_load_list[cooling_load_list == 0] = -5 # 플롯에서 0이 깔려서 보이지 않도록 조정 
heating_load_list[heating_load_list == 0] = -5 # 플롯에서 0이 깔려서 보이지 않도록 조정
ax[1].plot(cooling_load_list, color=color_cooling, label='Cooling load [kW]', linewidth=line_width, alpha=0.5)
ax[1].plot(heating_load_list, color=color_heating, label='Heating load [kW]', linewidth=line_width, alpha=0.5)
ax[1].set_xlabel('Hour of year [hour]', fontsize=fs['label'], labelpad=pad['label'])
ax[1].set_ylabel('Load [kW]', fontsize=fs['label'], labelpad=pad['label'] + 2)
ax[1].tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
ax[1].set_xticks(np.linspace(0, 8760, 7))
ax[1].set_yticks(np.arange(0, 31, 10))
ax[1].grid(True, linestyle='--', alpha=0.6)
ax[1].text(0.01, 0.97, '(b)', transform=ax[1].transAxes,
           fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')
ax[1].legend(
    labels=['Cooling load', 'Heating load'],
    loc='upper right', ncol=2, frameon=False,
    fontsize=fs['legend'], fancybox=False,
    columnspacing=1.00, labelspacing=0.8,
    bbox_to_anchor=(0.97, 0.995),
    handlelength=1.8
)
ax[1].yaxis.set_label_coords(-0.055, 0.5)  # y축 레이블 위치 고정
ax[1].set_xlim(0, 8760)  # x축 범위 고정
ax[1].set_ylim(0, 30)  # y축 범위 고정

# Save and show the figure
plt.savefig('../figure/Fig. 3.png', dpi=600)
plt.savefig('../figure/Fig. 3.pdf', dpi=600)
dm.util.save_and_show(fig)