#%%
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
from enex_analysis.plot_style import fs, pad
import matplotlib.colors as mcolors
from matplotlib.cm import get_cmap
dm.use_style()

# parameter
T_ev_l = 12
COP_ref = 4.0

# 2d grid for cooling
PLR_cooling = np.linspace(0.2, 1.0, 500)
T_cond_e_cooling = np.linspace(15, 35, 500)
PLR_grid_cooling, T_cond_e_grid_cooling = np.meshgrid(PLR_cooling, T_cond_e_cooling)

# calculate COP for cooling
EIRFTemp_cooling = 0.38 + 0.02 * T_ev_l + 0 * T_ev_l**2 + 0.01 * T_cond_e_grid_cooling + 0 * T_cond_e_grid_cooling**2 + 0 * T_ev_l * T_cond_e_grid_cooling
EIRFPLR_cooling = 0.22 + 0.50 * PLR_grid_cooling + 0.26 * PLR_grid_cooling**2
COP_cooling = PLR_grid_cooling * COP_ref / (EIRFTemp_cooling * EIRFPLR_cooling)

# 2d grid for heating
PLR_heating = np.linspace(0.2, 1.0, 500)
T_0_heating = np.linspace(-15, 25, 500)
PLR_grid_heating, T_0_grid_heating = np.meshgrid(PLR_heating, T_0_heating)

# calculate COP for heating
COP_heating = -7.46 * (PLR_grid_heating - 0.0047 * T_0_grid_heating - 0.477) ** 2 + 0.0941 * T_0_grid_heating + 4.34

# coolwarm 컬러맵의 왼쪽 절반만 사용
coolwarm_left = mcolors.LinearSegmentedColormap.from_list(
    'coolwarm_left', get_cmap('coolwarm')(np.linspace(0, 0.45, 256))
)

# coolwarm 컬러맵의 오른쪽 절반만 사용
coolwarm_right = mcolors.LinearSegmentedColormap.from_list(
    'coolwarm_right', get_cmap('coolwarm')(np.linspace(0.55, 1.0, 256))
)

# Set up the figure and axes
fig, (ax_cooling, ax_heating) = plt.subplots(1, 2, figsize=(dm.cm2in(17), dm.cm2in(7)))
plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.18, wspace=0.25)

# color
contour_cooling_color = 'dm.white'
contour_heating_color = 'dm.white'

# normalize for colorbar
norm_COP_cooling = mcolors.Normalize(vmin=0.0, vmax=6.0)
norm_COP_heating = mcolors.Normalize(vmin=0.0, vmax=7.0)

# Cooling colormap
colormap_cooling = ax_cooling.pcolormesh(T_cond_e_grid_cooling, PLR_grid_cooling, COP_cooling,
                                         shading='auto',
                                         cmap=coolwarm_left.reversed(),
                                         norm=norm_COP_cooling,
                                         rasterized=True)
cbar_cooling = fig.colorbar(colormap_cooling, ax=ax_cooling)
cbar_cooling.set_label('COP [ - ]', fontsize=fs['cbar_label'], labelpad=pad['label'])
cbar_cooling.ax.tick_params(labelsize=fs['cbar_tick'])
cbar_cooling.set_ticks(np.arange(0.0, 6.1, 1.0))
cbar_cooling.ax.minorticks_off()

# Cooling contour lines
contour_lines_cooling = ax_cooling.contour(T_cond_e_grid_cooling, PLR_grid_cooling, COP_cooling,
                                           levels=np.arange(3.0, 5.6, 0.5), 
                                           colors=contour_cooling_color,
                                           linewidths=0.7,
                                           alpha=0.9)
ax_cooling.clabel(contour_lines_cooling, 
                  inline=True, 
                  fontsize=fs['tick'],
                  fmt='%.1f')

# Cooling graph
ax_cooling.set_xlabel('Environmental temperature [$^{\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
ax_cooling.set_ylabel('Part load ratio [ - ]', fontsize=fs['label'], labelpad=pad['label'])
ax_cooling.set_xlim(15, 35)
ax_cooling.set_ylim(0.2, 1.0)
ax_cooling.tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
ax_cooling.tick_params(axis='both', which='minor', labelsize=fs['tick'], pad=pad['tick'])
ax_cooling.set_xticks(np.arange(15, 36, 5))
ax_cooling.set_yticks(np.arange(0.2, 1.1, 0.2))
ax_cooling.text(0.02, 1.09, '(a) Cooling mode', transform=ax_cooling.transAxes,
                fontsize=fs['subtitle'], va='top', ha='left')

# Heating colormap
colormap_heating = ax_heating.pcolormesh(T_0_grid_heating, PLR_grid_heating, COP_heating,
                                         shading='auto',
                                         cmap=coolwarm_right,
                                         norm=norm_COP_heating,
                                         rasterized=True)
cbar_heating = fig.colorbar(colormap_heating, ax=ax_heating)
cbar_heating.set_label('COP [ - ]', fontsize=fs['cbar_label'], labelpad=pad['label'])
cbar_heating.ax.tick_params(labelsize=fs['cbar_tick'])
cbar_heating.set_ticks(np.arange(0.0, 7.1, 1.0))
cbar_heating.ax.minorticks_off()

# Heating contour lines
contour_lines_heating = ax_heating.contour(T_0_grid_heating, PLR_grid_heating, COP_heating,
                                           levels=np.arange(1.0, 7.1, 1.0),
                                           colors=contour_heating_color,
                                           linewidths=0.7,
                                           alpha=0.9)
ax_heating.clabel(contour_lines_heating, 
                  inline=True, 
                  fontsize=fs['tick'],
                  fmt='%.1f')

# Heating graph
ax_heating.set_xlabel('Environmental temperature [$^{\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
ax_heating.set_ylabel('Part load ratio [ - ]', fontsize=fs['label'], labelpad=pad['label'])
ax_heating.set_xlim(-10, 25)
ax_heating.set_ylim(0.2, 1.0)
ax_heating.tick_params(axis='both', which='major', labelsize=fs['tick'], pad=pad['tick'])
ax_heating.tick_params(axis='both', which='minor', labelsize=fs['tick'], pad=pad['tick'])
ax_heating.set_xticks(np.arange(-15, 26, 10))
ax_heating.set_yticks(np.arange(0.2, 1.1, 0.2))
ax_heating.text(0.01, 1.09, '(b) Heating mode', transform=ax_heating.transAxes,
                fontsize=fs['subtitle'], va='top', ha='left')

# # Save and show the figure
plt.savefig('../figure/Fig. 6.png', dpi=600)
plt.savefig('../figure/Fig. 6.svg', dpi=600, transparent=True)
dm.util.save_and_show(fig)