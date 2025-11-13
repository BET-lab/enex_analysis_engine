#%%
# Graphical Abstract - Exergy Efficiency & COP (Jan, one-column, twin axis)
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
import pandas as pd
from data_setting import get_weekday_df
from enex_analysis.plot_style import fs, pad
from matplotlib.ticker import AutoMinorLocator
dm.use_style()

# Data
weekday_df = get_weekday_df()
month_to_show = 1
start_day_jan = 9
analyze_days = 1

day_all = weekday_df['Date/Time_clean'].str.slice(3, 5).astype(int)
month_s = weekday_df['Month']

mask = (
    (month_s == month_to_show) &
    (day_all >= start_day_jan) &
    (day_all < start_day_jan + analyze_days)
)
mask = mask | mask.shift(-1, fill_value=False)

df_sub = weekday_df.loc[mask].copy()
df_sub['Day'] = day_all[mask].values
hour_sub = df_sub['Date/Time_clean'].str.slice(6, 8).astype(int)
df_sub['xpos'] = df_sub['Day'] + hour_sub / 24.0

Q_r_max_cooling = (df_sub['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h).max()
Q_r_max_heating = (df_sub['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h).max()

def compute_metrics(row):
    Toa = row['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
    cooling_load = row['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
    heating_load = row['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h

    exergy_effi = 0.0
    cop = 0.0

    if cooling_load > 0:
        ASHP = enex.AirSourceHeatPump_cooling()
        ASHP.T0 = Toa
        ASHP.T_a_room = 22
        ASHP.Q_r_int = cooling_load
        ASHP.Q_r_max = Q_r_max_cooling
        ASHP.system_update()
        if ASHP.X_eff > 0:
            exergy_effi = ASHP.X_eff * 100.0
            cop = ASHP.COP_sys

    elif heating_load > 0:
        ASHP = enex.AirSourceHeatPump_heating()
        ASHP.T0 = Toa
        ASHP.T_a_room = 22
        ASHP.Q_r_int = heating_load
        ASHP.Q_r_max = Q_r_max_heating
        ASHP.system_update()
        if ASHP.X_eff > 0:
            exergy_effi = ASHP.X_eff * 100.0
            cop = ASHP.COP_sys

    return pd.Series({'COP': cop, 'ExergyEff': exergy_effi})

df_sub[['COP', 'ExergyEff']] = df_sub.apply(compute_metrics, axis=1)

# Settings
DAY_OF_HOUR = 24
XMIN = 0 * DAY_OF_HOUR
XMAX = 1 * DAY_OF_HOUR
XMAJOR_INT = 4
EXGY_YMIN, EXGY_YMAX, EXGY_YTICK = 0, 50, 10
COP_YMIN, COP_YMAX, COP_YTICK = 0, 5, 1

COLOR_EXERGY = 'dm.green5'
COLOR_COP = 'dm.orange6'
LINEWIDTH_MAIN = 2.0
LINEWIDTH_SPINE = 1.5

fs_ga = {
    'label': fs['label'],
    'tick': fs['tick'],
    'legend': fs['legend'],
}

MONTH_NAME = 'January 9th'
FIG_W_CM, FIG_H_CM = 8.5, 5.5


# x-axis: 시간축 (24시간)
x_hour = np.arange(0, DAY_OF_HOUR + 1e-4, 1) 
mdf = df_sub.sort_values('xpos')

# Plot
fig, ax1 = plt.subplots(1, 1, figsize=(dm.cm2in(FIG_W_CM), dm.cm2in(FIG_H_CM)))
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15)

# Left y-axis: COP
line_cop, = ax1.plot(
    x_hour, mdf['COP'].values, 
    color=COLOR_COP, linewidth=LINEWIDTH_MAIN, marker='o', markersize=4,
    label='COP', zorder=2
)
ax1.set_xlim(XMIN, XMAX)
ax1.set_xticks(np.arange(XMIN, XMAX*1.0001, XMAJOR_INT))
ax1.set_xlabel('Hour of day [hour]', fontsize=fs_ga['label'])
ax1.set_ylim(COP_YMIN, COP_YMAX)
# ylabel and y ticks removed
ax1.tick_params(axis='x', which='major', labelsize=fs_ga['tick'], pad=pad['tick'])
# y 축의 메이저, 마이너 틱 길이 0으로 설정
ax1.tick_params(axis='y', which='major', length=0)
ax1.tick_params(axis='y', which='minor', length=0)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_visible(False)

# Right y-axis: Exergy Efficiency
ax2 = ax1.twinx()
line_ex, = ax2.plot(
    x_hour, mdf['ExergyEff'].values, 
    color=COLOR_EXERGY, linewidth=LINEWIDTH_MAIN, marker='o', markersize=4,
    linestyle='--', label='Exergy efficiency', zorder=2
)
ax2.set_ylim(EXGY_YMIN, EXGY_YMAX)
# ylabel and y ticks removed
ax2.tick_params(axis='y', which='major', length=0)
ax2.tick_params(axis='y', which='minor', length=0)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(True)
ax2.spines['top'].set_visible(False)

ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
ax1.yaxis.set_minor_locator(AutoMinorLocator(1))
ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
ax1.set_yticklabels([], fontsize=fs_ga['tick'])
ax2.set_yticklabels([], fontsize=fs_ga['tick'])

# Grid
# ax1.grid(True, linestyle='--', alpha=0.3, linewidth=0.8, zorder=1)

# Legend - combine both, 두 컬럼으로 (ncols=2)
lines = [line_cop, line_ex]
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

# Save & show
plt.savefig('../figure/GraphicalAbstract_exergy_efficiency_Jan.png', dpi=600)
plt.savefig('../figure/GraphicalAbstract_exergy_efficiency_Jan.svg', dpi=600, transparent=True)

dm.save_and_show(fig)

# %%
