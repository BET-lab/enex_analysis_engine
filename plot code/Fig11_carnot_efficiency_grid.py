#%%
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm
from data_setting import get_weekday_df
from enex_analysis.plot_style import fs, pad
dm.use_style()

def grid_stats(Toa, Load, COP, bins_x, bins_y):
    avg_COP = np.full((len(bins_y)-1, len(bins_x)-1), np.nan)
    for i in range(len(bins_x)-1):  # x축 (Toa)
        for j in range(len(bins_y)-1):  # y축 (Load)
            mask = (
                (Toa >= bins_x[i]) & (Toa < bins_x[i+1]) &
                (Load >= bins_y[j]) & (Load < bins_y[j+1])
            )
            if np.any(mask):
                avg_COP[j,i] = np.mean(COP[mask])
    return avg_COP, bins_x, bins_y

# Data
df = get_weekday_df()
date_list = df['Date/Time_clean']
Toa_list = df['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)']
cooling_load_list = df['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h
heating_load_list = df['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h

ASHP_cooling_exergy_effi = []
ASHP_heating_exergy_effi = []
ASHP_cooling_COP = []
ASHP_heating_COP = []

for Toa, cooling_load, heating_load in zip(Toa_list, cooling_load_list, heating_load_list):
    # 냉방 엑서지 효율 계산
    if cooling_load > 0:
        ASHP_cooling = enex.AirSourceHeatPump_cooling()
        ASHP_cooling.T0 = Toa
        ASHP_cooling.T_a_room = 22
        ASHP_cooling.Q_r_int = cooling_load
        ASHP_cooling.Q_r_max = max(cooling_load_list)
        ASHP_cooling.system_update()
        if ASHP_cooling.X_eff < 0:
            ASHP_cooling_exergy_effi.append(None)
            ASHP_cooling_COP.append(None)
        else:          
            ASHP_cooling_exergy_effi.append(ASHP_cooling.X_eff)
            ASHP_cooling_COP.append(ASHP_cooling.COP_sys)
    else:
        ASHP_cooling_exergy_effi.append(None)
        ASHP_cooling_COP.append(None)

    # 난방 엑서지 효율 계산
    if heating_load > 0:
        ASHP_heating = enex.AirSourceHeatPump_heating()
        ASHP_heating.T0 = Toa
        ASHP_heating.T_a_room = 22
        ASHP_heating.Q_r_int = heating_load
        ASHP_heating.Q_r_max = max(heating_load_list)
        ASHP_heating.system_update() 
        if ASHP_heating.X_eff < 0:
            ASHP_heating_exergy_effi.append(None)
            ASHP_heating_COP.append(None)
        else:
            ASHP_heating_exergy_effi.append(ASHP_heating.X_eff)
            ASHP_heating_COP.append(ASHP_heating.COP_sys)
    else:
        ASHP_heating_exergy_effi.append(None)
        ASHP_heating_COP.append(None)

# COP None 값일 때 해당하는 COP를 제거한 필터링된 리스트
ASHP_cooling_COP_filtered = [cop for cop in ASHP_cooling_COP if cop is not None]
ASHP_heating_COP_filtered = [cop for cop in ASHP_heating_COP if cop is not None]
ASHP_cooling_exergy_effi_filtered = [eff for eff in ASHP_cooling_exergy_effi if eff is not None]
ASHP_heating_exergy_effi_filtered = [eff for eff in ASHP_heating_exergy_effi if eff is not None]

# 엑서지효율 None 값일 때 해당하는 온도, 부하를 제거한 필터링된 리스트
Date_cooling_list_filtered = [date for date, eff in zip(date_list, ASHP_cooling_exergy_effi) if eff is not None]
Date_heating_list_filtered = [date for date, eff in zip(date_list, ASHP_heating_exergy_effi) if eff is not None]
Toa_cooling_list_filtered = [Toa for Toa, eff in zip(Toa_list, ASHP_cooling_exergy_effi) if eff is not None]
Toa_heating_list_filtered = [Toa for Toa, eff in zip(Toa_list, ASHP_heating_exergy_effi) if eff is not None]
cooling_load_list_filtered = [load for load, eff in zip(cooling_load_list, ASHP_cooling_exergy_effi) if eff is not None]
heating_load_list_filtered = [load for load, eff in zip(heating_load_list, ASHP_heating_exergy_effi) if eff is not None]

# =========================
# 0) 한 곳에서 모두 수정 (튜닝 파라미터)
# =========================
CMAP_NAME            = 'coolwarm'
C_MIN, C_MAX, C_INT  = -0.1, 0.1, 0.02
BIN_TEMP             = 2.5
BIN_LOAD             = 5.0
X_RANGE              = (-10, 35)
Y_RANGE              = (0, 30)

GRID_RECT_LW         = 1.0
EDGE_COLOR_COOL      = 'white'
EDGE_COLOR_HEAT      = 'dm.white'
MAJOR_GRID_LS        = '--'
MAJOR_GRID_ALPHA     = 0.5
MINOR_X_LEN          = 1.6
MINOR_X_COLOR        = 'dm.gray7'

# 텍스트 색 판정: 컬러맵 밝기 기준 (0=검정,1=흰색). 0.5~0.6 사이가 무난.
LUMINANCE_CUTOFF     = 0.5

SETPOINT_VALUE       = 22
SETPOINT_LS          = '--'
SETPOINT_LW          = 0.6
SETPOINT_COLOR       = 'dm.teal6'
SETPOINT_TEXT        = 'Setpoint'
SETPOINT_TX_X        = 20.8
SETPOINT_TX_Y        = 26.2
SETPOINT_TX_ROT      = 90

FIG_W_CM, FIG_H_CM   = 17, 13
MARGINS              = dict(left=0.07, right=0.87, top=0.97, bottom=0.08)
HSPACE               = 0.3

CBAR_W               = 0.018
CBAR_OFF_H           = 0.03

PANEL_A              = '(a)'
PANEL_B              = '(b)'

# =========================
# 1) 데이터 & 빈
# =========================
Toa_c   = np.array(Toa_cooling_list_filtered)
Load_c  = np.array(cooling_load_list_filtered) / 1000.0
Carnot_eff_c = 1 - enex.C2K(np.array(Toa_cooling_list_filtered)) / enex.C2K(SETPOINT_VALUE)

Toa_h   = np.array(Toa_heating_list_filtered)
Load_h  = np.array(heating_load_list_filtered) / 1000.0
Carnot_eff_h = 1 - enex.C2K(np.array(Toa_heating_list_filtered)) / enex.C2K(SETPOINT_VALUE)

bins_x = np.arange(X_RANGE[0], X_RANGE[1] + BIN_TEMP, BIN_TEMP)
bins_y = np.arange(Y_RANGE[0], Y_RANGE[1] + BIN_LOAD, BIN_LOAD)

avg_c, xedges, yedges = grid_stats(Toa_c, Load_c, Carnot_eff_c, bins_x, bins_y)
avg_h, _,      _      = grid_stats(Toa_h, Load_h, Carnot_eff_h, bins_x, bins_y)

# =========================
# 2) 플롯
# =========================
fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(FIG_W_CM), dm.cm2in(FIG_H_CM)), sharex=False, sharey=True)
plt.subplots_adjust(**MARGINS, hspace=HSPACE)

# ★ 컬러맵/정규화: 텍스트 색 결정에 사용
_norm = mcolors.Normalize(vmin=C_MIN, vmax=C_MAX)
_cmap = cm.get_cmap(CMAP_NAME)

# --- (a) Cooling ---
im1 = axes[0].pcolormesh(xedges, yedges, avg_c, cmap=CMAP_NAME, vmin=C_MIN, vmax=C_MAX)
for i in range(len(xedges) - 1):
    for j in range(len(yedges) - 1):
        axes[0].add_patch(plt.Rectangle(
            (xedges[i], yedges[j]),
            xedges[i+1] - xedges[i],
            yedges[j+1] - yedges[j],
            linewidth=GRID_RECT_LW,
            edgecolor=EDGE_COLOR_COOL,
            facecolor='none',
            zorder=1
        ))
        val = avg_c[j, i]
        if not np.isnan(val):
            # ★ 배경색의 상대 휘도 기반으로 텍스트 색 자동 결정
            r, g, b, _ = _cmap(_norm(val))
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            txt_color = 'white' if luminance < LUMINANCE_CUTOFF else 'black'

            axes[0].text(
                (xedges[i] + xedges[i+1]) / 2.0,
                (yedges[j] + yedges[j+1]) / 2.0,
                f"{val:.2f}", ha='center', va='center',
                fontsize=fs['text'], color=txt_color
            )
axes[0].set_ylabel('Cooling load [kW]', fontsize=fs['label'], labelpad=pad['label'])
axes[0].tick_params(axis='both', which='major', labelsize=fs['tick'])
axes[0].minorticks_off()
axes[0].axvline(x=SETPOINT_VALUE, color=SETPOINT_COLOR, linestyle=SETPOINT_LS, linewidth=SETPOINT_LW)
axes[0].text(SETPOINT_TX_X, SETPOINT_TX_Y, SETPOINT_TEXT, rotation=SETPOINT_TX_ROT,
             fontsize=fs['setpoint'], color=SETPOINT_COLOR, ha='left', va='center')
axes[0].text(0.01, 0.97, PANEL_A, transform=axes[0].transAxes,
             fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')

# --- (b) Heating ---
im2 = axes[1].pcolormesh(xedges, yedges, avg_h, cmap=CMAP_NAME, vmin=C_MIN, vmax=C_MAX)
for i in range(len(xedges) - 1):
    for j in range(len(yedges) - 1):
        axes[1].add_patch(plt.Rectangle(
            (xedges[i], yedges[j]),
            xedges[i+1] - xedges[i],
            yedges[j+1] - yedges[j],
            linewidth=GRID_RECT_LW,
            edgecolor=EDGE_COLOR_HEAT,
            facecolor='none',
            zorder=1
        ))
        val = avg_h[j, i]
        if not np.isnan(val):
            # ★ 동일 로직 적용
            r, g, b, _ = _cmap(_norm(val))
            luminance = 0.2126*r + 0.7152*g + 0.0722*b
            txt_color = 'white' if luminance < LUMINANCE_CUTOFF else 'black'

            axes[1].text(
                (xedges[i] + xedges[i+1]) / 2.0,
                (yedges[j] + yedges[j+1]) / 2.0,
                f"{val:.2f}", ha='center', va='center',
                fontsize=fs['text'], color=txt_color
            )
axes[1].set_ylabel('Heating load [kW]', fontsize=fs['label'], labelpad=pad['label'])
axes[1].tick_params(axis='both', which='major', labelsize=fs['tick'])
axes[1].minorticks_off()
axes[1].axvline(x=SETPOINT_VALUE, color=SETPOINT_COLOR, linestyle=SETPOINT_LS, linewidth=SETPOINT_LW)
axes[1].text(SETPOINT_TX_X, SETPOINT_TX_Y, SETPOINT_TEXT, rotation=SETPOINT_TX_ROT,
             fontsize=fs['setpoint'], color=SETPOINT_COLOR, ha='left', va='center')
axes[1].text(0.01, 0.97, PANEL_B, transform=axes[1].transAxes,
             fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')

# --- 공통 축/그리드/틱 ---
for ax in axes:
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_xlabel('Environmental temperature [$^{\\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
    ax.grid(True, linestyle=MAJOR_GRID_LS, alpha=MAJOR_GRID_ALPHA)
    ax.set_xticks(xedges, minor=True)
    ax.tick_params(axis='x', which='minor', length=MINOR_X_LEN, color=MINOR_X_COLOR, pad=pad['tick'])

# --- 공용 컬러바 ---
bbox_top  = axes[0].get_position()
bbox_bot  = axes[1].get_position()
cb_ax = fig.add_axes([bbox_bot.x1 + CBAR_OFF_H, bbox_bot.y0, CBAR_W, bbox_top.y1 - bbox_bot.y0])
cbar = fig.colorbar(im1, cax=cb_ax, orientation='vertical')
cbar.ax.tick_params(direction='in', labelsize=fs['cbar_tick'], pad=pad['tick'])
cbar.set_ticks(np.arange(C_MIN, C_MAX + 1e-12, C_INT))
cbar.ax.minorticks_off()
cbar.ax.set_ylabel('Carnot efficiency ($1 - T_0/T_{set}$) [ - ]',
                   rotation=90, fontsize=fs['cbar_label'], labelpad=pad['label'], loc='center')


plt.savefig('../figure/Fig. 11.png', dpi=600)
plt.savefig('../figure/Fig. 11.pdf', dpi=600)
dm.util.save_and_show(fig)
