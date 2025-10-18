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
# Colormap 설정
CMAP_BASE_NAME       = 'coolwarm'
CMAP_LEFT_RANGE      = (0.00, 0.45)   # 냉방 패널용 구간
CMAP_RIGHT_RANGE     = (0.55, 1.00)   # 난방 패널용 구간
REVERSE_LEFT_CMAP    = True           # 냉방 colormap 반전 여부

# 격자/범위
BIN_TEMP             = 2.5            # x격자(외기온, °C)
BIN_LOAD             = 5.0            # y격자(부하, kW)
X_RANGE              = (-10, 35)      # x축 범위(°C)
Y_RANGE              = (0, 30)        # y축 범위(kW)

# 패널별 컬러바/스케일 (엑서지 효율, %)
COOL_VMIN, COOL_VMAX, COOL_VTICK = 0, 20, 5
HEAT_VMIN, HEAT_VMAX, HEAT_VTICK = 0, 32, 8

# 격자 테두리/그리드/틱
GRID_RECT_LW         = 1.0
EDGE_COLOR_COOL      = 'white'
EDGE_COLOR_HEAT      = 'dm.white'
MAJOR_GRID_LS        = '--'
MAJOR_GRID_ALPHA     = 0.5
MINOR_X_LEN          = 1.6
MINOR_X_COLOR        = 'dm.gray7'

# 텍스트 색 판단용 밝기 컷오프(0=검정, 1=흰색) — 필요시 0.50~0.60 조정
LUMINANCE_CUTOFF     = 0.5

# Setpoint
SETPOINT_VALUE       = 22
SETPOINT_LS          = '--'
SETPOINT_LW          = 0.6
SETPOINT_COLOR       = 'dm.teal6'
SETPOINT_TEXT        = 'Setpoint'
SETPOINT_TX_X        = 20.8
SETPOINT_TX_Y        = 26.2
SETPOINT_TX_ROT      = 90

# Figure/레이아웃
FIG_W_CM, FIG_H_CM   = 17, 13
MARGINS              = dict(left=0.07, right=0.89, top=0.97, bottom=0.08)
HSPACE               = 0.3

CBAR_W               = 0.018
CBAR_OFF_H           = 0.025  # 축 bbox 오른쪽으로의 거리(정규 좌표)

# 패널 라벨
PANEL_A              = '(a)'
PANEL_B              = '(b)'

# =========================
# 1) Colormap 준비
# =========================
_coolwarm = cm.get_cmap(CMAP_BASE_NAME)
coolwarm_left  = mcolors.LinearSegmentedColormap.from_list(
    'coolwarm_left',  _coolwarm(np.linspace(*CMAP_LEFT_RANGE, 256))
)
coolwarm_right = mcolors.LinearSegmentedColormap.from_list(
    'coolwarm_right', _coolwarm(np.linspace(*CMAP_RIGHT_RANGE, 256))
)
CMAP_COOL = coolwarm_left.reversed() if REVERSE_LEFT_CMAP else coolwarm_left
CMAP_HEAT = coolwarm_right

# =========================
# 2) 데이터 & 빈
# =========================
# Cooling
Toa_c   = np.array(Toa_cooling_list_filtered)
Load_c  = np.array(cooling_load_list_filtered) / 1000.0
X_eff_c = np.array(ASHP_cooling_exergy_effi_filtered) * 100.0  # [%]

# Heating
Toa_h   = np.array(Toa_heating_list_filtered)
Load_h  = np.array(heating_load_list_filtered) / 1000.0
X_eff_h = np.array(ASHP_heating_exergy_effi_filtered) * 100.0  # [%]

bins_x = np.arange(X_RANGE[0], X_RANGE[1] + BIN_TEMP, BIN_TEMP)
bins_y = np.arange(Y_RANGE[0], Y_RANGE[1] + BIN_LOAD, BIN_LOAD)

# 평균 엑서지 효율(%) 격자 계산 (grid_stats는 기존 정의 사용)
avg_c, xedges, yedges = grid_stats(Toa_c, Load_c, X_eff_c, bins_x, bins_y)
avg_h, _,      _      = grid_stats(Toa_h, Load_h, X_eff_h, bins_x, bins_y)

# =========================
# 3) 플롯
# =========================
fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(FIG_W_CM), dm.cm2in(FIG_H_CM)), sharex=False, sharey=True)
plt.subplots_adjust(**MARGINS, hspace=HSPACE)

# 패널 파라미터 목록(루프로 그리기)
panels = [
    dict(ax_idx=0, avg=avg_c, cmap=CMAP_COOL, vmin=COOL_VMIN, vmax=COOL_VMAX,
         y_label='Cooling load [kW]', edge_color=EDGE_COLOR_COOL,
         cbar_ticks=np.arange(COOL_VMIN, COOL_VMAX + 1e-9, COOL_VTICK),
         panel_tag=PANEL_A),
    dict(ax_idx=1, avg=avg_h, cmap=CMAP_HEAT, vmin=HEAT_VMIN, vmax=HEAT_VMAX,
         y_label='Heating load [kW]', edge_color=EDGE_COLOR_HEAT,
         cbar_ticks=np.arange(HEAT_VMIN, HEAT_VMAX + 1e-9, HEAT_VTICK),
         panel_tag=PANEL_B),
]

for p in panels:
    ax = axes[p['ax_idx']]
    im = ax.pcolormesh(xedges, yedges, p['avg'], cmap=p['cmap'], vmin=p['vmin'], vmax=p['vmax'])

    # ★ 패널별 정규화/컬러맵 (텍스트 색 결정에 사용)
    _norm = mcolors.Normalize(vmin=p['vmin'], vmax=p['vmax'])
    _cmap_for_text = p['cmap']  # 이미 Colormap 객체

    # 격자 테두리 + 셀 값 표기
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            ax.add_patch(plt.Rectangle(
                (xedges[i], yedges[j]),
                xedges[i+1] - xedges[i],
                yedges[j+1] - yedges[j],
                linewidth=GRID_RECT_LW,
                edgecolor=p['edge_color'],
                facecolor='none',
                zorder=1
            ))
            val = p['avg'][j, i]
            if not np.isnan(val):
                # ★ 배경색의 상대 휘도(luminance)로 텍스트 색 자동 선택
                r, g, b, _ = _cmap_for_text(_norm(val))
                luminance = 0.2126*r + 0.7152*g + 0.0722*b
                txt_color = 'white' if luminance < LUMINANCE_CUTOFF else 'black'

                ax.text(
                    (xedges[i] + xedges[i+1]) / 2.0,
                    (yedges[j] + yedges[j+1]) / 2.0,
                    f"{val:.1f}", ha='center', va='center',
                    fontsize=fs['text'], color=txt_color
                )

    # 축 라벨/마이너틱
    ax.set_ylabel(p['y_label'], fontsize=fs['label'], labelpad=pad['label'])
    ax.minorticks_off()

    # Setpoint 라인/텍스트
    ax.axvline(x=SETPOINT_VALUE, color=SETPOINT_COLOR, linestyle=SETPOINT_LS, linewidth=SETPOINT_LW)
    ax.text(SETPOINT_TX_X, SETPOINT_TX_Y, SETPOINT_TEXT, rotation=SETPOINT_TX_ROT,
            fontsize=fs['setpoint'], color=SETPOINT_COLOR, ha='left', va='center')

    # 패널 태그
    ax.text(0.01, 0.97, p['panel_tag'], transform=ax.transAxes,
            fontsize=fs['subtitle'], fontweight='bold', va='top', ha='left')
    
    
    # 컬러바(축 바깥 개별 배치)
    bbox = ax.get_position()
    cb_ax = fig.add_axes([bbox.x1 + CBAR_OFF_H, bbox.y0, CBAR_W, bbox.y1 - bbox.y0])
    cbar  = fig.colorbar(im, cax=cb_ax, orientation='vertical')
    cbar.ax.tick_params(direction='in', labelsize=fs['cbar_tick'], pad=pad['tick'])
    cbar.set_ticks(np.arange(COOL_VMIN, COOL_VMAX + 1e-9, COOL_VTICK) if p['ax_idx'] == 0 else np.arange(HEAT_VMIN, HEAT_VMAX + 1e-9, HEAT_VTICK))
    cbar.ax.minorticks_off()
    cbar.ax.set_ylabel('Exergy efficiency ($\eta_{X,sys}$) [ - ]',
                       rotation=90, fontsize=fs['cbar_label'], labelpad=pad['label'], loc='center')

# 공통 축/그리드/틱
for ax in axes:
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    ax.set_xlabel('Environmental temperature [$^{\\circ}$C]', fontsize=fs['label'], labelpad=pad['label'])
    ax.grid(True, linestyle=MAJOR_GRID_LS, alpha=MAJOR_GRID_ALPHA)
    ax.set_xticks(xedges, minor=True)
    ax.tick_params(axis='x', which='minor', length=MINOR_X_LEN, color=MINOR_X_COLOR)
    ax.tick_params(axis='both', which='major', labelsize=fs['tick'])
    


# 저장/표시 (파일명은 원 코드에 맞춰 Fig. 10)
plt.savefig('../figure/Fig. 10.png', dpi=600)
plt.savefig('../figure/Fig. 10.pdf', dpi=600)
dm.util.save_and_show(fig)