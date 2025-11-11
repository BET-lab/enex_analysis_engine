#%%
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from data_setting import get_weekday_df
from enex_analysis.plot_style import fs, pad
from matplotlib.ticker import AutoMinorLocator
dm.use_style()

# Data
# 1) 사용할 월과 '월별 시작일로부터 7일' 부분집합 만들기
weekday_df = get_weekday_df()
months_to_plot = [1, 4, 8, 10]
start_map = {1: 9, 4: 10, 8: 9, 10: 10}  # 월별 시작일

analyze_days = 1  # 시작일로부터 며칠간

day_all = weekday_df['Date/Time_clean'].str.slice(3, 5).astype(int)  # 'MM/DD ...' → DD
month_s = weekday_df['Month']
start_day = month_s.map(start_map)  # 행별 시작일(해당 월 아닌 행은 NaN)
end_day = start_day + analyze_days

mask = (
    month_s.isin(months_to_plot) &
    day_all.ge(start_day) &
    day_all.le(end_day-1)
)
mask = mask | mask.shift(-1, fill_value=False) # 바로 전날 24:00 포함

df_sub = weekday_df.loc[mask].copy()

# 표시/축용 보조 열
df_sub['Day'] = day_all[mask].values
hour_sub = df_sub['Date/Time_clean'].str.slice(6, 8).astype(int)  # '... HH:MM:SS' → HH
df_sub['xpos'] = df_sub['Day'] + hour_sub / 24.0  # 연속 x축

# 최대 부하(정규화용) 한 번만 계산
Q_r_max_cooling = (df_sub['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h).max()
Q_r_max_heating = (df_sub['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h).max()

# 2) COP / 엑서지 계산 (부분집합에만 수행)
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

# =========================
# 0) 상단 설정: 한 곳에서 전부 수정
# =========================
# X축
X_STEP_HR         = 1
XMIN              = 0
XMAX              = 24 * analyze_days
XMAJOR_INT        = 6
XMINOR_PER_MAJOR  = 1  # 메이저 사이 마이너틱 개수

# Y축 범위/틱
TEMP_YMIN,   TEMP_YMAX,   TEMP_YTICK   = -10, 40, 10
LOAD_YMIN,   LOAD_YMAX,   LOAD_YTICK   = 0,   25, 5
COP_YMIN,    COP_YMAX,    COP_YTICK    = 0,   5,  1
EXGY_YMIN,   EXGY_YMAX,   EXGY_YTICK   = 0,   50, 10

# 색/스타일
COLOR_TEMP       = 'dm.teal6'
COLOR_LOAD_HEAT  = 'dm.red5'
COLOR_LOAD_COOL  = 'dm.blue5'
COLOR_COP        = 'dm.orange4'
COLOR_EXERGY     = 'dm.violet6'
COLOR_SETPOINT   = 'dm.teal6'

LINEWIDTH_MAIN   = 0.7
LINEWIDTH_SPINE  = 0.6
ALPHA_FILL_LOAD  = 0.30
ALPHA_SETPOINT   = 0.70
SETPOINT_VALUE   = 22

# y라벨 고정 좌표 (좌/우 공용)
YLABEL_X_LEFT, YLABEL_X_RIGHT, YLABEL_Y = -0.13, 1.13, 0.5

# 월 표기/레이아웃
MONTH_NAME = ['January 9th', 'August 9th']
LAYOUT     = [1, 8]               # 좌: 1월, 우: 8월
HEATING_MONTHS = (1, 10)            # 난방 기준 월

# Figure/그리드/여백
FIG_W_CM, FIG_H_CM = 17, 10
OUTER_WSPACE, OUTER_HSPACE = 0.55, 0.3
INNER_HSPACE = 0.25
MARGINS = dict(left=0.08, right=0.92, top=0.94, bottom=0.1)

# =========================
# 1) 데이터 좌표 생성
# =========================
x_hour = np.arange(0, XMAX * 1.001, X_STEP_HR)

# =========================
# 2) Figure / GridSpec
# =========================
fig = plt.figure(figsize=(dm.cm2in(FIG_W_CM), dm.cm2in(FIG_H_CM)))
outer = gridspec.GridSpec(1, 2, figure=fig, wspace=OUTER_WSPACE, hspace=OUTER_HSPACE)


# =========================
# 3) 월 루프 (상/하 패널)
# =========================
for c in range(2):
    month = LAYOUT[c]
    mdf   = df_sub[df_sub['Month'] == month].sort_values('xpos')
    start = start_map[month]
    end   = start + 1

    # 월 내부 2행(상단/하단)
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[c], hspace=INNER_HSPACE)
    ax_top = fig.add_subplot(inner[0, 0])
    ax_bot = fig.add_subplot(inner[1, 0], sharex=ax_top)

    # -------------------------
    # 상단: 외기온(좌) + 부하(우)
    # -------------------------
    ax1 = ax_top
    ax2 = ax1.twinx()

    # 부하 선택(난방/냉방)
    if month in HEATING_MONTHS:
        load_series = mdf['DistrictHeatingWater:Facility [J](TimeStep) '] * enex.s2h * enex.W2kW
        load_label  = 'Heating load [kW]'
        color_load  = COLOR_LOAD_HEAT
    else:
        load_series = mdf['DistrictCooling:Facility [J](TimeStep)'] * enex.s2h * enex.W2kW
        load_label  = 'Cooling load [kW]'
        color_load  = COLOR_LOAD_COOL

    # z-order: 부하(면) 아래, 온도(선) 위
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    # 부하(우) 영역
    ax2.fill_between(x_hour, 0, load_series, color=color_load, alpha=ALPHA_FILL_LOAD, zorder=1)
    ax2.set_ylabel(load_label, fontsize=fs['label'])

    # 외기온(좌) 선
    ax1.plot(x_hour,
                mdf['Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)'],
                linewidth=LINEWIDTH_MAIN, color=COLOR_TEMP, zorder=3, alpha=1.0)

    # Setpoint
    ax1.axhline(y=SETPOINT_VALUE, color=COLOR_SETPOINT, linestyle='--',
                linewidth=LINEWIDTH_MAIN, alpha=ALPHA_SETPOINT)
    ax1.text(XMIN + 1, SETPOINT_VALUE - (TEMP_YMAX-TEMP_YMIN)/50, 'Setpoint',
                fontsize=fs['setpoint'], color=COLOR_SETPOINT, ha='left', va='top')

    # 축 범위/틱
    ax1.set_xlim(XMIN, XMAX)
    ax1.set_xticks(np.arange(XMIN, XMAX + 1, XMAJOR_INT))
    ax1.set_ylim(TEMP_YMIN, TEMP_YMAX)
    ax1.set_yticks(np.arange(TEMP_YMIN, TEMP_YMAX + 1e-9, TEMP_YTICK))

    ax2.set_ylim(LOAD_YMIN, LOAD_YMAX)
    ax2.set_yticks(np.arange(LOAD_YMIN, LOAD_YMAX + 1e-9, LOAD_YTICK))

    ax1.tick_params(axis='x', which='both', labelsize=fs['tick'], pad=pad['tick'])

    # 좌/우 스파인/색상/틱 색 정리
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax1.spines['left'].set_color(COLOR_TEMP)
    ax1.yaxis.label.set_color(COLOR_TEMP)
    ax1.tick_params(axis='y', which='both', colors=COLOR_TEMP)
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax2.spines['right'].set_color(color_load)
    ax2.yaxis.label.set_color(color_load)
    ax2.tick_params(axis='y', which='both', colors=color_load)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 라벨 위치(좌/우 동일 고정)
    ax1.yaxis.set_label_coords(YLABEL_X_LEFT,  YLABEL_Y)
    ax2.yaxis.set_label_coords(YLABEL_X_RIGHT, YLABEL_Y)

    # 틱 폰트/간격 + 스파인 두께
    ax1.tick_params(axis='y', which='major', labelsize=fs['tick'], pad=pad['tick'])
    ax2.tick_params(axis='y', which='major', labelsize=fs['tick'], pad=pad['tick'])
    ax1.spines['left'].set_linewidth(LINEWIDTH_SPINE)
    ax2.spines['right'].set_linewidth(LINEWIDTH_SPINE)

    # -------------------------
    # 하단: COP(좌) + 엑서지효율(우)
    # -------------------------
    ax3 = ax_bot
    ax4 = ax3.twinx()

    ax4.set_zorder(ax3.get_zorder() + 1)
    ax3.patch.set_visible(False)

    # 라벨/제목
    ax1.set_ylabel('Environmental temp. [$^{\\circ}$C]', fontsize=fs['label'])
    # (a)만 볼드 + 월 이름은 보통체, 좌상단 배치
    ax1.text(0.01, 1.02, '(' + chr(97+c) + ') ', transform=ax1.transAxes,
                fontsize=fs['subtitle'], fontweight='bold', va='bottom', ha='left')
    ax1.text(0.09, 1.02, MONTH_NAME[c], transform=ax1.transAxes,
            fontsize=fs['subtitle'], va='bottom', ha='left')

    # 선 그리기
    ax3.plot(x_hour, mdf['COP'],        linewidth=LINEWIDTH_MAIN, color=COLOR_COP,    zorder=2)
    ax4.plot(x_hour, mdf['ExergyEff'],  linewidth=LINEWIDTH_MAIN, color=COLOR_EXERGY, zorder=4, linestyle='--')

    # 범위/틱 (하단 x축은 해당 월의 하루 구간)
    ax3.set_xlim(start, end)
    ax3.set_xticks(np.arange(XMIN, XMAX + 1, XMAJOR_INT))
    ax3.xaxis.set_minor_locator(AutoMinorLocator(XMINOR_PER_MAJOR))

    ax3.set_ylim(COP_YMIN, COP_YMAX)
    ax3.set_yticks(np.arange(COP_YMIN, COP_YMAX + 1e-9, COP_YTICK))
    ax4.set_ylim(EXGY_YMIN, EXGY_YMAX)
    ax4.set_yticks(np.arange(EXGY_YMIN, EXGY_YMAX + 1e-9, EXGY_YTICK))

    ax3.set_xlabel('Hour of day [hour]', fontsize=fs['label'])
    ax3.set_ylabel('Energy efficiency [ - ]', fontsize=fs['label'])
    ax4.set_ylabel('Exergy efficiency [%]', fontsize=fs['label'])
    
    # 틱 폰트/간격
    ax3.tick_params(axis='x', which='both', labelsize=fs['tick'], pad=pad['tick'])
    ax3.tick_params(axis='y', which='both', colors=COLOR_COP, labelsize=fs['tick'], pad=pad['tick'])
    ax4.tick_params(axis='x', which='both', labelsize=fs['tick'], pad=pad['tick'])
    ax4.tick_params(axis='y', which='both', colors=COLOR_EXERGY, labelsize=fs['tick'], pad=pad['tick'])

    # 좌/우 스파인/색상/틱 색 정리
    ax3.spines['right'].set_visible(False)
    ax4.spines['left'].set_visible(False)

    ax3.spines['left'].set_color(COLOR_COP)
    ax3.yaxis.label.set_color(COLOR_COP)
    ax3.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax4.spines['right'].set_color(COLOR_EXERGY)
    ax4.yaxis.label.set_color(COLOR_EXERGY)
    ax4.yaxis.set_minor_locator(AutoMinorLocator(2))

    # 라벨 위치 고정
    ax3.yaxis.set_label_coords(YLABEL_X_LEFT,  YLABEL_Y)
    ax4.yaxis.set_label_coords(YLABEL_X_RIGHT, YLABEL_Y)

    # 틱 폰트/간격 + 스파인 두께
    ax3.tick_params(axis='y', which='major', labelsize=fs['tick'], pad=pad['tick'])
    ax4.tick_params(axis='y', which='major', labelsize=fs['tick'], pad=pad['tick'])
    ax3.spines['left'].set_linewidth(LINEWIDTH_SPINE)
    ax4.spines['right'].set_linewidth(LINEWIDTH_SPINE)
plt.subplots_adjust(**MARGINS)
# =========================
# 4) 저장/표시
# =========================
plt.savefig('../figure/Fig. 13.png', dpi=600)
plt.savefig('../figure/Fig. 13.pdf', dpi=600)
dm.util.save_and_show(fig)
# %%
