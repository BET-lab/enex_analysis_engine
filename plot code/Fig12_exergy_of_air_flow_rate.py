#%%
# import libraries
import enex_analysis as enex
import matplotlib.pyplot as plt
import dartwork_mpl as dm
import numpy as np
from enex_analysis.plot_style import fs, pad
from matplotlib.ticker import AutoMinorLocator 
dm.use_style()

# 원하는 마이너 틱 개수(축별)
x_minor_between = 1   # X축: 메이저 틱 사이 마이너 4개
y_minor_between = 1   # Y축: 메이저 틱 사이 마이너 2개


xmin, xmax, xint, xmar = -10, 40, 5, 0
ymin, ymax, yint, ymar =  0, 6, 1, 0

# constant
c_a = 1005 # Specific heat capacity of air [J/kgK]
rho_a = 1.225 # Density of air [kg/m³]
k_a = 0.0257 # Thermal conductivity of air [W/mK]


T_range = np.arange(xmin, xmax+xint, 0.1)
T_a_int_in = 22

T0_cooling = -6
T0_heating = 33
T0 = np.array([T0_cooling, T0_heating])

T_range_K = enex.C2K(T_range)
T0_K = enex.C2K(T0)


dV_int_unit = 1 # 환기량 [m³/s]
y1 = c_a * rho_a * dV_int_unit * ((T_range_K - T0_K[0]) - T0_K[0] * np.log(T_range_K / T0_K[0]))*enex.W2kW
y2 = c_a * rho_a * dV_int_unit * ((T_range_K - T0_K[1]) - T0_K[1] * np.log(T_range_K / T0_K[1]))*enex.W2kW

T_set = 22
T_a_int_out_cooling = 12
T_a_int_out_heating = 32

# 3) Figure 생성
fig = plt.figure(
    figsize=(dm.cm2in(14), dm.cm2in(7)),
    dpi=300
)

# 4) GridSpec 레이아웃
nrows, ncols = 1, 1
gs = fig.add_gridspec(
    nrows=nrows, ncols=ncols,
    left=0.16, right=0.98, top=0.96, bottom=0.18,
    hspace=0.10, wspace=0.10
)

# 5) 2중 for 문으로 축 생성 및 순회 (단일 플롯이어도 유지)
for row in range(nrows):
    for col in range(ncols):
        ax = fig.add_subplot(gs[row, col])

        # 6) 데이터 플로팅 (계단형 라인)
        lw = 1
        line1, = ax.plot(
            T_range, y1, 
            c= 'dm.red4', lw=lw, label=f'Exergy rate of unit air flow rate in winter ($T_0$ = - {abs(T0_cooling)} °C)',
        )
        line2, = ax.plot(
            T_range, y2,
            c= 'dm.blue3', lw=lw, label=f'Exergy rate of unit air flow rate in summer ($T_0$ = {T0_heating} °C)',
        )
        # T_set°C
        ds_lw = 0.5 # dashed line width
        c1 = 'dm.teal'
        c2 = 'dm.blue'
        c3 = 'dm.red'
        ax.axvline(x=T_set, color=c1 + '6', linestyle='--', linewidth=ds_lw, zorder=0)
        # ax.axvline(x=T_a_int_out_cooling, color=c2 + '6', linestyle='--', linewidth=ds_lw, zorder=0)
        # ax.axvline(x=T_a_int_out_heating, color=c3 + '6', linestyle='--', linewidth=ds_lw, zorder=0)

        dx = -0.3
        ax.text(T_set + dx, ymax*0.95, 'Setpoint', rotation=90, fontsize=fs['setpoint'], color=c1 + '6', ha='right', va= 'top')
        # ax.text(T_a_int_out_cooling + dx, ymax*0.95, 'Internal unit outlet air temp (cooling)', rotation=90, fontsize=text_fs, color=c2 + '8', ha='right', va= 'top')
        # ax.text(T_a_int_out_heating + dx, ymax*0.95, 'Internal unit outlet air temp (heating)', rotation=90, fontsize=text_fs, color=c3 + '8', ha= 'right', va='top')

        # 교차점 계산 및 scatter plot 추가
        # 1번 점선 (x=T_set)이 line1, line2와 만나는 점
        ss = 6 # scatter size
        sc1 = 'dm.red8'
        sc2 = 'dm.blue8'
        y1_at_T_set = np.interp(T_set, T_range, y1)
        y2_at_T_set = np.interp(T_set, T_range, y2)
        ax.scatter([T_set, T_set], [y1_at_T_set, y2_at_T_set], color=[sc1, sc2], s=ss, zorder=5) # s는 마커 크기
                # --- NEW: scatter 위에 y값 표시(소수점 한자리 '절삭') ---

        for x, y, col in [(T_set, y1_at_T_set, sc1), (T_set, y2_at_T_set, sc2)]:
            ax.annotate(
                f"{y:.2f}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 4),           # 점 바로 위로 6pt 오프셋
                ha="center",
                va="bottom",
                fontsize=fs['text']+1,    # 기존에 사용 중인 폰트 크기 변수 재사용
                color=col,               # 스캐터와 동일한 색상
                zorder=6
            )

        # 3번 점선 (x=T_a_int_out_heating)이 line1과 만나는 점
        y1_at_T_a_int_out_heating = np.interp(T_a_int_out_heating, T_range, y1)
        # ax.scatter(T_a_int_out_heating, y1_at_T_a_int_out_heating, color=sc1, s=ss, zorder=5)

        # 2번 점선 (x=T_a_int_out_cooling)이 line2와 만나는 점
        y2_at_T_a_int_out_cooling = np.interp(T_a_int_out_cooling, T_range, y2)
        # ax.scatter(T_a_int_out_cooling, y2_at_T_a_int_out_cooling, color=sc2, s=ss, zorder=5)
        
        y1_dy = y1_at_T_a_int_out_heating - y1_at_T_set
        y2_dy = y2_at_T_a_int_out_cooling - y2_at_T_set


        # 7) 축/눈금/범위 등 그래프 요소
        ax.set_xlabel('Indoor air temperature [°C]', labelpad = pad['label'], fontsize=fs['label'])
        ax.set_ylabel('Exergy rate in air flow [kW/(m$^3$/s)]', labelpad = pad['label'], fontsize=fs['label'])
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xticks(np.arange(xmin, xmax + 1, xint))
        ax.set_yticks(np.arange(ymin, ymax + 1, yint))
        
        ax.tick_params(axis='both', which='major', direction='in',  pad=pad['tick'], labelsize=fs['tick'])
        ax.tick_params(axis='both', which='minor', direction='in',  pad=pad['tick'], labelsize=fs['tick'])
        
        # 마이너 틱 로케이터 설정 (축별로 개수 조절)
        ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_between + 1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_between + 1))

        # (선택) 격자
        ax.grid(True, which='major', alpha=0.25)

        # 8) 범례
        handles = [line1, line2]
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc='upper left', fontsize=fs['legend'], frameon=False, handletextpad=1)

# 9) 레이아웃 최적화 (tight_layout 사용 금지)
dm.simple_layout(fig)
plt.savefig('../figure/Fig. 12.png', dpi=600)
plt.savefig('../figure/Fig. 12.svg', dpi=600, transparent=True)

dm.save_and_show(fig)
# %%
