"""Figure 11 – Net exergy rate available in ventilation air."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import dartwork_mpl as dm

import enex_analysis as enex

PNG_PATH = Path("figure") / "Fig. 11.png"
PDF_PATH = Path("figure") / "Fig. 11.pdf"


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9

    fs = {
        "label": dm.fs(0),
        "tick": dm.fs(-1.5),
        "subtitle": dm.fs(-0.5),
        "legend": dm.fs(-2.0),
        "setpoint": dm.fs(-1.0),
    }
    pad = {"label": 6, "tick": 4}

    xmin, xmax, xint = -10.0, 40.0, 5.0
    ymin, ymax, yint = 0.0, 6.0, 1.0

    c_a = 1005.0
    rho_a = 1.225
    T_range = np.arange(xmin, xmax + xint, 0.1)
    T_set = 22.0

    T0_cooling = -6.0
    T0_heating = 33.0
    T_range_K = enex.C2K(T_range)
    T0_cooling_K = enex.C2K(T0_cooling)
    T0_heating_K = enex.C2K(T0_heating)

    dV_int_unit = 1.0
    y1 = c_a * rho_a * dV_int_unit * ((T_range_K - T0_cooling_K) - T0_cooling_K * np.log(T_range_K / T0_cooling_K)) * enex.W2kW
    y2 = c_a * rho_a * dV_int_unit * ((T_range_K - T0_heating_K) - T0_heating_K * np.log(T_range_K / T0_heating_K)) * enex.W2kW

    fig = plt.figure(figsize=(dm.cm2in(14), dm.cm2in(7)), dpi=300)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=1,
        left=0.16,
        right=0.98,
        top=0.96,
        bottom=0.18,
        hspace=0.10,
        wspace=0.10,
    )

    ax = fig.add_subplot(gs[0, 0])

    line_width = 1.0
    line1 = ax.plot(
        T_range,
        y1,
        c="dm.red4",
        lw=line_width,
        label=f"Outdoor air temp = - {abs(T0_cooling):.0f} °C (winter)",
    )[0]
    line2 = ax.plot(
        T_range,
        y2,
        c="dm.blue3",
        lw=line_width,
        label=f"Outdoor air temp = {T0_heating:.0f} °C (summer)",
    )[0]

    ax.axvline(x=T_set, color="dm.teal6", linestyle="--", linewidth=0.6, zorder=0)
    ax.text(
        T_set - 0.3,
        ymax * 0.97,
        "Setpoint",
        rotation=90,
        fontsize=fs["setpoint"],
        color="dm.teal6",
        ha="right",
        va="top",
    )

    scatter_size = 6
    cooling_point = np.interp(T_set, T_range, y1)
    heating_point = np.interp(T_set, T_range, y2)
    ax.scatter([T_set], [cooling_point], color="dm.red8", s=scatter_size, zorder=5)
    ax.scatter([T_set], [heating_point], color="dm.blue8", s=scatter_size, zorder=5)
    ax.annotate(
        f"{cooling_point:.2f}",
        (T_set, cooling_point),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        va="bottom",
        fontsize=fs["setpoint"],
        color="dm.red8",
        zorder=6,
    )
    ax.annotate(
        f"{heating_point:.2f}",
        (T_set, heating_point),
        textcoords="offset points",
        xytext=(0, 4),
        ha="center",
        va="bottom",
        fontsize=fs["setpoint"],
        color="dm.blue8",
        zorder=6,
    )

    ax.set_xlabel("Air temperature [°C]", labelpad=pad["label"], fontsize=fs["label"])
    ax.set_ylabel("Exergy rate in air flow [kW/(m$^3$/s)]", labelpad=pad["label"], fontsize=fs["label"])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(xmin, xmax + 1, xint))
    ax.set_yticks(np.arange(ymin, ymax + 1, yint))
    ax.tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", alpha=0.25)

    legend = ax.legend(loc="upper left", fontsize=fs["legend"], frameon=False, handletextpad=1.0)
    legend.set_zorder(10)

    if hasattr(dm, "simple_layout"):
        dm.simple_layout(fig)
    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH, dpi=600, transparent=True)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
