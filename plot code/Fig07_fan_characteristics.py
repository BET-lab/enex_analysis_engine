"""Figure 7 – Fan performance characteristics for indoor and outdoor units."""
from __future__ import annotations

from pathlib import Path

import dartwork_mpl as dm
import matplotlib.pyplot as plt
import numpy as np

import enex_analysis as enex

PNG_PATH = Path("figure") / "Fig. 7.png"
PDF_PATH = Path("figure") / "Fig. 7.pdf"


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9

    fs = {
        "label": dm.fs(0),
        "tick": dm.fs(-1.5),
        "subtitle": dm.fs(-0.5),
        "legend": dm.fs(-2.0),
    }
    pad = {"label": 6, "tick": 4}

    fan_model = enex.Fan()

    int_flow_range = np.linspace(1.0, 3.0, 100)
    ext_flow_range = np.linspace(1.5, 3.5, 100)

    colors = ["dm.soft blue", "dm.cool green", "dm.red5"]

    fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(9), dm.cm2in(12)))
    plt.subplots_adjust(left=0.12, right=0.7, top=0.92, bottom=0.1, wspace=0.9, hspace=0.5)

    fan_labels = ["Indoor unit", "Outdoor unit"]
    flow_ranges = [int_flow_range, ext_flow_range]
    ylim_pressure = [(130.0, 210.0), (250.0, 370.0)]
    yticks_pressure = [np.linspace(130.0, 210.0, 5), np.linspace(250.0, 370.0, 5)]
    ylim_efficiency = [(50.0, 70.0), (50.0, 70.0)]
    yticks_efficiency = [np.linspace(50.0, 70.0, 5), np.linspace(50.0, 70.0, 5)]
    ylim_power = [(200.0, 1200.0), (600.0, 2200.0)]
    yticks_power = [np.linspace(200.0, 1200.0, 5), np.linspace(600.0, 2200.0, 5)]
    xlims = [(1.0, 3.0), (1.5, 3.5)]
    xticks = [np.arange(1.0, 3.1, 0.5), np.arange(1.5, 3.6, 0.5)]

    for idx, (ax, fan, label, flow_range, ylim_p, yticks_p, ylim_e, yticks_e, ylim_pow, yticks_pow, xlim, xtick) in enumerate(
        zip(
            axes,
            fan_model.fan_list,
            fan_labels,
            flow_ranges,
            ylim_pressure,
            yticks_pressure,
            ylim_efficiency,
            yticks_efficiency,
            ylim_power,
            yticks_power,
            xlims,
            xticks,
        )
    ):
        pressure = fan_model.get_pressure(fan, flow_range)
        efficiency = fan_model.get_efficiency(fan, flow_range)
        power = fan_model.get_power(fan, flow_range)

        ax.set_xlabel("Air flow rate [m$^3$/s]", fontsize=fs["label"], labelpad=pad["label"])
        ax.set_ylabel("Efficiency [%]", color=colors[1], fontsize=fs["label"], labelpad=pad["label"])
        eff_line, = ax.plot(flow_range, efficiency * 100.0, color=colors[1], linestyle="-", linewidth=1.0, label="Efficiency")
        ax.tick_params(axis="x", labelsize=fs["tick"], pad=pad["tick"])
        ax.tick_params(axis="y", labelsize=fs["tick"], colors=colors[1], pad=pad["tick"])
        ax.set_xlim(xlim)
        ax.set_xticks(xtick)
        ax.set_ylim(ylim_e)
        ax.set_yticks(yticks_e)
        ax.spines["left"].set_color(colors[1])

        ax_power = ax.twinx()
        ax_power.set_ylabel("Power [W]", color=colors[2], fontsize=fs["label"], labelpad=pad["label"])
        power_line, = ax_power.plot(flow_range, power, color=colors[2], linestyle=":", linewidth=1.0, label="Power")
        ax_power.tick_params(axis="y", labelsize=fs["tick"], colors=colors[2], pad=pad["tick"])
        ax_power.set_ylim(ylim_pow)
        ax_power.set_yticks(yticks_pow)
        ax_power.spines["top"].set_visible(False)
        ax_power.spines["bottom"].set_visible(False)
        ax_power.spines["left"].set_visible(False)
        ax_power.spines["right"].set_color(colors[2])

        ax_pressure = ax.twinx()
        ax_pressure.spines["right"].set_position(("axes", 1.28))
        ax_pressure.set_ylabel("Pressure drop [Pa]", color=colors[0], fontsize=fs["label"], labelpad=pad["label"])
        pressure_line, = ax_pressure.plot(flow_range, pressure, color=colors[0], linestyle="--", linewidth=1.0, label="Pressure drop")
        ax_pressure.tick_params(axis="y", labelsize=fs["tick"], colors=colors[0], pad=pad["tick"])
        ax_pressure.set_ylim(ylim_p)
        ax_pressure.set_yticks(yticks_p)
        ax_pressure.spines["top"].set_visible(False)
        ax_pressure.spines["bottom"].set_visible(False)
        ax_pressure.spines["left"].set_visible(False)
        ax_pressure.spines["right"].set_color(colors[0])

        ax.grid(True)
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.tick_params(axis="y", which="minor", left=False)
        ax_power.tick_params(axis="y", which="minor", right=False)
        ax_pressure.tick_params(axis="y", which="minor", right=False)

        ax.text(0.01, 1.13, f"({chr(97 + idx)}) {label} fan", transform=ax.transAxes, fontsize=fs["subtitle"], va="top", ha="left")

        lines = [pressure_line, eff_line, power_line]
        labels = ["Pressure drop", "Efficiency", "Power"]
        ax.legend(lines, labels, loc="upper left", fontsize=fs["legend"], frameon=False)

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH, dpi=600)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
