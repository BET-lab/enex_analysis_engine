"""Figure 13 – Monthly ambient temperatures and exergy balance."""
import sys
sys.path.append('src')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import dartwork_mpl as dm
from figure_setting import fs, pad
import enex_analysis as enex
import calc_util as cu
import enex_analysis as enex

from data_prep import load_processed_dataset

PATH = "figure"
FIG_NAME = "Fig. 13"
PNG_PATH = PATH +  '/' + FIG_NAME + ".png"
SVG_PATH = PATH +  '/' + FIG_NAME + ".svg"

COOLING_J_COL = "DistrictCooling:Facility [J](TimeStep)"
HEATING_J_COL = "DistrictHeatingWater:Facility [J](TimeStep)"


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9


    df = load_processed_dataset()
    if "Month" not in df:
        df["Month"] = df["Date/Time_clean"].str.slice(0, 2).astype(int)

    cooling_load_list = (df[COOLING_J_COL] / 3600.0).to_numpy()
    heating_load_list = (df[HEATING_J_COL] / 3600.0).to_numpy()
    max_cooling = float(np.nanmax(cooling_load_list)) if np.any(cooling_load_list > 0.0) else 1.0
    max_heating = float(np.nanmax(heating_load_list)) if np.any(heating_load_list > 0.0) else 1.0

    grouped = df.groupby("Month")

    monthly_exergy_input: list[float] = []
    monthly_exergy_consumption: list[float] = []
    monthly_exergy_output: list[float] = []
    monthly_avg_COP: list[float] = []

    monthly_avg_temp = grouped["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].mean().to_numpy()
    monthly_min_temp = grouped["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].min().to_numpy()
    monthly_max_temp = grouped["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].max().to_numpy()

    for _, group in grouped:
        input_exergy = 0.0
        consumption_exergy = 0.0
        output_exergy = 0.0

        total_cooling_exergy_efficiency = 0.0
        total_heating_exergy_efficiency = 0.0
        total_cooling_cop = 0.0
        total_heating_cop = 0.0
        cooling_count = 0
        heating_count = 0

        for _, row in group.iterrows():
            Toa = row["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]
            cooling_load = row[COOLING_J_COL] / 3600.0
            heating_load = row[HEATING_J_COL] / 3600.0

            if cooling_load > 0.0:
                ashp_cooling = enex.AirSourceHeatPump_cooling()
                ashp_cooling.T0 = Toa
                ashp_cooling.T_a_room = 22.0
                ashp_cooling.Q_r_int = cooling_load
                ashp_cooling.Q_r_max = max_cooling
                ashp_cooling.system_update()
                if ashp_cooling.X_eff > 0.0:
                    input_exergy += (ashp_cooling.E_cmp + ashp_cooling.E_fan_int + ashp_cooling.E_fan_ext) * 3600.0
                    output_exergy += (ashp_cooling.X_a_int_out - ashp_cooling.X_a_int_in) * 3600.0
                    total_cooling_cop += ashp_cooling.COP_sys
                    total_cooling_exergy_efficiency += ashp_cooling.X_eff
                    cooling_count += 1

            if heating_load > 0.0:
                ashp_heating = enex.AirSourceHeatPump_heating()
                ashp_heating.T0 = Toa
                ashp_heating.T_a_room = 22.0
                ashp_heating.Q_r_int = heating_load
                ashp_heating.Q_r_max = max_heating
                ashp_heating.system_update()
                if ashp_heating.X_eff > 0.0:
                    input_exergy += (ashp_heating.E_cmp + ashp_heating.E_fan_int + ashp_heating.E_fan_ext) * 3600.0
                    output_exergy += (ashp_heating.X_a_int_out - ashp_heating.X_a_int_in) * 3600.0
                    total_heating_cop += ashp_heating.COP_sys
                    total_heating_exergy_efficiency += ashp_heating.X_eff
                    heating_count += 1

        consumption_exergy = input_exergy - output_exergy
        monthly_exergy_input.append(input_exergy * Wh2GWh)
        monthly_exergy_consumption.append(consumption_exergy * Wh2GWh)
        monthly_exergy_output.append(output_exergy * enex.W2GW)

        total_count = cooling_count + heating_count
        if total_count > 0:
            avg_cop = (total_cooling_cop + total_heating_cop) / total_count
        else:
            avg_cop = np.nan
        monthly_avg_COP.append(avg_cop)

    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


    MONTH_LENGTH = len(labels)
    x = np.arange(1, MONTH_LENGTH + 1)
    total_exergy = np.array(monthly_exergy_input)

    fig, (ax_temp, ax_exergy) = plt.subplots(
        2,
        1,
        figsize=(dm.cm2in(17), dm.cm2in(11)),
        gridspec_kw={"height_ratios": [1.0, 1.4]},
    )
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.2)

    fill_color = "dm.gray1"
    min_temp_color = "dm.blue5"
    max_temp_color = "dm.red5"
    avg_temp_color = "dm.gray8"
    percentage_color = "dm.black"
    avg_cop_color = "dm.orange6"

    marker_size = 1.5
    line_width = 0.5

    ax_temp.fill_between(x, monthly_min_temp, monthly_max_temp, color=fill_color, alpha=0.6)
    ax_temp.plot(x, monthly_max_temp, color=max_temp_color, linewidth=line_width, label="Max", marker="o", markersize=marker_size)
    ax_temp.plot(x, monthly_avg_temp, color=avg_temp_color, linewidth=line_width, label="Avg", marker="o", markersize=marker_size)
    ax_temp.plot(x, monthly_min_temp, color=min_temp_color, linewidth=line_width, label="Min", marker="o", markersize=marker_size)

    ax_temp.set_xlim(0.5, 12.5)
    ax_temp.set_ylim(-10.0, 40.0)
    ax_temp.set_xticks(x)
    ax_temp.set_xticklabels(labels, fontsize=fs["tick"])
    ax_temp.set_yticks(np.arange(-10, 40*1.001, 10))
    ax_temp.tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    ax_temp.text(0.01, 0.97, "(a)", transform=ax_temp.transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")
    ax_temp.set_ylabel("Environmental temperature [$^{\\circ}$C]", fontsize=fs["label"], labelpad=pad["label"])
    ax_temp.grid(True, linestyle="--", alpha=0.5)
    ax_temp.axhline(y=22.0, color="dm.teal6", linestyle="--", linewidth=line_width)
    ax_temp.text(0.6, 24.0, "Setpoint", fontsize=fs["setpoint"], color="dm.teal6", ha="left", va="center")

    legend = ax_temp.legend(
        loc="upper right",
        ncol=3,
        frameon=False,
        fontsize=fs["legend"],
        columnspacing=1.0,
        labelspacing=0.8,
        bbox_to_anchor=(0.99, 1.01),
        handlelength=1.8,
    )
    legend.set_zorder(10)

    bar_width = 0.57
    edge_color = "dm.gray8"
    colors = ["dm.gray2", "dm.green3"]

    ax_exergy.bar(
        x,
        monthly_exergy_output,
        bar_width,
        label="Exergy output",
        color=colors[1],
        alpha=0.75,
        zorder=3,
    )
    ax_exergy.bar(
        x,
        monthly_exergy_consumption,
        bar_width,
        bottom=monthly_exergy_output,
        label="Exergy consumption",
        color=colors[0],
        alpha=1.0,
        zorder=3,
    )

    for i, xpos in enumerate(x):
        total = total_exergy[i]
        consumption = monthly_exergy_consumption[i]
        output = monthly_exergy_output[i]
        if total > 0.0:
            consumption_ratio = consumption * 100.0 / total
            output_ratio = output * 100.0 / total
            ax_exergy.text(
                xpos,
                output + consumption / 2.0,
                f"{consumption_ratio:.1f}%",
                ha="center",
                va="center",
                fontsize=fs["text"],
                color=percentage_color,
                zorder=4,
            )
            ax_exergy.text(
                xpos,
                output / 2.0,
                f"{output_ratio:.1f}%",
                ha="center",
                va="center",
                fontsize=fs["text"],
                color=percentage_color,
                zorder=4,
            )

        rect = plt.Rectangle(
            (xpos - bar_width / 2.0, 0.0),
            bar_width,
            output + consumption,
            linewidth=line_width,
            edgecolor=edge_color,
            facecolor="none",
            alpha=0.7,
            zorder=3,
        )
        ax_exergy.add_patch(rect)

    ax_exergy.set_ylabel("Exergy [GWh]", fontsize=fs["label"], labelpad=pad["label"] + 5)
    ax_exergy.set_xlim(0.5, 12.5)
    ax_exergy.set_ylim(0.0, 5.0)
    ax_exergy.set_xticks(x)
    ax_exergy.set_xticklabels(labels, fontsize=fs["tick"])
    ax_exergy.set_yticks(np.arange(0.0, 5.1, 1.0))
    ax_exergy.tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    ax_exergy.text(0.01, 0.97, "(b)", transform=ax_exergy.transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")
    ax_exergy.spines["right"].set_visible(False)
    ax_exergy.grid(True, linestyle="--", alpha=0.6, zorder=1)
    ax_exergy.xaxis.grid(False)

    ax_exergy_right = ax_exergy.twinx()
    ax_exergy_right.plot(
        x,
        monthly_avg_COP,
        color=avg_cop_color,
        linewidth=line_width,
        label="COP",
        marker="o",
        markersize=marker_size,
    )
    ax_exergy_right.set_ylabel("Average COP$_{sys}$ [ - ]", fontsize=fs["label"], labelpad=pad["label"] + 5, color=avg_cop_color)
    ax_exergy_right.set_ylim(0.0, 5.0)
    ax_exergy_right.set_yticks(np.arange(0.0, 5.1, 1.0))
    ax_exergy_right.tick_params(axis="y", labelsize=fs["tick"], colors=avg_cop_color)
    ax_exergy_right.spines["right"].set_color(avg_cop_color)
    ax_exergy_right.spines["right"].set_linewidth(0.5)

    consumption_patch = Patch(facecolor="none", edgecolor=edge_color, linewidth=line_width, alpha=0.7, label="Exergy input")
    handles_left, labels_left = ax_exergy.get_legend_handles_labels()
    handles_left.insert(0, consumption_patch)
    labels_left.insert(0, "Exergy input")

    handles_right, labels_right = ax_exergy_right.get_legend_handles_labels()
    legend_combined = ax_exergy.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=fs["legend"],
        columnspacing=1.0,
        labelspacing=1.0,
        bbox_to_anchor=(0.99, 1.01),
        handlelength=1.8,
    )
    legend_combined.set_zorder(10)

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(SVG_PATH, dpi=600, transparent=True)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
