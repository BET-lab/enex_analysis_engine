"""Figure 12 – Weekly profiles of temperature, load, COP, and exergy efficiency."""
from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import dartwork_mpl as dm

import enex_analysis as enex

from data_prep import load_processed_dataset

PNG_PATH = Path("figure") / "Fig. 12.png"
PDF_PATH = Path("figure") / "Fig. 12.pdf"

COOLING_J_COL = "DistrictCooling:Facility [J](TimeStep)"
HEATING_J_COL = "DistrictHeatingWater:Facility [J](TimeStep)"


def _compute_metrics(row, q_r_max_cooling: float, q_r_max_heating: float) -> tuple[float, float]:
    toa = row["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"]
    cooling_load = row[COOLING_J_COL] / 3600.0
    heating_load = row[HEATING_J_COL] / 3600.0

    exergy_effi = 0.0
    cop = 0.0

    if cooling_load > 0.0:
        ashp = enex.AirSourceHeatPump_cooling()
        ashp.T0 = toa
        ashp.T_a_room = 22.0
        ashp.Q_r_int = cooling_load
        ashp.Q_r_max = q_r_max_cooling
        ashp.system_update()
        if ashp.X_eff > 0.0:
            exergy_effi = ashp.X_eff * 100.0
            cop = ashp.COP_sys
    elif heating_load > 0.0:
        ashp = enex.AirSourceHeatPump_heating()
        ashp.T0 = toa
        ashp.T_a_room = 22.0
        ashp.Q_r_int = heating_load
        ashp.Q_r_max = q_r_max_heating
        ashp.system_update()
        if ashp.X_eff > 0.0:
            exergy_effi = ashp.X_eff * 100.0
            cop = ashp.COP_sys

    return cop, exergy_effi


def _style_axis_colors(ax_left: plt.Axes, ax_right: plt.Axes, left_color: str, right_color: str) -> None:
    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)

    ax_left.spines["left"].set_color(left_color)
    ax_left.yaxis.label.set_color(left_color)
    ax_left.tick_params(axis="y", which="both", colors=left_color)
    ax_left.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax_right.spines["right"].set_color(right_color)
    ax_right.yaxis.label.set_color(right_color)
    ax_right.tick_params(axis="y", which="both", colors=right_color)
    ax_right.yaxis.set_minor_locator(AutoMinorLocator(2))


def _align_ylabel_positions(ax_left: plt.Axes, ax_right: plt.Axes, left_x: float, right_x: float, y_coord: float) -> None:
    ax_left.yaxis.set_label_coords(left_x, y_coord)
    ax_right.yaxis.set_label_coords(right_x, y_coord)


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9

    fs = {
        "label": dm.fs(0),
        "tick": dm.fs(-1.5),
        "subtitle": dm.fs(-0.5),
        "setpoint": dm.fs(-1.0),
    }
    pad = {"label": 6, "tick": 4}

    textarea_left = -0.13
    textarea_right = 1.13
    textarea_y = 0.5

    months_to_plot = [1, 8, 4, 10]
    layout = [[1, 8], [4, 10]]
    month_titles = {1: "(a) January", 8: "(b) August", 4: "(c) April", 10: "(d) October"}
    start_map = {1: 9, 4: 10, 8: 7, 10: 10}

    temp_color = "dm.green4"
    cooling_load_color = "dm.blue5"
    heating_load_color = "dm.red5"
    cop_color = "dm.orange4"
    exergy_color = "dm.gray6"

    df = load_processed_dataset()
    weekday_df = df.copy()

    day_all = weekday_df["Date/Time_clean"].str.slice(3, 5).astype(int)
    month_series = weekday_df["Month"].astype(int)
    hour_all = weekday_df["Date/Time_clean"].str.slice(6, 8).astype(int)
    weekday_df["xpos"] = day_all + hour_all / 24.0
    weekday_df["Month"] = month_series

    start_series = month_series.map(start_map)
    mask = (
        month_series.isin(months_to_plot)
        & day_all.ge(start_series)
        & day_all.le(start_series + 4)
    )
    weekday_subset = weekday_df.loc[mask].copy()
    weekday_subset["Day"] = day_all[mask].values

    fig = plt.figure(figsize=(dm.cm2in(17), dm.cm2in(18)))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.55, hspace=0.25)
    plt.subplots_adjust(left=0.08, right=0.92, top=0.97, bottom=0.05)

    for row in range(2):
        for col in range(2):
            month = layout[row][col]
            month_df = weekday_subset[weekday_subset["Month"] == month].sort_values("xpos")
            if month_df.empty:
                continue

            start_day = start_map[month]
            end_day = start_day + 5
            month_df = month_df[(month_df["Day"] >= start_day) & (month_df["Day"] <= end_day)]

            q_r_max_cooling = (month_df[COOLING_J_COL] / 3600.0).max()
            q_r_max_heating = (month_df[HEATING_J_COL] / 3600.0).max()

            cop_vals = []
            exergy_vals = []
            for _, r in month_df.iterrows():
                cop, exergy = _compute_metrics(r, q_r_max_cooling, q_r_max_heating)
                cop_vals.append(cop)
                exergy_vals.append(exergy)
            month_df = month_df.assign(COP=cop_vals, ExergyEff=exergy_vals)

            inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[row, col], hspace=0.18)
            ax_top = fig.add_subplot(inner[0, 0])
            ax_bottom = fig.add_subplot(inner[1, 0], sharex=ax_top)

            ax_temp = ax_top
            ax_load = ax_top.twinx()
            ax_temp.set_zorder(ax_load.get_zorder() + 1)
            ax_temp.patch.set_visible(False)

            if month in [1, 10]:
                load_series = month_df[HEATING_J_COL] / 3600.0 / 1000.0
                load_label = "Heating load [kW]"
                load_color = heating_load_color
            else:
                load_series = month_df[COOLING_J_COL] / 3600.0 / 1000.0
                load_label = "Cooling load [kW]"
                load_color = cooling_load_color

            ax_load.fill_between(month_df["xpos"], 0.0, load_series, color=load_color, alpha=0.3, zorder=1)
            ax_load.set_ylabel(load_label, fontsize=fs["label"])

            ax_temp.plot(
                month_df["xpos"],
                month_df["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"],
                linewidth=0.7,
                color=temp_color,
                zorder=3,
            )
            ax_temp.axhline(y=22.0, color="dm.teal6", linestyle="--", linewidth=0.7, alpha=0.7)
            ax_temp.text(
                start_day + 0.1,
                22.7,
                "Setpoint",
                fontsize=fs["setpoint"],
                color="dm.teal6",
                ha="left",
                va="bottom",
            )

            ax_temp.set_xlim(start_day, end_day)
            ax_temp.set_ylim(-10.0, 40.0)
            ax_temp.set_yticks(np.arange(-10, 41, 10))
            ax_load.set_ylim(0.0, 30.0)
            ax_load.set_yticks(np.arange(0, 31, 10))

            ax_temp.set_ylabel("Environmental temp. [$^{\\circ}$C]", fontsize=fs["label"])
            ax_temp.set_title(month_titles[month], fontsize=fs["subtitle"], loc="left")
            ax_temp.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_temp.tick_params(axis="x", which="both", labelsize=fs["tick"], pad=pad["tick"])
            ax_temp.tick_params(axis="y", which="major", labelsize=fs["tick"], pad=pad["tick"])
            ax_load.tick_params(axis="y", which="major", labelsize=fs["tick"], pad=pad["tick"])
            ax_temp.spines["left"].set_linewidth(0.6)
            ax_load.spines["right"].set_linewidth(0.6)

            _style_axis_colors(ax_temp, ax_load, temp_color, load_color)
            _align_ylabel_positions(ax_temp, ax_load, textarea_left, textarea_right, textarea_y)

            ax_cop = ax_bottom
            ax_exergy = ax_bottom.twinx()
            ax_exergy.set_zorder(ax_cop.get_zorder() + 1)
            ax_cop.patch.set_visible(False)

            ax_cop.plot(month_df["xpos"], month_df["COP"], linewidth=0.7, color=cop_color, zorder=2)
            ax_exergy.plot(
                month_df["xpos"],
                month_df["ExergyEff"],
                linewidth=0.7,
                linestyle="--",
                color=exergy_color,
                zorder=4,
            )

            ax_cop.set_xlim(start_day, end_day)
            ax_cop.set_ylim(0.0, 5.0)
            ax_exergy.set_ylim(0.0, 50.0)
            ax_cop.set_ylabel("Energy efficiency [ - ]", fontsize=fs["label"])
            ax_exergy.set_ylabel("Exergy efficiency [%]", fontsize=fs["label"])

            ticks = np.arange(start_day, end_day + 1, 1)
            ax_cop.set_xticks(ticks)
            ax_cop.set_xticklabels([str(d) for d in ticks], fontsize=fs["tick"])
            ax_cop.xaxis.set_minor_locator(AutoMinorLocator(2))

            ax_cop.tick_params(axis="y", which="major", labelsize=fs["tick"], pad=pad["tick"])
            ax_exergy.tick_params(axis="y", which="major", labelsize=fs["tick"], pad=pad["tick"])
            ax_cop.spines["left"].set_linewidth(0.6)
            ax_exergy.spines["right"].set_linewidth(0.6)

            _style_axis_colors(ax_cop, ax_exergy, cop_color, exergy_color)
            _align_ylabel_positions(ax_cop, ax_exergy, textarea_left, textarea_right, textarea_y)

            ax_cop.set_xlabel("Day of month [day]", fontsize=fs["label"])

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH, dpi=600)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
