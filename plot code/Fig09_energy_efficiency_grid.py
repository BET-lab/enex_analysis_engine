"""Figure 9 – System energy efficiency (COP) across temperature and load bins."""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import patches
import dartwork_mpl as dm

from data_prep import load_processed_dataset

PNG_PATH = Path("figure") / "Fig. 9.png"
PDF_PATH = Path("figure") / "Fig. 9.pdf"


def _bin_average(values_x: np.ndarray, values_y: np.ndarray, metric: np.ndarray, bins_x: np.ndarray, bins_y: np.ndarray) -> np.ndarray:
    avg = np.full((len(bins_y) - 1, len(bins_x) - 1), np.nan, dtype=float)
    for i in range(len(bins_x) - 1):
        x_low, x_high = bins_x[i], bins_x[i + 1]
        x_mask = (values_x >= x_low) & (values_x < x_high)
        if not np.any(x_mask):
            continue
        for j in range(len(bins_y) - 1):
            y_low, y_high = bins_y[j], bins_y[j + 1]
            mask = x_mask & (values_y >= y_low) & (values_y < y_high)
            if np.any(mask):
                avg[j, i] = float(np.nanmean(metric[mask]))
    return avg


def _add_grid_rectangles(ax: plt.Axes, bins_x: np.ndarray, bins_y: np.ndarray, color: str = "white", linewidth: float = 1.0) -> None:
    for i in range(len(bins_x) - 1):
        for j in range(len(bins_y) - 1):
            rect = patches.Rectangle(
                (bins_x[i], bins_y[j]),
                bins_x[i + 1] - bins_x[i],
                bins_y[j + 1] - bins_y[j],
                linewidth=linewidth,
                edgecolor=color,
                facecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)


def _add_cell_labels(
    ax: plt.Axes,
    avg: np.ndarray,
    bins_x: np.ndarray,
    bins_y: np.ndarray,
    fs_text: float,
    threshold: float,
    dark: bool = False,
) -> None:
    for i in range(len(bins_x) - 1):
        for j in range(len(bins_y) - 1):
            value = avg[j, i]
            if np.isnan(value):
                continue
            text_color = "white" if (value >= threshold) ^ dark else "black"
            ax.text(
                (bins_x[i] + bins_x[i + 1]) / 2,
                (bins_y[j] + bins_y[j + 1]) / 2,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=fs_text,
                color=text_color,
                zorder=3,
            )


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9

    fs = {
        "label": dm.fs(0),
        "tick": dm.fs(-1.5),
        "subtitle": dm.fs(-0.5),
        "cbar_label": dm.fs(-2.0),
        "cbar_tick": dm.fs(-2.0),
        "setpoint": dm.fs(-1.0),
        "text": dm.fs(-3.0),
    }
    pad = {"label": 6, "tick": 4}

    df = load_processed_dataset()

    cooling_mask = df["ASHP_cooling_COP"].notna() & (df["DistrictCooling:Facility [W](TimeStep)"] > 0)
    heating_mask = df["ASHP_heating_COP"].notna() & (df["DistrictHeatingWater:Facility [W](TimeStep)"] > 0)

    toa_cooling = df.loc[cooling_mask, "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].to_numpy()
    load_cooling = df.loc[cooling_mask, "DistrictCooling:Facility [W](TimeStep)"].to_numpy() / 1000.0
    cop_cooling = df.loc[cooling_mask, "ASHP_cooling_COP"].to_numpy()

    toa_heating = df.loc[heating_mask, "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].to_numpy()
    load_heating = df.loc[heating_mask, "DistrictHeatingWater:Facility [W](TimeStep)"].to_numpy() / 1000.0
    cop_heating = df.loc[heating_mask, "ASHP_heating_COP"].to_numpy()

    bin_temp = 2.5
    bin_load = 5.0
    x_range = (-10.0, 35.0)
    y_range = (0.0, 30.0)

    bins_x = np.arange(x_range[0], x_range[1] + bin_temp, bin_temp)
    bins_y = np.arange(y_range[0], y_range[1] + bin_load, bin_load)

    avg_cooling = _bin_average(toa_cooling, load_cooling, cop_cooling, bins_x, bins_y)
    avg_heating = _bin_average(toa_heating, load_heating, cop_heating, bins_x, bins_y)

    coolwarm_left = mcolors.LinearSegmentedColormap.from_list(
        "coolwarm_left", cm.get_cmap("coolwarm")(np.linspace(0, 0.45, 256))
    )
    coolwarm_right = mcolors.LinearSegmentedColormap.from_list(
        "coolwarm_right", cm.get_cmap("coolwarm")(np.linspace(0.55, 1.0, 256))
    )

    fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(17), dm.cm2in(13)), sharex=False, sharey=True)
    plt.subplots_adjust(left=0.07, right=1.07, top=0.95, bottom=0.08, hspace=0.25)

    im1 = axes[0].pcolormesh(bins_x, bins_y, avg_cooling, cmap=coolwarm_left.reversed(), vmin=1.5, vmax=4.0)
    _add_grid_rectangles(axes[0], bins_x, bins_y, color="white", linewidth=1.0)
    _add_cell_labels(axes[0], avg_cooling, bins_x, bins_y, fs["text"], threshold=3.65)

    axes[0].set_ylabel("Cooling load [kW]", fontsize=fs["label"], labelpad=pad["label"])
    axes[0].tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    axes[0].minorticks_off()
    axes[0].axvline(x=22, color="dm.teal6", linestyle="--", linewidth=0.6)
    axes[0].text(20.8, 26.2, "Setpoint", rotation=90, fontsize=fs["setpoint"], color="dm.teal6", ha="left", va="center")
    axes[0].text(0.01, 0.97, "(a)", transform=axes[0].transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")

    cbar1 = fig.colorbar(im1, ax=axes[0], orientation="vertical", pad=0.02, aspect=20)
    cbar1.set_label("Energy efficiency (COP$_{sys}$) [ - ]", fontsize=fs["cbar_label"], labelpad=pad["label"])
    cbar1.ax.tick_params(labelsize=fs["cbar_tick"])

    im2 = axes[1].pcolormesh(bins_x, bins_y, avg_heating, cmap=coolwarm_right, vmin=1.5, vmax=4.0)
    _add_grid_rectangles(axes[1], bins_x, bins_y, color="dm.white", linewidth=1.0)
    _add_cell_labels(axes[1], avg_heating, bins_x, bins_y, fs["text"], threshold=3.65)

    axes[1].set_ylabel("Heating load [kW]", fontsize=fs["label"], labelpad=pad["label"])
    axes[1].tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    axes[1].minorticks_off()
    axes[1].axvline(x=22, color="dm.teal6", linestyle="--", linewidth=0.6)
    axes[1].text(20.8, 26.2, "Setpoint", rotation=90, fontsize=fs["setpoint"], color="dm.teal6", ha="left", va="center")
    axes[1].text(0.01, 0.97, "(b)", transform=axes[1].transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")

    cbar2 = fig.colorbar(im2, ax=axes[1], orientation="vertical", pad=0.02, aspect=20)
    cbar2.set_label("Energy efficiency (COP$_{sys}$) [ - ]", fontsize=fs["cbar_label"], labelpad=pad["label"])
    cbar2.ax.tick_params(labelsize=fs["cbar_tick"])

    for ax in axes:
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel("Environmental temperature [$^{\\circ}$C]", fontsize=fs["label"], labelpad=pad["label"])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(bins_x, minor=True)
        ax.tick_params(axis="x", which="minor", length=1.6, color="dm.gray7")

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH, dpi=600)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
