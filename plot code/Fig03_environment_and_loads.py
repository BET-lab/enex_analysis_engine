"""Figure 3 – Ambient temperature and space conditioning loads across the year."""
from __future__ import annotations

from pathlib import Path

import dartwork_mpl as dm
import matplotlib.pyplot as plt
import numpy as np

from data_prep import load_processed_dataset

PNG_PATH = Path("figure") / "Fig. 3.png"
PDF_PATH = Path("figure") / "Fig. 3.pdf"


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

    df = load_processed_dataset()

    toa_series = df["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].to_numpy()
    cooling_load_kw = df["DistrictCooling:Facility [W](TimeStep)"].to_numpy() / 1000.0
    heating_load_kw = df["DistrictHeatingWater:Facility [W](TimeStep)"].to_numpy() / 1000.0

    hours = np.arange(len(toa_series))
    xticks = np.linspace(0, len(toa_series), 7)

    fig, axes = plt.subplots(2, 1, figsize=(dm.cm2in(17), dm.cm2in(11)))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.1, hspace=0.25)

    axes[0].plot(hours, toa_series, color="dm.green4", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("Environmental temperature [$^{\\circ}$C]", fontsize=fs["label"], labelpad=pad["label"])
    axes[0].tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    axes[0].set_xticks(xticks)
    axes[0].set_yticks(np.arange(-10, 41, 10))
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].text(0.01, 0.97, "(a)", transform=axes[0].transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")

    axes[1].plot(hours, cooling_load_kw, color="dm.blue6", linewidth=0.8, alpha=0.5, label="Cooling load")
    axes[1].plot(hours, heating_load_kw, color="dm.red6", linewidth=0.8, alpha=0.5, label="Heating load")
    axes[1].set_xlabel("Hour of year [h]", fontsize=fs["label"], labelpad=pad["label"])
    axes[1].set_ylabel("Load [kW]", fontsize=fs["label"], labelpad=pad["label"] + 2)
    axes[1].tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    axes[1].set_xticks(xticks)
    axes[1].set_yticks(np.arange(0, 31, 10))
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].text(0.01, 0.97, "(b)", transform=axes[1].transAxes, fontsize=fs["subtitle"], fontweight="bold", va="top", ha="left")
    axes[1].legend(loc="upper right", ncol=2, frameon=False, fontsize=fs["legend"], columnspacing=1.0, labelspacing=0.8, bbox_to_anchor=(0.97, 0.995), handlelength=1.8)

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH, dpi=600)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
