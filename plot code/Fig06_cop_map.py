"""Figure 6 – Part-load COP maps for cooling and heating modes."""
from __future__ import annotations

from pathlib import Path

import dartwork_mpl as dm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np

PNG_PATH = Path("figure") / "Fig. 6.png"
SVG_PATH = Path("figure") / "Fig. 6.svg"


def _build_colormap_segment(name: str, start: float, stop: float) -> mcolors.LinearSegmentedColormap:
    cmap = get_cmap("coolwarm")
    return mcolors.LinearSegmentedColormap.from_list(name, cmap(np.linspace(start, stop, 256)))


def main() -> None:
    dm.use_style()
    plt.rcParams["font.size"] = 9

    fs = {
        "label": dm.fs(0),
        "tick": dm.fs(-1.5),
        "subtitle": dm.fs(-0.5),
        "cbar_label": dm.fs(-2.0),
        "cbar_tick": dm.fs(-2.0),
    }
    pad = {"label": 6, "tick": 4}

    t_ev_l = 12.0
    cop_ref = 4.0

    plr_cooling = np.linspace(0.2, 1.0, 500)
    t_cond_e = np.linspace(15.0, 35.0, 500)
    plr_grid_cooling, t_cond_grid = np.meshgrid(plr_cooling, t_cond_e)

    eirf_temp_cooling = 0.38 + 0.02 * t_ev_l + 0.01 * t_cond_grid
    eirf_plr_cooling = 0.22 + 0.50 * plr_grid_cooling + 0.26 * plr_grid_cooling**2
    cop_cooling = plr_grid_cooling * cop_ref / (eirf_temp_cooling * eirf_plr_cooling)

    plr_heating = np.linspace(0.2, 1.0, 500)
    t_0 = np.linspace(-15.0, 25.0, 500)
    plr_grid_heating, t_0_grid = np.meshgrid(plr_heating, t_0)
    cop_heating = -7.46 * (plr_grid_heating - 0.0047 * t_0_grid - 0.477) ** 2 + 0.0941 * t_0_grid + 4.34

    cmap_cooling = _build_colormap_segment("coolwarm_left", 0.0, 0.45).reversed()
    cmap_heating = _build_colormap_segment("coolwarm_right", 0.55, 1.0)

    fig, (ax_cooling, ax_heating) = plt.subplots(1, 2, figsize=(dm.cm2in(17), dm.cm2in(7)))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.18, wspace=0.25)

    norm_cooling = mcolors.Normalize(vmin=0.0, vmax=6.0)
    norm_heating = mcolors.Normalize(vmin=0.0, vmax=7.0)

    mappable_cooling = ax_cooling.pcolormesh(t_cond_grid, plr_grid_cooling, cop_cooling, shading="auto", cmap=cmap_cooling, norm=norm_cooling, rasterized=True)
    cbar_cooling = fig.colorbar(mappable_cooling, ax=ax_cooling)
    cbar_cooling.set_label("COP [ - ]", fontsize=fs["cbar_label"], labelpad=pad["label"])
    cbar_cooling.ax.tick_params(labelsize=fs["cbar_tick"])
    cbar_cooling.set_ticks(np.arange(0.0, 6.1, 1.0))

    contour_cooling = ax_cooling.contour(t_cond_grid, plr_grid_cooling, cop_cooling, levels=np.arange(3.0, 5.6, 0.5), colors="dm.white", linewidths=0.7, alpha=0.9)
    ax_cooling.clabel(contour_cooling, inline=True, fontsize=fs["tick"], fmt="%.1f")

    ax_cooling.set_xlabel("Environmental temperature [$^{\\circ}$C]", fontsize=fs["label"], labelpad=pad["label"])
    ax_cooling.set_ylabel("Part load ratio [ - ]", fontsize=fs["label"], labelpad=pad["label"])
    ax_cooling.set_xlim(15.0, 35.0)
    ax_cooling.set_ylim(0.2, 1.0)
    ax_cooling.set_xticks(np.arange(15, 36, 5))
    ax_cooling.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax_cooling.tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    ax_cooling.tick_params(axis="both", which="minor", labelsize=fs["tick"], pad=pad["tick"])
    ax_cooling.text(0.02, 1.09, "(a) Cooling mode", transform=ax_cooling.transAxes, fontsize=fs["subtitle"], va="top", ha="left")

    mappable_heating = ax_heating.pcolormesh(t_0_grid, plr_grid_heating, cop_heating, shading="auto", cmap=cmap_heating, norm=norm_heating, rasterized=True)
    cbar_heating = fig.colorbar(mappable_heating, ax=ax_heating)
    cbar_heating.set_label("COP [ - ]", fontsize=fs["cbar_label"], labelpad=pad["label"])
    cbar_heating.ax.tick_params(labelsize=fs["cbar_tick"])
    cbar_heating.set_ticks(np.arange(0.0, 7.1, 1.0))

    contour_heating = ax_heating.contour(t_0_grid, plr_grid_heating, cop_heating, levels=np.arange(1.0, 7.1, 1.0), colors="dm.white", linewidths=0.7, alpha=0.9)
    ax_heating.clabel(contour_heating, inline=True, fontsize=fs["tick"], fmt="%.1f")

    ax_heating.set_xlabel("Environmental temperature [$^{\\circ}$C]", fontsize=fs["label"], labelpad=pad["label"])
    ax_heating.set_ylabel("Part load ratio [ - ]", fontsize=fs["label"], labelpad=pad["label"])
    ax_heating.set_xlim(-15.0, 25.0)
    ax_heating.set_ylim(0.2, 1.0)
    ax_heating.set_xticks(np.arange(-15, 26, 10))
    ax_heating.set_yticks(np.arange(0.2, 1.1, 0.2))
    ax_heating.tick_params(axis="both", which="major", labelsize=fs["tick"], pad=pad["tick"])
    ax_heating.tick_params(axis="both", which="minor", labelsize=fs["tick"], pad=pad["tick"])
    ax_heating.text(0.01, 1.09, "(b) Heating mode", transform=ax_heating.transAxes, fontsize=fs["subtitle"], va="top", ha="left")

    PNG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(SVG_PATH, dpi=600, transparent=True)
    dm.util.save_and_show(fig)


if __name__ == "__main__":
    main()
