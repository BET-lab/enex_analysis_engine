"""Utilities to build the cleaned small office dataset for plotting scripts."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Ensure the src directory is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import enex_analysis as enex
from enex_analysis import calc_util as cu

RAW_FILE = Path("data") / "small_office_hour.csv"
PROCESSED_DIR = Path("data") / "processed"
PROCESSED_FILE = PROCESSED_DIR / "small_office_ready.csv"
COOLING_FILTERED_FILE = PROCESSED_DIR / "small_office_cooling_filtered.csv"
HEATING_FILTERED_FILE = PROCESSED_DIR / "small_office_heating_filtered.csv"

COOLING_J_COL = "DistrictCooling:Facility [J](TimeStep)"
HEATING_J_COL_RAW = "DistrictHeatingWater:Facility [J](TimeStep) "
HEATING_J_COL = "DistrictHeatingWater:Facility [J](TimeStep)"
COOLING_W_COL = "DistrictCooling:Facility [W](TimeStep)"
HEATING_W_COL = "DistrictHeatingWater:Facility [W](TimeStep)"


def load_raw_small_office(path: Path | str = RAW_FILE) -> pd.DataFrame:
    """Return the raw small office dataset."""
    return pd.read_csv(Path(path))


def _strip_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["Date/Time_clean"] = (
        cleaned["Date/Time"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )
    return cleaned


def _apply_weekday_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()
    filtered["Relative_Day_Index"] = filtered.index // 24
    filtered["Day"] = (6 + filtered["Relative_Day_Index"]) % 7
    weekend = filtered["Day"].isin([5, 6])
    filtered.loc[weekend, COOLING_J_COL] = 0
    filtered.loc[weekend, HEATING_J_COL_RAW] = 0
    return filtered


def _apply_month_filters(df: pd.DataFrame) -> pd.DataFrame:
    month_filtered = df.copy()
    month_filtered["Month"] = month_filtered["Date/Time_clean"].str.slice(0, 2).astype(int)
    heating_months = [1, 2, 3, 10, 11, 12]
    cooling_months = [4, 5, 6, 7, 8, 9]
    month_filtered.loc[
        month_filtered["Month"].isin(heating_months), COOLING_J_COL
    ] = 0
    month_filtered.loc[
        month_filtered["Month"].isin(cooling_months), HEATING_J_COL_RAW
    ] = 0
    month_filtered.reset_index(drop=True, inplace=True)
    return month_filtered


def _safe_max(values: Iterable[float]) -> float:
    series = pd.Series(values)
    maximum = series.max()
    return float(maximum if pd.notna(maximum) and maximum > 0 else 1.0)


def _create_base_lists(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mirror the notebook's initial list creation for downstream processing."""

    base_lists = {
        "date_list": df["Date/Time_clean"].to_numpy(),
        "Toa_list": df["Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "Tia_list": df["CORE_ZN:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "cooling_load_J": df[COOLING_J_COL].astype(float).to_numpy(),
        "heating_load_J": df[HEATING_J_COL_RAW].astype(float).to_numpy(),
        "attic_temp": df["ATTIC:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "core_temp": df["CORE_ZN:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "perimeter_z1": df["PERIMETER_ZN_1:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "perimeter_z2": df["PERIMETER_ZN_2:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "perimeter_z3": df["PERIMETER_ZN_3:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "perimeter_z4": df["PERIMETER_ZN_4:Zone Air Temperature [C](TimeStep)"].astype(float).to_numpy(),
        "date_original": df["Date/Time"].to_numpy(),
    }

    base_lists["cooling_load_W"] = base_lists["cooling_load_J"] / 3600.0
    base_lists["heating_load_W"] = base_lists["heating_load_J"] / 3600.0

    zone_temp_stack = [
        base_lists["core_temp"],
        base_lists["perimeter_z1"],
        base_lists["perimeter_z2"],
        base_lists["perimeter_z3"],
        base_lists["perimeter_z4"],
    ]
    base_lists["zone_temp_list"] = zone_temp_stack

    return base_lists


def _add_performance_metrics(df: pd.DataFrame, base_lists: dict[str, np.ndarray]) -> pd.DataFrame:
    """Calculate COP, exergy efficiency, and demand columns based on prepared lists."""

    Toa = base_lists["Toa_list"]
    Tia = base_lists["Tia_list"]
    cooling_w = base_lists["cooling_load_W"]
    heating_w = base_lists["heating_load_W"]

    max_cooling = _safe_max(cooling_w)
    max_heating = _safe_max(heating_w)

    cooling_cop = np.full_like(cooling_w, np.nan, dtype=float)
    heating_cop = np.full_like(heating_w, np.nan, dtype=float)
    cooling_ex_eff = np.full_like(cooling_w, np.nan, dtype=float)
    heating_ex_eff = np.full_like(heating_w, np.nan, dtype=float)

    for idx, (t_out, q_cool, q_heat) in enumerate(zip(Toa, cooling_w, heating_w)):
        if q_cool > 0.0:
            ashp_cool = enex.AirSourceHeatPump_cooling()
            ashp_cool.T0 = float(t_out)
            ashp_cool.T_a_room = 22.0
            ashp_cool.Q_r_int = float(q_cool)
            ashp_cool.Q_r_max = float(max_cooling)
            ashp_cool.system_update()
            if ashp_cool.X_eff >= 0.0:
                cooling_ex_eff[idx] = float(ashp_cool.X_eff)
                cooling_cop[idx] = float(ashp_cool.COP_sys)

        if q_heat > 0.0:
            ashp_heat = enex.AirSourceHeatPump_heating()
            ashp_heat.T0 = float(t_out)
            ashp_heat.T_a_room = 22.0
            ashp_heat.Q_r_int = float(q_heat)
            ashp_heat.Q_r_max = float(max_heating)
            ashp_heat.system_update()
            if ashp_heat.X_eff >= 0.0:
                heating_ex_eff[idx] = float(ashp_heat.X_eff)
                heating_cop[idx] = float(ashp_heat.COP_sys)

    df = df.copy()
    df["ASHP_cooling_COP"] = cooling_cop
    df["ASHP_heating_COP"] = heating_cop
    df["ASHP_cooling_exergy_efficiency"] = cooling_ex_eff
    df["ASHP_heating_exergy_efficiency"] = heating_ex_eff

    df["Cooling_demand_W_m2"] = cooling_w / 511.0
    df["Heating_demand_W_m2"] = heating_w / 511.0

    carnot_factor = 1 - (
        cu.C2K(Toa.astype(float)) / cu.C2K(Tia.astype(float))
    )
    df["Cooling_exergy_demand_W_m2"] = -df["Cooling_demand_W_m2"] * carnot_factor
    df["Heating_exergy_demand_W_m2"] = df["Heating_demand_W_m2"] * carnot_factor

    df["ASHP_cooling_exergy_efficiency_zero_filled"] = (
        df["ASHP_cooling_exergy_efficiency"].fillna(0.0)
    )
    df["ASHP_heating_exergy_efficiency_zero_filled"] = (
        df["ASHP_heating_exergy_efficiency"].fillna(0.0)
    )

    return df


def prepare_small_office(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook-derived cleaning rules to the raw dataset."""
    cleaned = _strip_datetime_column(df)
    cleaned = _apply_weekday_filters(cleaned)
    cleaned = _apply_month_filters(cleaned)
    base_lists = _create_base_lists(cleaned)
    columns_to_keep = [
        "Date/Time",
        "Date/Time_clean",
        "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
        "CORE_ZN:Zone Air Temperature [C](TimeStep)",
        "ATTIC:Zone Air Temperature [C](TimeStep)",
        "PERIMETER_ZN_1:Zone Air Temperature [C](TimeStep)",
        "PERIMETER_ZN_2:Zone Air Temperature [C](TimeStep)",
        "PERIMETER_ZN_3:Zone Air Temperature [C](TimeStep)",
        "PERIMETER_ZN_4:Zone Air Temperature [C](TimeStep)",
        COOLING_J_COL,
        HEATING_J_COL_RAW,
        "Month",
        "Day",
    ]
    result = cleaned.loc[:, columns_to_keep].copy()
    # Keep the original spaced heating column, while adding a trimmed alias to match the notebook-derived scripts.
    result[HEATING_J_COL] = base_lists["heating_load_J"]
    result[COOLING_W_COL] = base_lists["cooling_load_W"]
    result[HEATING_W_COL] = base_lists["heating_load_W"]
    result = _add_performance_metrics(result, base_lists)
    return result


def build_input_lists(df: pd.DataFrame | None = None) -> dict[str, np.ndarray]:
    """Return the core lists (temps, loads, dates) exactly as assembled in the notebook."""

    source = df if df is not None else load_raw_small_office()
    cleaned = _strip_datetime_column(source)
    cleaned = _apply_weekday_filters(cleaned)
    cleaned = _apply_month_filters(cleaned)
    return _create_base_lists(cleaned)


def build_dataset(force: bool = False) -> Path:
    """Create the processed CSVs if missing or when force=True."""
    if PROCESSED_FILE.exists() and COOLING_FILTERED_FILE.exists() and HEATING_FILTERED_FILE.exists() and not force:
        return PROCESSED_FILE

    raw_df = load_raw_small_office()
    prepared_df = prepare_small_office(raw_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(PROCESSED_FILE, index=False)

    cooling_filtered = prepared_df.loc[
        prepared_df["ASHP_cooling_exergy_efficiency"].notna(),
        [
            "Date/Time",
            "Date/Time_clean",
            "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
            "CORE_ZN:Zone Air Temperature [C](TimeStep)",
            COOLING_W_COL,
            "Cooling_demand_W_m2",
            "Cooling_exergy_demand_W_m2",
            "ASHP_cooling_COP",
            "ASHP_cooling_exergy_efficiency",
        ],
    ]
    cooling_filtered.to_csv(COOLING_FILTERED_FILE, index=False)

    heating_filtered = prepared_df.loc[
        prepared_df["ASHP_heating_exergy_efficiency"].notna(),
        [
            "Date/Time",
            "Date/Time_clean",
            "Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)",
            "CORE_ZN:Zone Air Temperature [C](TimeStep)",
            HEATING_W_COL,
            "Heating_demand_W_m2",
            "Heating_exergy_demand_W_m2",
            "ASHP_heating_COP",
            "ASHP_heating_exergy_efficiency",
        ],
    ]
    heating_filtered.to_csv(HEATING_FILTERED_FILE, index=False)

    return PROCESSED_FILE


def load_processed_dataset(path: Path | str = PROCESSED_FILE) -> pd.DataFrame:
    """Load the processed dataset, building it first when needed."""
    file_path = Path(path)
    if not file_path.exists():
        if file_path == PROCESSED_FILE:
            build_dataset(force=False)
        else:
            raise FileNotFoundError(f"Processed dataset not found at {file_path}")
    return pd.read_csv(file_path)


def main() -> None:
    built_path = build_dataset(force=False)
    print(f"Processed dataset ready at {built_path}")
    if COOLING_FILTERED_FILE.exists():
        print(f"Cooling dataset ready at {COOLING_FILTERED_FILE}")
    if HEATING_FILTERED_FILE.exists():
        print(f"Heating dataset ready at {HEATING_FILTERED_FILE}")


if __name__ == "__main__":
    main()
