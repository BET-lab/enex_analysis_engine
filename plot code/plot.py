import matplotlib.pyplot as plt
from pathlib import Path
from data_setting import get_weekday_df


def main():
    # Use the default get_weekday_df behavior which auto-locates the data/ folder
    df = get_weekday_df()
    print('DataFrame columns:', df.columns.tolist())
    print('First 5 rows:')
    print(df.head())

    # Simple plot: total cooling and heating time series (if columns exist)
    cooling_col = 'DistrictCooling:Facility [J](TimeStep)'
    heating_col = 'DistrictHeatingWater:Facility [J](TimeStep) '

    plt.figure(figsize=(10, 4))
    if cooling_col in df.columns:
        plt.plot(df[cooling_col].values, label='Cooling [J]')
    if heating_col in df.columns:
        plt.plot(df[heating_col].values, label='Heating [J]')

    plt.legend()
    plt.title('Cooling and Heating time series (processed)')
    plt.xlabel('Hourly index')
    plt.ylabel('Energy [J]')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
