import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------- Load and Prepare the Data ---------------------- #
def load_data(file_path):
    """Load an ERA5 dataset and make sure 'timestamp' is a datetime (UTC)."""
    try:
        df = pd.read_csv(file_path)

        # Ensure column exists and is parsed correctly
        if 'timestamp' not in df.columns:
            raise ValueError("❌ Missing 'timestamp' column in file.")

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File '{file_path}' is empty or invalid.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error while loading '{file_path}': {e}")
        return None


# ---------------------- Explore the Dataset ---------------------- #
def explore_data(name, df):
    """Show dataset structure and fill numeric NaNs with column means."""
    print(f"\n=== {name} Dataset ===")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Data types:\n", df.dtypes)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    print(f"\n{name} dataset ready.")


# ---------------------- Summary Statistics ---------------------- #
def summary_statistics(name, df):
    print(f"\nSummary statistics for {name}:")
    print(df.describe())


# ========================= TEMPORAL AGGREGATIONS ========================= #
def calculate_wind_speed(df):
    df['wind_speed'] = np.sqrt(df['u10m']**2 + df['v10m']**2)
    return df

def monthly_averages(df):
    """Monthly (month-end) averages for wind_speed (+ t2m if present)."""
    tmp = df.set_index('timestamp').sort_index()
    cols = ['wind_speed'] + (['t2m'] if 't2m' in tmp.columns else [])
    
    # Calculate monthly averages and reset index so timestamp is a normal column
    monthly = tmp[cols].resample('MS').mean(numeric_only=True).reset_index()
    return monthly

def seasonal_averages(df):
    """Meteorological seasons (DJF, MAM, JJA, SON) averages."""
    season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }
    tmp = df.set_index('timestamp').sort_index().copy()
    tmp['season'] = tmp.index.month.map(season_map)
    cols = ['wind_speed'] + (['t2m'] if 't2m' in tmp.columns else [])
    
    seasonal = tmp.groupby('season')[cols].mean(numeric_only=True)
    seasonal = seasonal.reset_index()  # make 'season' a normal column
    return seasonal


def compare_seasonal_patterns(berlin_seasonal, munich_seasonal):
    comp = berlin_seasonal.join(munich_seasonal, lsuffix='_Berlin', rsuffix='_Munich')
    print("\n=== Seasonal patterns (means) ===")
    print(comp)
    return comp


# ============================ STATISTICAL ANALYSIS ============================ #
def daily_extreme_wind(name,df, top_n=5):
    """Find and display the top-N days with highest daily max wind speeds."""
    # Calculate daily maximum wind speed
    daily_max = df.set_index('timestamp')['wind_speed'].resample('D').max()

    # Convert to DataFrame and reset index to align headers
    top_days = daily_max.nlargest(top_n).to_frame('daily_max_wind_speed').reset_index()

    print(f"\n=== {name} Top Extreme Wind Days ===")
    print(top_days.to_string(index=False))  # remove index numbers, align headers
    return top_days



def diurnal_pattern(df,name):
    """Calculate average wind speed for hours in UTC (00, 06, 12, 18)."""
    df = df.set_index('timestamp').sort_index().copy()
    hourly_avg = (
        df.groupby(df.index.hour)['wind_speed']
        .mean()
        .to_frame('mean_wind_speed')
        .reset_index()
        .rename(columns={'index': 'hour', 'timestamp': 'hour'})
    )
    hourly_avg['hour'] = hourly_avg['hour'].apply(lambda h: f"{int(h):02d}:00")
    print(f"\n=== {name} Diurnal (Hourly) Average Wind Speed ===")
    print(hourly_avg.to_string(index=False))
    return hourly_avg




# =============================== VISUALIZATIONS =============================== #
def plot_monthly_wind_speed(berlin_month, munich_month):
    """Plot monthly mean wind speed using timestamps on x-axis (no rotation)."""
    plt.figure(figsize=(9, 4))

    # Use timestamp column directly for x-axis
    plt.plot(berlin_month['timestamp'], berlin_month['wind_speed'], marker='o', label='Berlin')
    plt.plot(munich_month['timestamp'], munich_month['wind_speed'], marker='s', label='Munich')

    plt.title("Monthly Average Wind Speed (m/s)")
    plt.xlabel("Month")
    plt.ylabel("Wind Speed (m/s)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Format timestamps as Year-Month (no rotation)
    plt.xticks(berlin_month['timestamp'], berlin_month['timestamp'].dt.strftime('%Y-%m'))

    plt.tight_layout()
    plt.savefig("monthly_wind_speed.png", dpi=200)
    plt.close()



def plot_seasonal_comparison(berlin_season, munich_season):
    """Bar chart comparing seasonal mean wind speeds between Berlin and Munich."""
    seasons = berlin_season['season'].tolist()  # get the season names directly
    x = np.arange(len(seasons))
    w = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, berlin_season['wind_speed'], width=w, label='Berlin')
    plt.bar(x + w/2, munich_season['wind_speed'], width=w, label='Munich')

    plt.xticks(x, seasons)  # show DJF, MAM, JJA, SON
    plt.ylabel("Wind Speed (m/s)")
    plt.title("Seasonal Mean Wind Speed")
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("seasonal_wind_speed.png", dpi=200)
    plt.close()



def plot_wind_speed_histogram(berlin, munich):
    """Overlayed wind-speed histograms with KDE for Berlin & Munich."""
    plt.figure(figsize=(8, 5))
    sns.histplot(berlin['wind_speed'], label='Berlin', color='blue',  kde=True, alpha=0.4)
    sns.histplot(munich['wind_speed'], label='Munich', color='orange', kde=True, alpha=0.4)
    plt.title("Wind Speed Distribution")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("wind_speed_histogram.png", dpi=200)
    plt.close()


# ---------------------- Run Everything in Order ---------------------- #
def main():
    berlin = load_data("berlin_era5_wind_20241231_20241231.csv")
    munich = load_data("munich_era5_wind_20241231_20241231.csv")
    if berlin is None or munich is None:
        return

    explore_data("Berlin", berlin)
    explore_data("Munich", munich)
    summary_statistics("Berlin", berlin)
    summary_statistics("Munich", munich)

    berlin = calculate_wind_speed(berlin)
    munich = calculate_wind_speed(munich)
    berlin['wind_dir_deg'] = (np.degrees(np.arctan2(berlin['v10m'], berlin['u10m'])) + 360) % 360
    munich['wind_dir_deg'] = (np.degrees(np.arctan2(munich['v10m'], munich['u10m'])) + 360) % 360


    # Monthly and Seasonal averages
    berlin_month = monthly_averages(berlin)
    munich_month = monthly_averages(munich)
    berlin_season = seasonal_averages(berlin)
    munich_season = seasonal_averages(munich)
    print("\n=== Berlin Monthly Averages ===")
    print(berlin_month.head())
    print("\n=== Munich Monthly Averages ===")
    print(munich_month.head())
    print("\n=== Berlin Seasonal Averages ===")
    print(berlin_season)
    print("\n=== Munich Seasonal Averages ===")
    print(munich_season)

    compare_seasonal_patterns(berlin_season, munich_season)
    daily_extreme_wind("Berlin",berlin, top_n=5)
    daily_extreme_wind("Munich",munich, top_n=5)
    diurnal_pattern(berlin,"Berlin")
    diurnal_pattern(munich,"Munich")

    plot_monthly_wind_speed(berlin_month, munich_month)
    plot_seasonal_comparison(berlin_season, munich_season)
    plot_wind_speed_histogram(berlin, munich)


if __name__ == "__main__":
    main()



