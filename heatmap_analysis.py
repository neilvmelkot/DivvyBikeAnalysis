import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR = "../output"

sns.set(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

def save_fig(fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

def main():
    print("📥 Fetching weather data…")
    weather_path = kagglehub.dataset_download("curiel/chicago-weather-database")
    if weather_path.endswith(".zip"):
        import zipfile, pathlib
        with zipfile.ZipFile(weather_path) as z:
            z.extractall(pathlib.Path(weather_path).with_suffix(""))
        weather_path = str(pathlib.Path(weather_path).with_suffix(""))

    # load weather
    wfiles = glob.glob(os.path.join(weather_path, "*.csv"))
    weather_df = pd.concat([pd.read_csv(f) for f in wfiles], ignore_index=True)
    weather_df.columns = weather_df.columns.str.strip().str.upper()
    weather_df["datetime"] = pd.to_datetime(
        weather_df[["YEAR","MO","DY","HR"]]
            .rename(columns={"YEAR":"year","MO":"month","DY":"day","HR":"hour"})
    )
    weather = weather_df.set_index("datetime")[["TEMP"]]  # only need temp for merging

    print("📚 Loading bike trip files…")
    TRIP_ROOT = os.path.join("..", "data", "bikes_raw")
    trips = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root:
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            df = pd.read_csv(
                os.path.join(root, fn),
                usecols=["ride_id","started_at"],
                parse_dates=["started_at"]
            )
            trips.append(df)
    bikes = pd.concat(trips, ignore_index=True)
    bikes.dropna(subset=["started_at"], inplace=True)
    bikes.set_index("started_at", inplace=True)

    print("⚙️ Computing hourly ride counts…")
    hourly = bikes["ride_id"].resample("h").count().rename("ride_count")

    print("🔗 Merging with weather…")
    merged = hourly.to_frame().join(weather, how="inner")

    print("📈 Creating heatmap…")
    heat = merged.pivot_table(
        values="ride_count",
        index=merged.index.hour,
        columns=merged.index.dayofweek,
        aggfunc="mean"
    )

    # --- save the pivot table behind the heatmap ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    heat.to_csv(os.path.join(OUTPUT_DIR, "heatmap_hourly_dayofweek.csv"))

    plt.figure(figsize=(8,6))
    sns.heatmap(heat, cmap="coolwarm")
    plt.xlabel("Day of Week (0=Mon)")
    plt.ylabel("Hour of Day")
    plt.title("Average Rides/hour by Day & Hour")
    save_fig("heatmap_hourly_dayofweek.png")
    print("✅ Saved heatmap_hourly_dayofweek.png")

if __name__ == "__main__":
    main()
