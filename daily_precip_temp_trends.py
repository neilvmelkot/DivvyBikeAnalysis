import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = "../output"
CHUNK_SIZE = 100_000
PRCP_DATASET = "curiel/chicago-weather-database"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1) Load & aggregate weather to daily precip + daily mean temp ---
    weather_path = kagglehub.dataset_download(PRCP_DATASET)
    if weather_path.endswith(".zip"):
        import zipfile, pathlib
        with zipfile.ZipFile(weather_path) as z:
            z.extractall(pathlib.Path(weather_path).with_suffix(""))
        weather_path = str(pathlib.Path(weather_path).with_suffix(""))

    wfiles = glob.glob(os.path.join(weather_path, "*.csv"))
    weather_df = pd.concat([pd.read_csv(f) for f in wfiles], ignore_index=True)
    weather_df.columns = weather_df.columns.str.strip().str.upper()
    weather_df["datetime"] = pd.to_datetime(
        weather_df[["YEAR","MO","DY","HR"]]
                  .rename(columns={"YEAR":"year","MO":"month","DY":"day","HR":"hour"})
    )
    weather_df.set_index("datetime", inplace=True)

    # clamp precipitation to >=0, then sum by day
    weather_df["PRCP"] = weather_df["PRCP"].clip(lower=0)
    daily_precip = weather_df["PRCP"].resample("D").sum().rename("precip")

    # compute daily mean temperature
    daily_temp = weather_df["TEMP"].resample("D").mean().rename("temp")

    # --- 2) Load trips & aggregate to daily ride counts ---
    TRIP_ROOT = os.path.join("..", "data", "bikes_raw")
    daily_chunks = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root:
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            path = os.path.join(root, fn)
            for chunk in pd.read_csv(path,
                                     usecols=["ride_id","started_at"],
                                     parse_dates=["started_at"],
                                     chunksize=CHUNK_SIZE):
                chunk.dropna(subset=["started_at"], inplace=True)
                chunk.set_index("started_at", inplace=True)
                daily_chunks.append(chunk["ride_id"].resample("D").count())

    daily_rides = (
        pd.concat(daily_chunks)
          .groupby(level=0).sum()
          .rename("ride_count")
    )

    # --- 3) Merge everything into one DataFrame ---
    df = pd.concat([daily_rides, daily_precip, daily_temp], axis=1).dropna()
    # rename columns for clarity
    df.columns = ["ride_count", "precip", "temp"]

    # --- 4) Define 4 categories based on rain + temp threshold ---
    conditions = [
        (df["temp"] < 15) & (df["precip"] > 0),
        (df["temp"] < 15) & (df["precip"] == 0),
        (df["temp"] >= 15) & (df["precip"] > 0),
        (df["temp"] >= 15) & (df["precip"] == 0),
    ]
    labels = [
        "Rain & Temp < 15°C",
        "No Rain & Temp < 15°C",
        "Rain & Temp ≥ 15°C",
        "No Rain & Temp ≥ 15°C",
    ]
    df["category"] = np.select(conditions, labels, default="Other")

    # --- 5) Compute average rides per category ---
    summary = df.groupby("category")["ride_count"].mean().reindex(labels)

    # Save raw numbers
    summary.to_frame("avg_rides_per_day").to_csv(
        os.path.join(OUTPUT_DIR, "daily_rides_by_rain_temp_category.csv")
    )

    # --- 6) Plot bar chart ---
    plt.figure(figsize=(8,5))
    summary.plot(kind="bar")
    plt.xlabel("Condition")
    plt.ylabel("Average Rides per Day")
    plt.title("Average Daily Rides by Rain & Temperature Condition")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "daily_rides_rain_temp_bar.png"))
    plt.close()
    print("✅ Saved bar chart to output/daily_rides_rain_temp_bar.png")

if __name__ == "__main__":
    main()
