import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
OUTPUT_DIR    = "../output"
HUMIDITY_BINS = 30
SAMPLE_SIZE   = 5000

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
    weather = weather_df.set_index("datetime")[["HMDT"]].rename(columns={"HMDT":"humidity"})

    print("📚 Loading bike trip files…")
    TRIP_ROOT = os.path.join("..", "data", "bikes_raw")
    trips = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root:
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            path = os.path.join(root, fn)
            df = pd.read_csv(path,
                             usecols=["ride_id","started_at"],
                             parse_dates=["started_at"])
            trips.append(df)
    bikes = pd.concat(trips, ignore_index=True)
    bikes.dropna(subset=["started_at"], inplace=True)
    bikes.set_index("started_at", inplace=True)

    print("⚙️ Computing hourly ride counts…")
    hourly = bikes["ride_id"].resample("h").count().rename("ride_count")

    print("🔗 Merging with weather…")
    merged = hourly.to_frame().join(weather, how="inner").dropna()

    # Remove negative humidity values
    merged = merged[merged["humidity"] >= 0]

    # remove ride_count outliers via IQR
    Q1 = merged["ride_count"].quantile(0.25)
    Q3 = merged["ride_count"].quantile(0.75)
    IQR = Q3 - Q1
    mask = merged["ride_count"].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    merged = merged[mask]

    print("📈 Plotting scatter + percentile lines…")
    sample = merged.sample(min(len(merged), SAMPLE_SIZE), random_state=1)
    plt.figure(figsize=(8,5))
    plt.scatter(sample["humidity"], sample["ride_count"], alpha=0.25, s=10)

    # compute percentiles by humidity bin
    bins = np.linspace(merged["humidity"].min(), merged["humidity"].max(), HUMIDITY_BINS)
    grp = merged.groupby(pd.cut(merged["humidity"], bins), observed=True)["ride_count"]
    p25 = grp.quantile(0.25)
    p50 = grp.quantile(0.50)
    p75 = grp.quantile(0.75)
    mids = p50.index.map(lambda interval: interval.mid)

    # --- save the percentile-lines data behind the plot ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({
        "mid": mids,
        "p25": p25.values,
        "p50": p50.values,
        "p75": p75.values
    }).to_csv(os.path.join(OUTPUT_DIR, "rides_vs_humidity_percentiles.csv"), index=False)

    # plot percentile lines
    plt.plot(mids, p50.values, color="red",   label="Median")

    plt.xlabel("Humidity (%)")
    plt.ylabel("Rides per hour")
    plt.title("Hourly Rides vs Humidity (%)")
    plt.legend()
    save_fig("rides_vs_humidity_percentiles.png")
    print("✅ Saved rides_vs_humidity_percentiles.png to output/")
 
if __name__ == "__main__":
    main()
