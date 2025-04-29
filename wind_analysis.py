import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR    = "../output"
WIND_BINS     = 30
SAMPLE_SIZE   = 5000

sns.set(style="whitegrid")
plt.rcParams.update({"figure.dpi": 120})

def save_fig(fname):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

def main():
    print("Fetching weather data…")
    weather_path = kagglehub.dataset_download("curiel/chicago-weather-database")
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
    weather = weather_df.set_index("datetime")[["WND_SPD"]].rename(columns={"WND_SPD":"wind"})

    print("📚 Loading bike trip files…")
    TRIP_ROOT = os.path.join("..","data","bikes_raw")
    trips = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root: continue
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

    print("Computing hourly ride counts…")
    hourly = bikes["ride_id"].resample("h").count().rename("ride_count")

    print("Merging with weather…")
    merged = hourly.to_frame().join(weather, how="inner").dropna()

    # strict wind bounds before outlier removal
    merged = merged[(merged["wind"] >= 0) & (merged["wind"] <= 40)]

    # remove wind outliers
    wQ1, wQ3 = merged["wind"].quantile([0.25, 0.75])
    wIQR = wQ3 - wQ1
    merged = merged[merged["wind"].between(wQ1 - 1.0 * wIQR, wQ3 + 1.0 * wIQR)]

    # remove ride_count outliers
    Q1, Q3 = merged["ride_count"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    merged = merged[merged["ride_count"].between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)]

    print("Plotting rides vs wind median…")
    sample = merged.sample(min(len(merged), SAMPLE_SIZE), random_state=1)
    plt.figure(figsize=(8,5))
    plt.scatter(sample["wind"], sample["ride_count"], alpha=0.25, s=10)

    bins = np.linspace(merged["wind"].min(), merged["wind"].max(), WIND_BINS)
    grp = merged.groupby(pd.cut(merged["wind"], bins), observed=True)["ride_count"]
    median = grp.quantile(0.50)
    mids = median.index.map(lambda interval: interval.mid)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({
        "mid": mids,
        "median": median.values
    }).to_csv(os.path.join(OUTPUT_DIR, "rides_vs_wind_median.csv"), index=False)

    plt.plot(mids, median.values, color="red", label="Median")
    plt.xlabel("Wind Speed (mph)")
    plt.ylabel("Rides per hour")
    plt.title("Hourly Rides vs Wind Speed")
    plt.legend()
    save_fig("rides_vs_wind_median.png")
    print("Saved rides_vs_wind_median.png")

if __name__ == "__main__":
    main()
