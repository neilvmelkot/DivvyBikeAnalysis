import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR    = "../output"
TEMP_BINS     = 30
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
    weather = weather_df.set_index("datetime")[["TEMP"]]

    print("Loading bike trip files…")
    TRIP_ROOT = os.path.join("..","data","bikes_raw")
    trips = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root: 
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            path = os.path.join(root, fn)
            df = pd.read_csv(path, usecols=["ride_id","started_at"], parse_dates=["started_at"])
            trips.append(df)
    bikes = pd.concat(trips, ignore_index=True)
    bikes.dropna(subset=["started_at"], inplace=True)
    bikes.set_index("started_at", inplace=True)

    print("Computing hourly ride counts…")
    hourly = bikes["ride_id"].resample("h").count().rename("ride_count")

    print("Merging with weather…")
    merged = hourly.to_frame().join(weather, how="inner").dropna()

    # remove ride_count outliers
    rc_Q1, rc_Q3 = merged["ride_count"].quantile([0.25,0.75])
    rc_IQR = rc_Q3 - rc_Q1
    merged = merged[merged["ride_count"].between(rc_Q1 - 1.5*rc_IQR, rc_Q3 + 1.5*rc_IQR)]

    # remove temp outlier
    t_Q1, t_Q3 = merged["TEMP"].quantile([0.25,0.75])
    t_IQR = t_Q3 - t_Q1
    merged = merged[merged["TEMP"].between(t_Q1 - 1.5*t_IQR, t_Q3 + 1.5*t_IQR)]

    print("Plotting rides vs temp percentiles…")
    sample = merged.sample(min(len(merged), SAMPLE_SIZE), random_state=1)
    plt.figure(figsize=(8,5))
    plt.scatter(sample["TEMP"], sample["ride_count"], alpha=0.25, s=10)

    bins = np.linspace(merged["TEMP"].min(), merged["TEMP"].max(), TEMP_BINS)
    grp = merged.groupby(pd.cut(merged["TEMP"], bins), observed=True)["ride_count"]
    p25 = grp.quantile(0.25)
    p50 = grp.quantile(0.50)
    p75 = grp.quantile(0.75)
    mids = p50.index.map(lambda interval: interval.mid)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({
        "mid": mids,
        "p25": p25.values,
        "p50": p50.values,
        "p75": p75.values
    }).to_csv(os.path.join(OUTPUT_DIR, "rides_vs_temp_percentiles.csv"), index=False)

    plt.plot(mids, p50.values, color="red",   label="Median")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Rides per hour")
    plt.title("Hourly Rides vs Temperature")
    plt.legend()
    save_fig("rides_vs_temp_percentiles.png")
    print("Saved rides_vs_temp_percentiles.png")

if __name__ == "__main__":
    main()
