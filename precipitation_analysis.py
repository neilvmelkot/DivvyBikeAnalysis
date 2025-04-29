import os
import glob
from datetime import datetime
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR      = "../output"
PRCP_BINS       = 30
SAMPLE_SIZE     = 5000
CHUNK_SIZE      = 100_000

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
    weather = weather_df.set_index("datetime")[["PRCP"]].rename(columns={"PRCP":"precip"})
    weather["precip"] = weather["precip"].clip(lower=0)
    daily_precip = weather["precip"].resample("D").sum().rename("precip")

    print("Loading bike trip files and computing daily counts in chunks…")
    TRIP_ROOT = os.path.join("..","data","bikes_raw")
    daily_chunks = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root:
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            path = os.path.join(root, fn)
            for chunk in pd.read_csv(
                path,
                usecols=["ride_id","started_at"],
                parse_dates=["started_at"],
                chunksize=CHUNK_SIZE
            ):
                chunk = chunk.dropna(subset=["started_at"])
                chunk.set_index("started_at", inplace=True)
                daily_chunks.append(chunk["ride_id"].resample("D").count())

    if not daily_chunks:
        print("No bike data found under", TRIP_ROOT)
        return

    daily_rides = (
        pd.concat(daily_chunks)
          .groupby(level=0)
          .sum()
          .rename("ride_count")
    )

    print("Merging daily rides with daily precipitation…")
    merged = pd.concat([daily_rides, daily_precip], axis=1).dropna()


    rc_Q1, rc_Q3 = merged["ride_count"].quantile([0.25,0.75])
    rc_IQR = rc_Q3 - rc_Q1
    merged = merged[
        merged["ride_count"]
              .between(rc_Q1 - 1.5*rc_IQR, rc_Q3 + 1.5*rc_IQR)
    ]
    p_Q1, p_Q3 = merged["precip"].quantile([0.25,0.75])
    p_IQR = p_Q3 - p_Q1
    merged = merged[
        merged["precip"]
              .between(p_Q1 - 1.5*p_IQR, p_Q3 + 1.5*p_IQR)
    ]

    print("Plotting daily rides vs precipitation…")
    sample = merged.sample(min(len(merged), SAMPLE_SIZE), random_state=1)
    plt.figure(figsize=(8,5))
    plt.scatter(sample["precip"], sample["ride_count"], alpha=0.6, s=20)
    plt.xlabel("Daily Precipitation (inches)")
    plt.ylabel("Total Rides per Day")
    plt.title("Daily Rides vs Daily Precipitation")
    bins = np.linspace(merged["precip"].min(), merged["precip"].max(), PRCP_BINS)
    grp = merged.groupby(pd.cut(merged["precip"], bins), observed=True)["ride_count"]
    p25 = grp.quantile(0.25)
    p50 = grp.quantile(0.50)
    p75 = grp.quantile(0.75)
    mids = p50.index.map(lambda interval: interval.mid)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({
        "precip_mid": mids,
        "p25": p25.values,
        "p50": p50.values,
        "p75": p75.values
    }).to_csv(
        os.path.join(OUTPUT_DIR, "rides_vs_daily_precip_percentiles.csv"),
        index=False
    )
    plt.plot(mids, p50.values, linewidth=2, label="Median rides", zorder=3)
    plt.legend()
    save_fig("rides_vs_daily_precip_percentiles.png")
    print("Saved rides_vs_daily_precip_percentiles.png")

if __name__ == "__main__":
    main()
