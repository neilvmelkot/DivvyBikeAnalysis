import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR    = os.path.join("..", "output")
TRIP_ROOT     = os.path.join("..", "data", "bikes_raw")
GOV_TEMP_CSV  = os.path.join(
    "..", "data", "weather_raw",
    "chicago_monthly_avg_temp_weathergov.csv"
)
YEAR = "2023"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Find every real 2023 *.csv, skip macOS metadata
    pattern = re.compile(rf"{YEAR}\d{{2}}-divvy-tripdata\.csv$")
    files = []
    for root, _, fnames in os.walk(TRIP_ROOT):
        # skip any __MACOSX directories
        if "__MACOSX" in root:
            continue
        for fn in fnames:
            if fn.startswith("._"):
                continue
            if pattern.match(fn):
                files.append(os.path.join(root, fn))
    files = sorted(files)
    if not files:
        print(f"❌  No files found matching {YEAR}*.csv under {TRIP_ROOT}")
        return

    print(f"✅  Found {len(files)} real files for {YEAR}:")
    for f in files:
        print("   ", os.path.basename(f))

    # 2) Load & concat, count rides per hour
    dfs = []
    for fp in files:
        df = pd.read_csv(fp,
                         usecols=["ride_id", "started_at"],
                         parse_dates=["started_at"])
        dfs.append(df)
    bikes = pd.concat(dfs, ignore_index=True)
    bikes.dropna(subset=["started_at"], inplace=True)
    bikes.set_index("started_at", inplace=True)

    hourly = bikes["ride_id"].resample("h").count()
    monthly_avg_rides = (
        hourly
        .groupby(hourly.index.month)
        .mean()
        .reindex(range(1,13), fill_value=0)
    )

    # 3) Load gov temps
    gov = pd.read_csv(GOV_TEMP_CSV)
    gov.set_index("datetime", inplace=True)
    gov_temps = gov["TEMP"].reindex(range(1,13), fill_value=float("nan"))

    # --- save combined monthly data for the chart ---
    df_monthly = pd.DataFrame({
        "month": range(1,13),
        "avg_rides": monthly_avg_rides.values,
        "avg_temp": gov_temps.values
    })
    df_monthly.to_csv(
        os.path.join(OUTPUT_DIR, f"monthly_riders_temp_comparison_{YEAR}.csv"),
        index=False
    )

    # 4) Plot
    plt.figure(figsize=(10,6))
    ax = plt.gca()
    ax.plot(
        monthly_avg_rides.index,
        monthly_avg_rides.values,
        marker="o",
        label=f"Avg Riders/hour {YEAR}"
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Riders/hour")

    ax2 = ax.twinx()
    ax2.plot(
        gov_temps.index,
        gov_temps.values,
        marker="s",
        linestyle="--",
        color="tab:orange",
        label="Gov Avg Temp (°C)"
    )
    ax2.set_ylabel("Avg Temp (°C)")

    plt.title(f"Monthly Avg Riders {YEAR} vs Gov Avg Temp")
    lines, labels = ax.get_legend_handles_labels()
    l2, l2lab = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + l2lab, loc="upper left")

    out_file = os.path.join(
        OUTPUT_DIR,
        f"monthly_riders_temp_comparison_{YEAR}.png"
    )
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print("✅  Saved plot to", out_file)


if __name__ == "__main__":
    main()
