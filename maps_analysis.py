import os
import glob
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
import branca.colormap as cm

# Configuration
OUTPUT_DIR = "../output"
TRIP_ROOT  = "../data/bikes_raw"
TOP_N      = 10

def create_colormap(vmin, vmax):
    return cm.LinearColormap(
        ["blue","green","yellow","red"],
        vmin=vmin, vmax=vmax, caption="Ride volume"
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📚 Loading station data…")
    # scan for all trip CSVs
    trip_files = []
    for root, _, files in os.walk(TRIP_ROOT):
        if "__MACOSX" in root:
            continue
        for fn in files:
            if fn.startswith("._") or not fn.endswith("-divvy-tripdata.csv"):
                continue
            trip_files.append(os.path.join(root, fn))

    # aggregate station counts
    stats = []
    for fp in trip_files:
        df = pd.read_csv(
            fp,
            usecols=["start_station_id","start_lat","start_lng","ride_id"]
        )
        stats.append(df)
    all_df = pd.concat(stats, ignore_index=True)
    station_stats = (
        all_df.groupby("start_station_id")
              .agg(start_lat=("start_lat","first"),
                   start_lng=("start_lng","first"),
                   count=("ride_id","count"))
    )

    # 1) Density heatmap
    print("🗺️ Building density heatmap…")
    m1 = folium.Map([41.8781,-87.6298], zoom_start=12)
    HeatMap(
        station_stats[["start_lat","start_lng","count"]].values,
        radius=15, blur=25, max_zoom=18, min_opacity=0.4
    ).add_to(m1)
    # add legend
    cmap = create_colormap(
        station_stats["count"].min(),
        station_stats["count"].max()
    )
    cmap.add_to(m1)
    m1.save(os.path.join(OUTPUT_DIR, "station_density_heatmap.html"))

    # 2) Gradient circles
    print("🗺️ Building gradient circle map…")
    m2 = folium.Map([41.8781,-87.6298], zoom_start=12)
    for sid, r in station_stats.iterrows():
        color = cmap(r["count"])
        folium.Circle(
            [r["start_lat"], r["start_lng"]],
            radius=10 + np.log1p(r["count"])*5,
            color=color, fill=True, fill_color=color, fill_opacity=0.6,
            popup=f"Station {sid}<br>Total rides: {r['count']}"
        ).add_to(m2)
    cmap.add_to(m2)
    m2.save(os.path.join(OUTPUT_DIR, "station_gradient_map.html"))

    # 3) Top‑10 markers
    print("🗺️ Building top‑10 station markers…")
    top10 = station_stats.nlargest(TOP_N, "count")
    m3 = folium.Map([41.8781,-87.6298], zoom_start=12)
    for sid, r in top10.iterrows():
        folium.Marker(
            [r["start_lat"], r["start_lng"]],
            popup=f"<b>Station {sid}</b><br>Total rides: {r['count']}",
            icon=folium.Icon(color="blue", icon="bicycle", prefix="fa")
        ).add_to(m3)
    m3.save(os.path.join(OUTPUT_DIR, "top10_stations_map.html"))

    print("✅ All maps saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
