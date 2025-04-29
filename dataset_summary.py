import os
import glob
import pandas as pd

# Summary script for Divvy bike-sharing dataset with outlier removal by ride duration and daily ride count
# Computes basic dataset metrics without heavy resampling

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
TRIP_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'bikes_raw'))

# Duration thresholds in minutes
MIN_DURATION = 1      # minimum 1 minute
MAX_DURATION = 1440   # maximum 24 hours

# Outlier removal for daily ride counts using IQR
IQR_FACTOR = 1.5

def main():
    import datetime
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gather all trip data files
    trip_files = []
    for root, _, files in os.walk(TRIP_ROOT):
        for fn in files:
            if fn.endswith('-divvy-tripdata.csv') and not fn.startswith('._'):
                trip_files.append(os.path.join(root, fn))

    all_dfs = []
    # Load and filter by duration
    for fp in trip_files:
        df = pd.read_csv(
            fp,
            usecols=['ride_id', 'started_at', 'ended_at'],
            parse_dates=['started_at', 'ended_at'],
            infer_datetime_format=True
        )
        df = df.dropna(subset=['started_at', 'ended_at'])
        df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60.0
        df = df[(df['duration_min'] >= MIN_DURATION) & (df['duration_min'] <= MAX_DURATION)]
        all_dfs.append(df[['ride_id','started_at']])

    # Concatenate all filtered rides
    master = pd.concat(all_dfs, ignore_index=True)
    master.set_index('started_at', inplace=True)

    # Compute daily ride counts
    daily_counts = master['ride_id'].resample('D').count()

    # Remove daily count outliers via IQR
    q1 = daily_counts.quantile(0.25)
    q3 = daily_counts.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - IQR_FACTOR * iqr, q3 + IQR_FACTOR * iqr
    filtered = daily_counts[(daily_counts >= lower) & (daily_counts <= upper)]

    # Compute summary metrics
    total_files = len(trip_files)
    start_date = filtered.index.min().date()
    end_date   = filtered.index.max().date()
    days_covered = len(filtered)
    total_rides = int(filtered.sum())
    avg_rides_per_day  = round(filtered.mean(), 2)
    avg_rides_per_hour = round(filtered.mean() / 24, 2)

    # Build summary DataFrame
    summary = pd.DataFrame({
        'metric': [
            'total_files',
            'start_date',
            'end_date',
            'days_covered',
            'total_rides',
            'avg_rides_per_day',
            'avg_rides_per_hour'
        ],
        'value': [
            total_files,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            days_covered,
            total_rides,
            avg_rides_per_day,
            avg_rides_per_hour
        ]
    })

    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, 'dataset_summary.csv')
    summary.to_csv(out_path, index=False)
    print(f'Dataset summary saved to {out_path}')

if __name__ == '__main__':
    main()
