from pathlib import Path
import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = Path(__file__).parent.resolve()
EVAL_DIR = BASE_DIR.parent / 'output' / 'evaluation_results'
EVAL_DIR.mkdir(parents=True, exist_ok=True)
BIKES_DIR = BASE_DIR.parent / 'data' / 'bikes_raw'
WEATHER_KG = 'curiel/chicago-weather-database'
DATE_START = '2021-01-01'
DATE_END = '2024-12-31'

def get_season(month):
    if month in [12, 1, 2]: return 'winter'
    if month in [3, 4, 5]: return 'spring'
    if month in [6, 7, 8]: return 'summer'
    return 'fall'

print('Fetching weather data...')
weather_path = kagglehub.dataset_download(WEATHER_KG)
if str(weather_path).endswith('.zip'):
    import zipfile
    zp = Path(weather_path)
    with zipfile.ZipFile(weather_path) as z:
        z.extractall(zp.with_suffix(''))
    weather_path = str(zp.with_suffix(''))

wfiles = glob.glob(os.path.join(str(weather_path), '*.csv'))
wdf = pd.concat([pd.read_csv(f) for f in wfiles], ignore_index=True)
wdf.columns = wdf.columns.str.strip().str.upper()
print(f'Loaded weather: {len(wfiles)} files, {len(wdf)} rows')

for col in ['TEMP','PRCP','HMDT','WND_SPD','ATM_PRESS']:
    if col in wdf.columns:
        wdf[col].replace([-999, -9999], np.nan, inplace=True)

wdf['datetime'] = pd.to_datetime(
    wdf[['YEAR','MO','DY','HR']]
       .rename(columns={'YEAR':'year','MO':'month','DY':'day','HR':'hour'})
)
wdf.set_index('datetime', inplace=True)

def preprocess_weather(df):
    df = df[['TEMP','PRCP','HMDT','WND_SPD','ATM_PRESS']].rename(
        columns={'TEMP':'temp','PRCP':'precip','HMDT':'humidity','WND_SPD':'wind','ATM_PRESS':'pressure'}
    )
    df['precip'] = df['precip'].clip(lower=0)
    df.dropna(inplace=True)
    return df

weather_df = preprocess_weather(wdf)
print(f'Weather processed: {len(weather_df)} hourly records')

print('Loading bike trip data...')
all_parts = []
count_files = 0
for root, _, files in os.walk(str(BIKES_DIR)):
    if '__MACOSX' in root: continue
    for fn in files:
        if fn.startswith('._') or not fn.endswith('-divvy-tripdata.csv'): continue
        count_files += 1
        df_chunk = pd.read_csv(
            os.path.join(root, fn),
            usecols=['ride_id','started_at'], parse_dates=['started_at']
        )
        df_chunk.dropna(subset=['started_at'], inplace=True)
        df_chunk.set_index('started_at', inplace=True)
        all_parts.append(df_chunk['ride_id'].resample('h').count())

hourly_rides = pd.concat(all_parts).groupby(level=0).sum().rename('rides')
print(f'Loaded rides: {count_files} files, {len(hourly_rides)} hourly records')

df = hourly_rides.to_frame().join(weather_df, how='inner')
df = df.loc[DATE_START:DATE_END]
print(f'Merged and filtered to {len(df)} records between {DATE_START} and {DATE_END}')

initial_len = len(df)
df.dropna(inplace=True)
print(f'Dropped NaNs: {initial_len - len(df)} records removed, {len(df)} remain')
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['season'] = df['month'].apply(get_season)
print('Added temporal and season features')

features = ['temp','precip','humidity','wind','hour','dayofweek','month']
X = df[features]
y = df['rides']
print(f'Feature matrix: {X.shape[0]} samples, {X.shape[1]} features')

models = {
    'LinearRegression': Pipeline([
        ('scale', StandardScaler()),
        ('linreg', LinearRegression())
    ]),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

tscv = TimeSeriesSplit(n_splits=5)
results = []

for name, model in models.items():
    print(f'--- Model: {name}')
    split = int(len(df) * 0.8)
    print(f'    Training on {split}, testing on {len(df) - split}')
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model.fit(X_tr, y_tr)
    print(f'    Fitted {name}')

    if name == 'LinearRegression':
        coefs = model.named_steps['linreg'].coef_
        weights = pd.Series(coefs, index=features)
    else:
        importances = model.feature_importances_
        weights = pd.Series(importances, index=features)

    weights_df = weights.reset_index()
    weights_df.columns = ['feature', 'weight']
    weight_path = EVAL_DIR / f'{name}_feature_weights.csv'
    weights_df.to_csv(weight_path, index=False)
    print(f'    Feature weights saved to {weight_path}')

    y_pred_test = pd.Series(
        np.clip(model.predict(X_te), 0, None),
        index=y_te.index
    )
    y_pred_full = pd.Series(
        np.clip(model.predict(X), 0, None),
        index=y.index
    )

    r2 = r2_score(y_te, y_pred_test)
    mse = mean_squared_error(y_te, y_pred_test)
    mae = mean_absolute_error(y_te, y_pred_test)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
    print(f'    Test Metrics: R2={r2:.4f}, MSE={mse:.1f}, MAE={mae:.1f}, CV_R2_mean={cv_scores.mean():.4f}')

    test_df = pd.DataFrame({
        'actual': y_te,
        'predicted': y_pred_test,
        'season': df['season'].iloc[split:]
    })
    seasonal = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        mask = test_df['season'] == season
        n = mask.sum()
        if n > 0:
            season_r2 = r2_score(test_df.loc[mask, 'actual'], test_df.loc[mask, 'predicted'])
            seasonal.append({'model': name, 'season': season, 'n': int(n), 'r2': season_r2})
    seasonal_df = pd.DataFrame(seasonal)
    total_n = seasonal_df['n'].sum()
    seasonal_df['weight'] = seasonal_df['n'] / total_n
    seasonal_df['weighted_r2'] = seasonal_df['r2'] * seasonal_df['weight']
    weighted_avg = seasonal_df['weighted_r2'].sum()
    print(f'    Weighted seasonal R2 average: {weighted_avg:.4f}')

    seasonal_path = EVAL_DIR / f'{name}_seasonal_metrics.csv'
    seasonal_df.to_csv(seasonal_path, index=False)
    print(f'    Seasonal metrics saved to {seasonal_path}')

    results.append({
        'model': name,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    })


    plt.figure(figsize=(10, 4))
    plt.plot(y_te.index, y_te, label='Actual')
    plt.plot(y_pred_test.index, y_pred_test, label='Predicted (test)', alpha=0.7)
    plt.title(f'{name}: Actual vs Predicted (Test)')
    plt.xlabel('Datetime')
    plt.ylabel('Rides')
    plt.legend()
    plt.tight_layout()
    test_plot = EVAL_DIR / f'{name}_timeseries_test.png'
    plt.savefig(test_plot)
    plt.close()
    print(f'    Test plot saved to {test_plot}')

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, y, label='Actual')
    plt.plot(y_pred_full.index, y_pred_full, label='Predicted (full)', alpha=0.7)
    plt.title(f'{name}: Actual vs Predicted (Full Range)')
    plt.xlabel('Datetime')
    plt.ylabel('Rides')
    plt.legend()
    plt.tight_layout()
    full_plot = EVAL_DIR / f'{name}_timeseries_full.png'
    plt.savefig(full_plot)
    plt.close()
    print(f'    Full-range plot saved to {full_plot}')

res_df = pd.DataFrame(results)
metrics_path = EVAL_DIR / 'model_evaluation_metrics.csv'
res_df.to_csv(metrics_path, index=False)
print(f'Overall metrics saved to {metrics_path}')

print('Generating HTML report...')
html = ['<!DOCTYPE html>', '<html><head><meta charset="UTF-8"><title>Ride Prediction Evaluation</title></head><body>']
html.append('<h1>Overall Model Metrics</h1>')
html.append(res_df.to_html(index=False))
for name in models:
    html.append(f'<h2>{name} Feature Weights</h2>')
    fw_df = pd.read_csv(EVAL_DIR / f'{name}_feature_weights.csv')
    html.append(fw_df.to_html(index=False))
    html.append(f'<h2>{name} Seasonal R²</h2>')
    season_df = pd.read_csv(EVAL_DIR / f'{name}_seasonal_metrics.csv')
    html.append(season_df.to_html(index=False))
    html.append(f'<h2>{name} Actual vs Predicted (Test)</h2>')
    html.append(f'<img src="{name}_timeseries_test.png" style="max-width:800px;">')
    html.append(f'<h2>{name} Actual vs Predicted (Full Range)</h2>')
    html.append(f'<img src="{name}_timeseries_full.png" style="max-width:800px;">')
html.append('</body></html>')

with open(EVAL_DIR / 'index.html', 'w', encoding='utf-8') as f:
    f.write('\n'.join(html))
print(f'HTML report at {EVAL_DIR / "index.html"}')
