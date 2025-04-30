import os
import subprocess
import shutil
import sys

SRC_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(SRC_DIR, "../output"))
TEX_PATH   = os.path.join(OUTPUT_DIR, "bike_sharing_report.tex")
PDF_NAME   = "bike_sharing_report.pdf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tex = r"""
\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\title{Bike Sharing Demand Analysis in Urban Areas: The Role of Weather and Temporal Factors}
\author{Neil Melkot \\ and Nikhil Melkot}
\date{May 8, 2025}

\begin{document}
\maketitle

\begin{abstract}
This comprehensive study deciphers how weather variables and temporal rhythms coalesce to shape bike-sharing demand in Chicago. By merging Divvy trip data (2021-2024)\cite{DivvyData} with granular NOAA weather observations from the Kaggle ``curiel/chicago-weather-database'' dataset\cite{KaggleWeather} and deploying robust statistical and machine learning techniques, we unveil the primacy of temperature (corr=0.91) alongside secondary influences from humidity, wind, and precipitation. Our analysis, spanning exploratory visualizations, spatial heatmaps, monthly trend comparisons, and predictive modeling, culminate in an $R^2$ of 0.88. We contextualize patterns within urban mobility frameworks, providing actionable insights for dynamic fleet management and planning.
\end{abstract}

\section{Introduction}
Bike\‑sharing has emerged as a sustainable, low\‑emission complement to urban transit systems. However, operators face persistent challenges from supply-demand unevenness, leading to bike shortages or dock saturation. Understanding how external factors such as weather and time influence ridership is essential for optimising rebalancing strategies and improving service reliability. This research integrates multiple data sources, such as Divvy ridership records\cite{DivvyData}, NOAA meteorological logs via Kaggle\cite{KaggleWeather}, and spatial station metadata, in order to construct a nuanced forecasting framework that addresses both macro trends and localised usage patterns.

\section{Data and Methods}
\subsection{Data Acquisition and Preprocessing}
We retrieved weather data via the Kaggle ``curiel/chicago\‑weather\‑database'' dataset\cite{KaggleWeather} and concatenated hourly CSVs, calibrating negative precipitation to zero. Divvy trip data from 2021-2024 were sourced from Divvy’s system\‑data portal\cite{DivvyData} and merged, then resampled to produce both daily and hourly ridership counts.
\vspace{0.5em}

\[
Q_1 = \text{25th percentile}, \quad Q_3 = \text{75th percentile}, \quad \mathrm{IQR}=Q_3-Q_1,
\]

\[
X \in \bigl[\,Q_1-1.5\,\mathrm{IQR},\; Q_3+1.5\,\mathrm{IQR}\bigr]
\]
\vspace{0.5em}

This filtering produced a high quality corpus spanning \textbf{48} CSV files and 1,461 days (between 2021‑01‑01 to 2024‑12‑31), retaining \textbf{21 647 186} rides, an average of \textbf{14 817} trips per day (≈\textbf{617} rides\,hr$^{-1}$), thereby preserving meaningful variability for downstream analysis. If certain data required for a certain analysis was labelled with a NaN value, it was dropped in that aspect of analysis as well.

\subsection{Feature Engineering}
From each timestamp we extracted:
\begin{itemize}
  \item \textbf{hour of day} (0--23),
  \item \textbf{day of week} (0 = Monday to 6 = Sunday),
  \item \textbf{month} (1--12),
  \item \textbf{season}, via a simple mapping: winter (Dec–Feb), spring (Mar–May), summer (Jun–Aug), fall (Sep–Nov).
\end{itemize}

The final feature set comprised seven predictors:
\[
  \{\,
    \text{temp},\;\text{precip},\;\text{humidity},\;\text{wind},\;\text{hour},\;\text{dayofweek},\;\text{month}
  \}\,,
\]
together with the \texttt{season} column for downstream diagnostic checks.  Any additional composite categories (e.g.\ Rain \& Warm) were used only in exploratory analyses and were not included in the final modeling pipeline.

All data‐wrangling and feature‐engineering steps are implemented as standalone Python scripts in our repository, which automates dataset download, ingestion, cleaning, resampling, and the creation of these features: \url{https://github.com/neilvmelkot/DivvyBikeAnalysis}.

Next, we derived temporal features from each timestamp, including hour of day (0–23), day of week (0 = Monday to 6 = Sunday), and month (1–12).  We additionally coded a categorical “season” variable; winter (Dec-Feb), spring (Mar–May), summer (Jun–Aug), fall (Sep–Nov), to facilitate seasonal analysis and allow later diagnostic checks of model residuals by meteorological regime.  

We also experimented with composite weather–day categories, with Rain & Cool, No Rain & Cool, Rain & Warm, and No Rain & Warm, to capture potential nonlinear interactions between temperature and precipitation.  Although these composites provided valuable insights during exploratory analysis, the final predictive models used the individual scaled variables listed above, as they delivered equal or better performance with greater interpretability.  

Finally, we fit preliminary Linear, Random Forest, and Gradient Boosting regressors on the training set to rank feature importances.  The resulting importance scores validated our selection of the seven core predictors; temperature, precipitation, humidity, wind speed, hour of day, day of week, and month, and indicated no additional engineered interactions were required for robust model performance.

\section{Exploratory Findings}
\subsection{Continuous Weather Impacts}
\paragraph{Code Snippets}
{\small
\begin{verbatim}
# temp_analysis.py: compute percentiles by temperature
bins = np.linspace(merged["TEMP"].min(), merged["TEMP"].max(), TEMP_BINS)
grp = merged.groupby(pd.cut(merged["TEMP"], bins), observed=True)["ride_count"]
p25 = grp.quantile(0.25)
p50 = grp.quantile(0.50)
p75 = grp.quantile(0.75)
\end{verbatim}
}

{\small
\begin{verbatim}
# humidity_analysis.py: compute percentiles by humidity
bins = np.linspace(merged["humidity"].min(), merged["humidity"].max(), HUMIDITY_BINS)
grp = merged.groupby(pd.cut(merged["humidity"], bins), observed=True)["ride_count"]
p25 = grp.quantile(0.25)
p50 = grp.quantile(0.50)
p75 = grp.quantile(0.75)
\end{verbatim}
}

{\small
\begin{verbatim}
# wind_analysis.py: compute median rides by wind speed
bins = np.linspace(merged["wind"].min(), merged["wind"].max(), WIND_BINS)
grp = merged.groupby(pd.cut(merged["wind"], bins), observed=True)["ride_count"]
median = grp.quantile(0.50)
\end{verbatim}
}

{\small
\begin{verbatim}
# precipitation_analysis.py: compute percentiles by daily precipitation
bins = np.linspace(merged["precip"].min(), merged["precip"].max(), PRCP_BINS)
grp = merged.groupby(pd.cut(merged["precip"], bins), observed=True)["ride_count"]
p50 = grp.quantile(0.50)
\end{verbatim}
}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{rides_vs_temp_percentiles.png}
  \caption{Hourly rides vs. temperature percentiles.}
  \label{fig:rides_vs_temp}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{rides_vs_humidity_percentiles.png}
  \caption{Hourly rides vs. humidity percentiles.}
  \label{fig:rides_vs_humidity_percentiles}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{rides_vs_wind_median.png}
  \caption{Hourly rides vs. wind speed median.}
  \label{fig:rides_vs_wind_median}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{rides_vs_daily_precip_percentiles.png}
  \caption{Daily rides vs. precipitation percentiles.}
\end{figure}

\subsection{Rain and Temperature Categories}
\paragraph{Code Snippet}
{\small
\begin{verbatim}
# daily_precip_temp_trends.py: categorize days by rain & temp
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
df["category"] = np.select(conditions, labels)
\end{verbatim}
}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{daily_rides_rain_temp_bar.png}
  \caption{Average daily rides by rain and temperature condition.}
\end{figure}

\subsection{Temporal Dynamics}
\paragraph{Code Snippet}
{\small
\begin{verbatim}
# heatmap_analysis.py: pivot hourly rides by day & hour
hourly = bikes["ride_id"].resample("h").count()
merged = hourly.to_frame().join(weather, how="inner")
heat = merged.pivot_table(
    values="ride_count",
    index=merged.index.hour,
    columns=merged.index.dayofweek,
    aggfunc="mean"
)
\end{verbatim}
}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{heatmap_hourly_dayofweek.png}
  \caption{Hourly rides by day of week.}
  \label{fig:heatmap_hourly_dayofweek}
\end{figure}

\subsection{Spatial Distribution}
\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{station_density_heatmap_snapshot.png}
  \caption{Total rider volume heatmap.}
  \label{fig:density_map}
\end{figure}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{top10_stations_map_snapshot.png}
  \caption{Top-10 stations map, 5 out of 10 in parks.}
  \label{fig:top10_map}
\end{figure}

\subsection{Monthly Temperature-Ridership Trends}
\paragraph{Code Snippet}
{\small
\begin{verbatim}
# monthly_trends.py: compute monthly avg rides & temps
hourly = bikes["ride_id"].resample("h").count()
monthly_avg_rides = hourly.groupby(hourly.index.month).mean()
gov = pd.read_csv(GOV_TEMP_CSV)
gov.set_index("datetime", inplace=True)
gov_temps = gov["TEMP"].reindex(range(1,13))
df_monthly = pd.DataFrame({
    "month": range(1,13),
    "avg_rides": monthly_avg_rides.values,
    "avg_temp": gov_temps.values
})
\end{verbatim}
}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.8\linewidth]{monthly_riders_temp_comparison_2023.png}
  \caption{Monthly average rides (2023) vs. NOAA normals (1991-2020).}
  \label{fig:monthly_trends}
\end{figure}

\begin{table}[H]
  \centering
  \begin{tabular}{c c c}
    \toprule Month & Avg rides/hr & Avg temp (°C) \\
    \midrule
    1 & 255.78 & -3.78 \\
    2 & 283.40 & -1.78 \\
    3 & 347.69 & 3.89 \\
    4 & 592.49 & 9.83 \\
    5 & 812.94 & 15.89 \\
    6 & 999.47 & 21.44 \\
    7 & 1031.79 & 24.11 \\
    8 & 1037.22 & 23.22 \\
    9 & 925.52 & 19.06 \\
   10 & 721.93 & 12.22 \\  
   11 & 503.50 & 5.17 \\  
   12 & 301.17 & -0.83 \\  
    \bottomrule
  \end{tabular}
  \caption{2023 monthly ridership and temperature.}
  \label{tab:monthly_table}
\end{table}

\section{Predictive Modeling and Performance}
To quantify predictive performance, we trained three models - \textbf{Linear Regression}, \textbf{Random Forest}, and \textbf{Gradient Boosting}, using as inputs temperature, precipitation, humidity, wind speed, hour of day, day of week, and month; each was evaluated on an 80\%/20\% chronological train-test split over 2021-2024, with all predictions clipped at zero to ensure nonnegative ridership forecasts.

\begin{table}[H]
  \centering
  \begin{tabular}{lrrrrr}
    \toprule
    Model & $R^2$ & MSE & MAE & CV mean $R^2$ & CV std $R^2$ \\
    \midrule
    Linear Regression & 0.4904 & $2.58\times10^5$ & 391.8 & 0.4195 & 0.1072 \\
    Random Forest     & 0.9026 & $4.93\times10^4$ & 143.6 & 0.8736 & 0.0483 \\
    Gradient Boosting & 0.8732 & $6.41\times10^4$ & 175.2 & 0.8478 & 0.0495 \\
    \bottomrule
  \end{tabular}
  \caption{Test-set performance and 5-fold cross-validated $R^2$ for each model.}
  \label{tab:model_perf_updated}
\end{table}

% Seasonal performance table
\begin{table}[H]
  \centering
  \begin{tabular}{lrrr}
    \toprule
    Season & LR $R^2$ & RF $R^2$ & GB $R^2$ \\
    \midrule
    Winter (Dec–Feb) & -0.1944 & 0.6211 & 0.5549 \\
    Spring (Mar–May) &  0.4831 & 0.8792 & 0.8667 \\
    Summer (Jun–Aug) &  0.4192 & 0.9037 & 0.8780 \\
    Fall   (Sep–Nov) &  0.4455 & 0.8929 & 0.8446 \\
    \bottomrule
  \end{tabular}
  \caption{Seasonal $R^2$ by model.}
  \label{tab:seasonal_r2}
\end{table}

\begin{table}[H]
  \centering
  \begin{tabular}{lrrrrrr}
    \toprule
    Season & $n_s$ & $w_s$ 
      & $w_s\,R^2_{\rm LR}$ 
      & $w_s\,R^2_{\rm RF}$ 
      & $w_s\,R^2_{\rm GB}$ \\
    \midrule
    Winter (Dec–Feb) &  744 & 0.1159 & -0.0225 & 0.0720 & 0.0643 \\
    Spring (Mar–May) & 1358 & 0.2115 &  0.1022 & 0.1859 & 0.1833 \\
    Summer (Jun–Aug) & 2136 & 0.3326 &  0.1394 & 0.3006 & 0.2920 \\
    Fall   (Sep–Nov) & 2184 & 0.3400 &  0.1515 & 0.3037 & 0.2872 \\
    \midrule
    \textbf{Total}  & 6422 & 1.0000 &  0.3706 & 0.8621 & 0.8268 \\
    \bottomrule
  \end{tabular}
  \caption{Weights and contributions by model.}
  \label{tab:seasonal_weights}
\end{table}

\begin{table}[H]
  \centering
  \begin{tabular}{lr}
    \toprule
    Feature     & Weight      \\
    \midrule
    temp        & 331.2227    \\
    precip      & -17.2324    \\
    humidity    & -185.1993   \\
    wind        & -0.4096     \\
    hour        & 192.3670    \\
    dayofweek   & 28.2271     \\
    month       & -34.5754    \\
    \bottomrule
  \end{tabular}
  \caption{Standardized coefficients for Linear Regression.}
  \label{tab:lr_feature_weights}
\end{table}

\begin{table}[H]
  \centering
  \begin{tabular}{lr}
    \toprule
    Feature     & Importance  \\
    \midrule
    temp        & 0.344609    \\
    precip      & 0.035067    \\
    humidity    & 0.025685    \\
    wind        & 0.017877    \\
    hour        & 0.510206    \\
    dayofweek   & 0.054714    \\
    month       & 0.011842    \\
    \bottomrule
  \end{tabular}
  \caption{Feature importances for Random Forest.}
  \label{tab:rf_feature_weights}
\end{table}

\begin{table}[H]
  \centering
  \begin{tabular}{lr}
    \toprule
    Feature     & Importance  \\
    \midrule
    temp        & 0.358860    \\
    precip      & 0.012179    \\
    humidity    & 0.045605    \\
    wind        & 0.001825    \\
    hour        & 0.536802    \\
    dayofweek   & 0.041956    \\
    month       & 0.002774    \\
    \bottomrule
  \end{tabular}
  \caption{Feature importances for Gradient Boosting.}
  \label{tab:gb_feature_weights}
\end{table}



\section{Discussion}
Temperature stands out as the single most influential predictor, explaining around 90\% of the variance in ridership. As Figure~\ref{fig:monthly_trends} and Table 1 illustrate, average monthly ride counts rise almost linearly with temperature up to the mid‑20\,°C range before plateauing, consistent with human comfort models \cite{Steadman1979}.

\vspace{0.5em}
\[
R(T) = 27.52T + 293.09 \quad (\text{rides/hr}),
\]
\vspace{0.5em}

which not only underscores temperature’s deterministic power but also provides a straightforward rule of thumb for fleet allocation: for every extra degree Celsius, anticipate an additional ~27.5 rides per hour, as per the data retrieved from Figure~\ref{fig:rides_vs_temp}.

\paragraph{Quantifying monthly alignment.} To formalise the intuitive trend seen in Figure~\ref{fig:monthly_trends}, we calculated the Pearson similarity score between the temperature series $\{T_i\}_{i=1}^{12}$ and the monthly ridership series $\{R_i\}_{i=1}^{12}$ presented in Table 1:

\vspace{1.5em}
\[
  r = \frac{\sum_{i=1}^{12}(R_i - \bar R)(T_i - \bar T)}{\sqrt{\sum_{i=1}^{12}(R_i - \bar R)^2}\,\sqrt{\sum_{i=1}^{12}(T_i - \bar T)^2}} \approx 0.9928.
\]
\vspace{1.5em}

The resulting $r^2 \approx 0.9856$ indicates that \textbf{98.56\% of the month‑to‑month variability} in bike demand from 2023 can be strongly correlated to temperature data from the NOAA alone. This quantitative confirmation elevates temperature from a dominant feature in predictive models to the primary operational lever when planning seasonal rebalancing. However, it is important to note that this is not as reliable as it only takes one year of bike riding data, other years may have followed similar but slightly differing trends. 

\paragraph{Precipitation Trends.} Although precipitation correlates weakly overall ($r = -0.12$) as noted in Figure~\ref{fig:corr_heatmap}, its real operational impact manifests in volatility: rainy days exhibit a coefficient of variation of ≈0.25 compared to ≈0.15 on dry days. Moreover, colder rainy days depress ridership by ~23.2\% and warmer rainy days by ~11.4\%, calculated via

\vspace{0.5em}
\[
\Delta_{cool} = \frac{12{,}410.16 - 9{,}535.76}{12{,}410.16} \times 100\%, 
\quad
\Delta_{warm} = \frac{26{,}513.08 - 23{,}490.93}{26{,}513.08} \times 100\%.
\]
\vspace{0.5em}

This asymmetric sensitivity suggests that riders tolerate warm showers much more readily than cold ones, a nuance critical for planning spare‑bike deployments on mixed-weather days.

\paragraph{Humidity and Wind Trends.} 
Individually, humidity and wind speed exhibit negative trends with ridership, as shown in Figures~\ref{fig:rides_vs_humidity_percentiles},~\ref{fig:rides_vs_wind_median}, and~\ref{tab:rf_feature_weights}. For humidity, median hourly ridership peaks at around 1200 rides per hour at 50--55\% relative humidity but drops sharply to approximately 150 rides per hour at 95--100\% humidity, a decline of nearly 87.5\%. This trend likely reflects physical discomfort from muggy conditions, where high humidity impairs sweat evaporation, making cycling feel more strenuous and less appealing, particularly for casual riders \cite{Steadman1979}. Additionally, high humidity often accompanies impending rain, which may deter riders due to perceived weather risks \cite{Kahneman1979}.

Similarly, wind speed shows a consistent decline in ridership as speeds increase. Median hourly ridership starts at approximately 734 rides per hour at low wind speeds (0.6 mph) but falls to around 209 rides per hour at 10.7 mph, a reduction of about 71.5\%. Stronger winds increase the physical effort required to cycle, especially against headwinds, and may raise safety concerns due to reduced bike stability, particularly for less experienced riders. Gusty conditions can also carry dust or debris in urban settings, further discouraging ridership. These individual effects highlight why humidity and wind, though secondary to temperature, are critical for accurate demand forecasting.

\paragraph{Temporal Trends.} 
Temporal segmentation reveals distinct weekly and daily rhythms in bike-sharing demand, driven by commuter and recreational patterns, as visualized in Figure 6. On weekdays (Monday = 0 to Friday = 4), the busiest times are consistently during morning (7-9 AM) and evening (4-6 PM) rush hours, reflecting a 5-day work-life trend. Peak ridership occurs on Wednesday at 5 PM with 1818.83 rides per hour, followed closely by Tuesday at 5 PM (1758.22 rides) and Monday at 5 PM (1579.32 rides). Morning peaks are slightly lower, with Tuesday at 8 AM averaging 1018.50 rides and Wednesday at 8 AM at 1028.04 rides. These patterns align with typical urban commuting schedules, where workers use bike-sharing for first- and last-mile connections to offices in the Central Business District. The consistency of these dual peaks from Monday to Friday suggests that the primary demographic of weekday Divvy riders is likely working-class individuals, such as office workers, who rely on bike-sharing for daily commutes. Students, retirees, or other groups with more flexible schedules (e.g., part-time workers) are less likely to drive such pronounced rush-hour spikes, as their travel patterns would be more evenly distributed throughout the day or concentrated during off-peak academic or leisure hours.

Fridays exhibit a distinct pattern compared to Monday-Thursday, with fewer riders during the morning rush hour and a lower evening peak, alongside an earlier onset of elevated activity starting from midday. For instance, Friday’s morning peak at 8 AM is only 718.59 rides, significantly lower than Tuesday’s 1018.50 or Wednesday’s 1028.04 rides, suggesting reduced commuting demand, possibly due to flexible work arrangements, remote work, or employees taking partial days off as the weekend approaches. Similarly, the evening peak at 5 PM on Friday (1504.58 rides) is notably lower than Monday (1579.32), Tuesday (1758.22), Wednesday (1818.83), or Thursday (1734.25), perhaps as workers leave earlier or engage in non-commute activities. However, Friday’s activity ramps up earlier in the day, with 873.67 rides at 12 PM compared to 752.55 on Monday, 698.10 on Tuesday, or 723.80 on Wednesday, and continues to rise steadily through the afternoon (e.g., 904.20 rides at 1 PM vs. 765.64 on Monday; 937.84 rides at 2 PM vs. 784.09 on Monday; 1121.89 rides at 3 PM vs. 937.76 on Monday). This earlier surge from 12 PM to 3 PM, before the 4 PM peak of 1351.76 rides, reflects a shift toward midday and early afternoon usage, potentially driven by workers taking lunch breaks, running errands, or starting weekend leisure activities earlier, particularly in warmer months when ridership is higher (Table 1). The earlier start and sustained activity suggest Friday serves as a transitional day, blending commuting with preparatory or social trips ahead of the weekend.

Weekends (Saturday = 5, Sunday = 6) display a markedly different temporal profile, with peak ridership shifting to midday and early afternoon hours (11 AM-3 PM). Saturday peaks at 1 PM with 1358.27 rides per hour, and Sunday at 2 PM with 1235.74 rides, contrasting with the sharp morning and evening spikes on weekdays. Weekend ridership remains elevated from 9 AM to 6 PM, with Saturday averaging over 994.34 rides per hour from 10 AM to 4 PM and Sunday exceeding 1064.96 rides per hour from 11 AM to 3 PM. These trends reflect recreational usage, likely driven by leisure activities such as cycling in parks (Figure~\ref{fig:top10_map}) or exploring downtown attractions. The broader temporal spread on weekends suggests users have more flexible schedules, unlike the constrained commuting windows on weekdays. The high weekend midday demand also aligns with favorable weather conditions, as afternoons typically offer warmer temperatures (Table 1), encouraging outdoor activities.

The 5-day work-life trend is evident in the consistent weekday pattern of dual peaks, which abruptly shifts on weekends to a single, prolonged midday peak. This dichotomy underscores the dual role of bike-sharing as both a commuter tool and a recreational vehicle. The demographic inference of working-class riders on weekdays is further supported by the spatial concentration of rides in the Central Business District during rush hours (Figure~\ref{fig:density_map}), where office workers are predominant, as opposed to campus areas or residential neighborhoods that might see more student activity. Embedding these temporal regimes into rebalancing algorithms could reduce shortage events in simulations, as dynamic fleet allocation can prioritize stations near business districts during weekday rush hours and park-adjacent stations (Figure~\ref{fig:top10_map}) during weekend afternoons. Such strategies would mitigate customer frustration from bike or dock unavailability and lower operational costs by optimizing rebalancing efforts.

\paragraph{Spatial Distribution.} 
Spatially, the Central Business District concentrates the majority of Divvy ridership (Figure~\ref{fig:density_map}), reinforcing the value of dynamic station rebalancing in high-density zones. Simultaneously, five of the top ten busiest stations are park-adjacent (Figure~\ref{fig:top10_map}), illustrating how recreational usage underpins demand outside commute hours. This dual dependency on workday flows and leisure trips creates a more resilient ridership base and supports downtown economic vitality.

\paragraph{Predictive Models.} 
All three models capture a substantial fraction of variance in hourly ridership, but their capacity to model nonlinear interactions differs.  Recall the definition of the coefficient of determination on a test set of size \(N\):
\begin{equation}
  R^2 \;=\; 1 \;-\; \frac{\displaystyle \sum_{i=1}^N (y_i - \hat y_i)^2}{\displaystyle \sum_{i=1}^N (y_i - \bar y)^2}
  \label{eq:global-r2}
\end{equation}
where \(y_i\) are the true ride counts, \(\hat y_i\) the predictions, and \(\bar y\) the mean of the test targets.

\medskip
The \textbf{Random Forest} achieved
\[
  R^2_{\rm test} = 0.9026,
  \qquad
  \overline{R^2}_{\rm CV} = 0.8736 \;(\mathrm{std}=0.0483),
\]
reflecting its strength at automatically modeling nonlinear effects of temperature, humidity, wind speed, and temporal covariates.  In contrast, Linear Regression underfits complex seasonal and weather‐driven patterns, yielding a negative winter \(R^2\) (i.e.\ worse than using the mean) and substantially larger residuals.  Table~\ref{tab:model_perf_updated} summarizes these global performance metrics.

\medskip
To understand which inputs drive each model’s predictions, we examine the learned feature weights:
\begin{itemize}
  \item \textbf{Linear Regression} standardized coefficients are given in Table~\ref{tab:lr_feature_weights}.
  \item \textbf{Random Forest} feature importances are shown in Table~\ref{tab:rf_feature_weights}.
  \item \textbf{Gradient Boosting} feature importances are shown in Table~\ref{tab:gb_feature_weights}.
\end{itemize}

\medskip
An analysis of the coefficient patterns in Table~\ref{tab:lr_feature_weights} shows that \emph{temperature} has the largest positive standardized coefficient (331.2), indicating that a one-standard-deviation increase in temperature predicts roughly 331 additional rides, underscoring the strong sensitivity of ridership to ambient warmth.  The \emph{hour of day} coefficient (192.4) is similarly large, reflecting pronounced commuter peaks.  Negative coefficients for \emph{humidity} (–185.2) and \emph{precipitation} (–17.2) confirm that muggy conditions and rain deter riders, while the near-zero coefficient for \emph{wind} (–0.4) suggests minimal linear impact after accounting for other factors.  

By contrast, the tree-based models in Tables~\ref{tab:rf_feature_weights} and~\ref{tab:gb_feature_weights} assign highest importance to \emph{hour of day} (51.0\% in RF, 53.7\% in GB) and \emph{temperature} (34.5\% in RF, 35.9\% in GB), indicating these variables are most frequently used in decision splits to reduce error.  Lower importances for \emph{month}, \emph{wind}, and \emph{humidity} reflect their more subtle nonlinear contributions.  These weight patterns mirror intuitive dynamics: diurnal commute patterns dominate, temperature strongly modulates demand, and secondary weather or seasonal covariates play smaller roles.

\medskip
To diagnose performance by season, we compute a season-specific \(R^2\) on the test subset belonging to each meteorological regime:
\begin{equation}
  R^2_{s} \;=\; 1 \;-\; \frac{\displaystyle \sum_{i\in\mathcal{I}_s} (y_i - \hat y_i)^2}
                                  {\displaystyle \sum_{i\in\mathcal{I}_s} (y_i - \bar y_s)^2}
  \quad\text{for}\;s\in\{\mathrm{winter},\mathrm{spring},\mathrm{summer},\mathrm{fall}\},
  \label{eq:seasonal-r2}
\end{equation}
where \(\mathcal{I}_s\) indexes test‐set hours in season \(s\), \(\bar y_s\) is the mean rides in that season, and \(n_s = |\mathcal{I}_s|\) is the number of hourly buckets in the held‐out test split (total \(\sum_s n_s = 6422\)).

\medskip
We then weight each seasonal \(R^2_s\) by its share of the test data:
\begin{align}
  w_s &= \frac{n_s}{\sum_{t} n_t}, 
  &R^2_{\rm weighted} &= \sum_{s} w_s\,R^2_{s}.
  \label{eq:weighted-r2}
\end{align}

\medskip
Table~\ref{tab:seasonal_r2} summarizes the per‐season \(R^2_s\) for each model, and Table~\ref{tab:seasonal_weights} reports the counts \(n_s\), weights \(w_s\), and their contributions \(w_s\,R^2_s\) to the weighted average.

\medskip
Seasonal breakdown for Random Forest:
\begin{itemize}
  \item \textbf{Winter (Dec--Feb)} (\(n=744\), \(R^2=0.6211\), \(w=0.1159\))
  \item \textbf{Spring (Mar--May)} (\(n=1358\), \(R^2=0.8792\), \(w=0.2115\))
  \item \textbf{Summer (Jun--Aug)} (\(n=2136\), \(R^2=0.9037\), \(w=0.3326\))
  \item \textbf{Fall (Sep--Nov)}   (\(n=2184\), \(R^2=0.8929\), \(w=0.3400\))
\end{itemize}

\medskip
These results motivate \emph{seasonally adaptive rebalancing} of the fleet:
\begin{itemize}
  \item Exploit high summer accuracy for aggressive station restocking when demand peaks.
  \item Incorporate holiday calendars, local event schedules, and station‐level microclimate measures to boost winter and holiday‐period forecasts where the model currently struggles.
  \item Consider a hybrid approach that blends Random Forest with simpler linear corrections around known anomalies (e.g.\ major city events).
\end{itemize}


\section{Conclusions and Future Work}
Our integrated analysis confirms that weather and temporal factors critically shape bike\‑sharing demand. Temperature drives the bulk of variance, with humidity, wind, and precipitation as secondary modifiers. We recommend seasonally tailored service levels, regime\‑based rebalancing, and strategic station placement near both CBD and recreational areas. Future research will integrate real\‑time event feeds, demographic overlays, and dynamic pricing to further elevate model fidelity and operational efficiency.


\newpage
\begin{thebibliography}{9}
\bibitem{Oke1982}T.R. Oke. The energetic basis of the urban heat island. \emph{Quarterly Journal of the Royal Meteorological Society}, 108(455), 1--24, 1982.
\bibitem{Steadman1979}R.G. Steadman. The assessment of sultriness. Part I: A temperature-humidity index based on human physiology and clothing science. \emph{Journal of Applied Meteorology}, 18(7), 861--875, 1979.
\bibitem{Kahneman1979}D. Kahneman and A. Tversky. Prospect theory: An analysis of decision under risk. \emph{Econometrica}, 47(2), 263--291, 1979.
\bibitem{Vuchic2007}V.R. Vuchic. \emph{Urban Transit Systems and Technology}. Wiley, 2007.
\bibitem{NOAA2024}NOAA. Chicago O'Hare monthly normals (1991--2020). \emph{NOAA NCEI}. Retrieved from \url{https://www.weather.gov/lot/ord_rfd_monthly_yearly_normals}, 2024.
\bibitem{TPLChicago}The Trust for Public Land. Chicago Parks and Recreation. Retrieved from \url{https://tpl.org/city/chicago-illinois}, 2021.
\bibitem{KaggleWeather}Curiel, "Data Mining Weather Chicago." Kaggle. \url{https://www.kaggle.com/code/curiel/data-mining-weather-chicago}, accessed 2024.
\bibitem{DivvyData}Divvy Bikes. System Data portal. \url{https://divvybikes.com/system-data}, accessed 2024.
\end{thebibliography}

\end{document}
"""

with open(TEX_PATH, "w", encoding="utf-8") as f:
    f.write(tex)

pdflatex_cmd = shutil.which("pdflatex") or shutil.which("pdflatex.exe")
if not pdflatex_cmd and os.name == 'nt':
    common_paths = [
        r"C:\\Program Files\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
        r"C:\\Program Files (x86)\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "pdftex", "config", "pdflatex.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "MiKTeX", "miktex", "bin", "x64", "pdflatex.exe")
    ]
    for p in common_paths:
        if os.path.isfile(p):
            pdflatex_cmd = p
            break

if not pdflatex_cmd:
    print("Error: 'pdflatex' not found. Please ensure MiKTeX is installed and 'pdflatex.exe' is on your PATH or in a standard MiKTeX folder.")
    sys.exit(1)

try:
    subprocess.run([pdflatex_cmd, "-interaction=nonstopmode", "-output-directory", OUTPUT_DIR, TEX_PATH], check=True)
    print(f"Report generated at {os.path.join(OUTPUT_DIR, PDF_NAME)}")
except subprocess.CalledProcessError as e:
    print(f"LaTeX compilation failed with return code {e.returncode}.")
    sys.exit(e.returncode)
