***Karachi AQI Predictor — End-to-End MLOps Pipeline***

Live Dashboard: https://aqiproject-6rqnaxgtnz7bbizdkr2ylt.streamlit.app/

Developed by: Syed Huzaifa Ali — 10Pearls Data Science Internship 2026

**Overview--**
Karachi AQI Predictor is a fully automated, production-grade machine learning system that predicts Air Quality Index for Karachi, Pakistan up to 72 hours ahead. 
The system collects real-time pollution and weather data every hour, engineers features automatically, retrains machine learning models daily, and serves predictions through an interactive Streamlit dashboard all without any manual intervention.

**For a better understanding of the visualizations, please refer to the EDA file and its corresponding plots file. Similarly, for SHAP-based explanations, open the explainability file and then the plots file to explore the visual outputs.**

**PAGE 1**
<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/f244fe07-39cc-4dd1-9efb-32f16c3bd76d" />
**PAGE 2**
<img width="1907" height="692" alt="image" src="https://github.com/user-attachments/assets/e14a9b46-4e0c-4db9-b4bd-b7879bff6970" />
<img width="1877" height="776" alt="image" src="https://github.com/user-attachments/assets/5bac1418-201d-4e06-93e3-978195c65014" />
**PAGE 3**
<img width="1917" height="845" alt="image" src="https://github.com/user-attachments/assets/c90fb988-5dd5-4d7a-88fb-623ca110f165" />
<img width="1919" height="900" alt="image" src="https://github.com/user-attachments/assets/18c38fe0-cf0b-4eec-805c-f323b930d30d" />
**PAGE 4**
<img width="1888" height="792" alt="image" src="https://github.com/user-attachments/assets/ffadd3b7-52c6-499e-998a-29222f423498" />
<img width="1916" height="683" alt="image" src="https://github.com/user-attachments/assets/acb80b28-0fd2-414b-9226-5a9ea4200966" />
<img width="1909" height="730" alt="image" src="https://github.com/user-attachments/assets/c5c7d778-0c43-4ba3-847f-e53fd820bb47" />
<img width="1883" height="795" alt="image" src="https://github.com/user-attachments/assets/4695a40d-e7e2-4841-a8fc-c59e03b2f609" />
<img width="1909" height="892" alt="image" src="https://github.com/user-attachments/assets/564e90f7-3f58-4dd9-9ffd-be2b0a5dfc38" />
<img width="1900" height="621" alt="image" src="https://github.com/user-attachments/assets/3351ac1b-266b-4a72-85eb-3f59491a66f3" />
<img width="1886" height="662" alt="image" src="https://github.com/user-attachments/assets/4a685da7-dd9a-4998-9a3b-01058f59143e" />
**PAGE 5**
<img width="1875" height="834" alt="image" src="https://github.com/user-attachments/assets/2289cc5d-a041-4ce5-9a89-046fde208448" />
<img width="1908" height="910" alt="image" src="https://github.com/user-attachments/assets/ceb8e91c-05e6-4640-9c53-99836f50bd61" />
**PAGE 6**
<img width="1915" height="891" alt="image" src="https://github.com/user-attachments/assets/5cb6066c-c10f-45ed-a0e0-83ce219a918c" />
<img width="1874" height="835" alt="image" src="https://github.com/user-attachments/assets/525f2069-6306-40d3-8c89-dd3ab58869ff" />




**Live Dashboard--**
The dashboard is deployed on Streamlit Cloud and updates automatically every hour. 
It includes six pages covering current conditions, 72-hour forecast, model performance metrics, exploratory data analysis, SHAP explainability, and project information.

**Project Architecture--**
The system runs in two automated loops triggered by GitHub Actions.
The hourly loop runs every hour at minute zero. It fetches air pollution data from the OpenWeather API and weather data from the Open-Meteo API, inserts one new row into the Hopsworks Feature Store, runs feature engineering to compute 18 engineered features, saves the latest features to a CSV file, and commits it to the GitHub repository so Streamlit Cloud always has fresh data.
The daily loop runs every day at 2:00 AM UTC. It loads all accumulated data from the Hopsworks Feature View, trains three machine learning models using a time-based 80/20 split, evaluates each model and selects the best by RMSE, saves the winning model to the Hopsworks Model Registry with full lineage tracking, runs a 72-hour recursive forecast, and commits the prediction CSV files back to the repository.

**Tech Stack--**
Data Collection: OpenWeather Air Pollution API, Open-Meteo Archive API
Feature Store and Model Registry: Hopsworks (eu-west free tier)
Machine Learning: scikit-learn (Ridge Regression, Random Forest), XGBoost
Explainability: SHAP (SHapley Additive exPlanations)
Dashboard: Streamlit, Plotly, Leaflet.js (live location map)
CI/CD Automation: GitHub Actions (hourly + daily scheduled workflows)
Deployment: Streamlit Cloud
Language: Python 3.10

**Features Engineered (18 total)--**
The raw pollution and weather measurements are transformed into 18 features before model training.
Time features capture when in the day and week the measurement was taken, since Karachi pollution follows strong daily cycles tied to rush hour traffic. These include hour, day, month, day of week, and a weekend indicator.
Pollution lag features capture what PM2.5 and PM10 levels were one and three hours ago, which is the strongest available signal for near-future air quality. These include pm25_lag_1h, pm25_lag_3h, pm10_lag_1h, and pm10_lag_3h.
Pollution rolling features smooth out noise by averaging recent measurements. These include pm25_roll_3h, pm25_roll_6h, and pm10_roll_3h.
Weather lag and rolling features capture how temperature, humidity, and wind speed have been changing, since these meteorological conditions strongly influence how pollutants disperse or concentrate. These include temp_lag_1h, humidity_lag_1h, wind_lag_1h, temp_roll_3h, humidity_roll_3h, and wind_roll_3h.

**Model Performance--**
Best Model: XGBoost (aqi_predictor v77)
Three regression models are trained daily and compared automatically. The best model is selected by RMSE and saved to the Hopsworks Model Registry.
XGBoost consistently achieves the best performance and is currently running as version 77 in production with RMSE of 3.75, MAE of 2.34, and R-squared of 0.9942 for 6-hour ahead prediction. This means the model is off by roughly 4 AQI units on average and explains 99.4 percent of the variance in AQI values — a significant improvement from earlier versions as more training data accumulated over time.
Random Forest achieved RMSE of 11.81 and R-squared of 0.9451. Ridge Regression achieved RMSE of 22.10 and R-squared of 0.8080 and serves as a linear baseline.

**72-Hour Recursive Forecasting--**
The model is trained to predict AQI six hours ahead. To generate a 72-hour forecast, the system runs 12 prediction steps recursively. It predicts AQI at T+6 using current features, then updates the lag features with that prediction, then predicts T+12, and so on until T+72. This produces 12 forecast points at 6-hour intervals which are then aggregated into a 3-day daily summary showing average, minimum, and maximum predicted AQI per day.

**Key Technical Challenges Solved--**
Several significant technical challenges were identified and resolved during development.
Data leakage was the most critical issue. Initially the model achieved RMSE of zero because the OpenWeather API returns AQI on a 1-5 European scale rather than the 0-500 EPA scale. All training rows had constant AQI=3, so the model learned to always predict 3. The fix was computing AQI from PM2.5 using EPA breakpoints. Additional leakage from raw pollutant columns being included as features was also eliminated, and the prediction horizon was changed from 1 hour to 6 hours to make the task genuinely challenging.
Historical weather data was unavailable because the OpenWeather history API requires a paid subscription. The fix was switching to the Open-Meteo Archive API which provides free historical hourly weather data going back to 1940 with no API key required.
Hopsworks Spark materialization outage affected all students on the eu-west free tier from May 21, 2026. Spark materialization jobs were failing with FailedScheduling errors indicating zero nodes available. The fix was removing event_time from feature group definitions and setting wait_for_job to False, which stopped Spark jobs from being triggered while leaving all required pipeline functionality intact.
GitHub Actions missing confluent-kafka dependency caused insert operations to fail silently for several days. The fix was adding confluent-kafka to requirements.txt.
Streamlit Cloud auto-refreshing forecast was causing the dashboard to regenerate predictions on the cloud server where Hopsworks connections returned empty data, producing AQI values of 1.0. The fix was detecting when the app runs on Streamlit Cloud and skipping the auto-refresh entirely, relying instead on GitHub Actions to commit fresh CSVs to the repository daily.

**Project Structure--**

<img width="807" height="494" alt="image" src="https://github.com/user-attachments/assets/0a6ae5d2-b4b3-4700-a5b1-9c7c4f22a6bb" />
<img width="728" height="563" alt="image" src="https://github.com/user-attachments/assets/564c8e52-009a-4334-921a-788f066a1d39" />


**GitHub Actions Workflows--**
Hourly Feature Pipeline runs at minute 0 of every hour. It fetches new pollution and weather data, inserts into Hopsworks Feature Store, runs feature engineering, saves the latest 50 rows to latest_features.csv, and commits to GitHub.
Daily Model Training Pipeline runs at 2:00 AM UTC every day. It retrains all three models on accumulated data, saves the best model to the Hopsworks Model Registry, generates a fresh 72-hour forecast, and commits the prediction CSV files to GitHub.
Both workflows generate config.yaml from GitHub Secrets at runtime so no sensitive credentials are ever stored in the repository.

**EDA Key Findings--**
Analysis of over 3,800 hourly observations from December 2025 to present revealed several clear patterns. The average AQI across the full period is approximately 95, with December being the worst month at around 140 due to winter temperature inversions, and May being the cleanest at around 85. Clear rush hour peaks appear at 7 to 9 AM and 5 to 7 PM corresponding to Karachi traffic patterns. Weekdays are consistently 8 to 15 percent more polluted than weekends. Wind speed shows the strongest negative correlation with AQI at approximately minus 0.25, confirming that wind dispersal is a key factor in Karachi air quality.

**SHAP Feature Importance--**
SHAP analysis revealed that the five most influential features for AQI prediction are pm25_lag_1h, pm10_lag_1h, pm25_roll_3h, hour of day, and wind_lag_1h. Recent PM2.5 and PM10 values being most important makes intuitive sense since recent pollution is the strongest signal for near-future air quality. Hour of day being important confirms that rush hour patterns are a significant driver captured by the model.

**AQI Scale Reference--**
Good (0-50) means no health impact. Moderate (51-100) means very sensitive people may be affected. Unhealthy for Sensitive Groups (101-150) means sensitive groups should reduce outdoor activity. Unhealthy (151-200) means everyone may be affected. Very Unhealthy (201-300) is a health alert for everyone. Hazardous (301-500) means emergency conditions affecting the entire population.

**Data Sources--**
Pollution measurements including PM2.5, PM10, CO, NO, NO2, O3, SO2, and NH3 come from the OpenWeather Air Pollution API. Weather measurements including temperature, humidity, wind speed, and pressure come from the Open-Meteo API which is completely free with no API key required. Historical data goes back to December 1, 2025 and the dataset grows by one row every hour.

**HOPSWORK**

**FEATURES GROUP**

<img width="1914" height="925" alt="image" src="https://github.com/user-attachments/assets/6e4dff0c-6f4d-4788-82dd-37c0897cc8d3" />
<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/348882d7-66a8-464b-933b-09a746639fe6" />
<img width="1883" height="955" alt="image" src="https://github.com/user-attachments/assets/ec603b16-7d3f-4bb9-b18d-6da211a185b8" />
<img width="1919" height="961" alt="image" src="https://github.com/user-attachments/assets/a5078f61-6cd4-437f-b3d4-902d79909a3e" />


**JOBS**

<img width="1886" height="903" alt="image" src="https://github.com/user-attachments/assets/73cdc9b3-6be3-4f4b-b29f-713e5690ddc1" />
<img width="1901" height="851" alt="image" src="https://github.com/user-attachments/assets/ba846dc9-9a52-419a-a735-0821b5a7e067" />
<img width="1919" height="945" alt="image" src="https://github.com/user-attachments/assets/8244a628-22eb-4ca8-a753-c096f1d839ca" />


**MODEL REGISTRY**

<img width="1913" height="892" alt="image" src="https://github.com/user-attachments/assets/9becf046-b9a4-4908-98cc-a8d6af6fbbf7" />
<img width="1915" height="937" alt="image" src="https://github.com/user-attachments/assets/f290fa14-929f-4a94-b3d4-69948b11fbb0" />

