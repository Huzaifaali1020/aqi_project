import streamlit as st
from storage.hopsworks_fs import read_features
from storage.model_registry import load_model
from alerts.email_alerts import send_alert

st.title("Karachi AQI Forecast")

df = read_features()
model = load_model()

latest = df.iloc[-1:]
X = latest.drop(columns=["aqi", "timestamp"])

aqi = model.predict(X)[0]
st.metric("Current AQI", int(aqi))

if aqi > 150:
    st.error("Hazardous AQI")
    send_alert(int(aqi))

st.line_chart(df.set_index("timestamp")["aqi"])