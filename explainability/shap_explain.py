import shap
from storage.model_registry import load_model
from storage.hopsworks_fs import read_features

model = load_model()
df = read_features().drop(columns=["aqi", "timestamp"])

explainer = shap.Explainer(model, df)
shap_values = explainer(df)

shap.summary_plot(shap_values, df)