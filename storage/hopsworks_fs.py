import hopsworks

def write_features(df):
    project = hopsworks.login()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        description="Hourly AQI + weather features"
    )

    fg.insert(df)