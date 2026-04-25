from ingestion.fetch_data import fetch_data
from storage.hopsworks_fs import write_features

def run_feature_pipeline():
    df = fetch_data()
    write_features(df)
    print("✅ Features stored in Hopsworks")

if __name__ == "__main__":
    run_feature_pipeline()