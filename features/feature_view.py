import hopsworks

def create_feature_view():
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Get feature group
    fg = fs.get_feature_group(
        name="air_quality_features",
        version=1
    )

    # Create feature view
    fv = fs.get_or_create_feature_view(
        name="air_quality_fv",
        version=1,
        query=fg.select_all(),
        description="Feature view for AQI prediction"
    )

    print("✅ Feature View created successfully")

if __name__ == "__main__":
    create_feature_view()