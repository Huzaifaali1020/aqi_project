import matplotlib.pyplot as plt
from storage.hopsworks_fs import read_features

df = read_features()

df.set_index("timestamp")["aqi"].plot(title="AQI Trend")
plt.show()