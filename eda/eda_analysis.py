import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hopsworks
import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

PLOT_DIR = os.path.join(BASE_DIR, "eda", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

plt.style.use("dark_background")


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data() -> pd.DataFrame:
    project = hopsworks.login(
        host=config["hopsworks"]["host"],
        api_key_value=config["hopsworks"]["api_key"]
    )
    fs = project.get_feature_store()
    fv = fs.get_feature_view("aqi_features_fv", version=1)
    df = fv.get_batch_data()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"]  = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"]       = df["timestamp"].dt.hour
    df["day_name"]   = df["timestamp"].dt.day_name()
    df["month_name"] = df["timestamp"].dt.strftime("%b")
    df["month_num"]  = df["timestamp"].dt.month
    df["date"]       = df["timestamp"].dt.date
    print(f"✅ Loaded {len(df)} rows")
    print(f"📅 {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


# --------------------------------------------------
# 1. AQI Distribution
# --------------------------------------------------
def plot_aqi_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AQI Distribution — Karachi", fontsize=14, fontweight="bold")

    axes[0].hist(df["aqi"], bins=40, color="#3498db", edgecolor="none", alpha=0.8)
    axes[0].axvline(df["aqi"].mean(),   color="#e74c3c", linestyle="--",
                    label=f"Mean: {df['aqi'].mean():.1f}")
    axes[0].axvline(df["aqi"].median(), color="#2ecc71", linestyle="--",
                    label=f"Median: {df['aqi'].median():.1f}")
    axes[0].set_xlabel("AQI")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("AQI Histogram")
    axes[0].legend()

    categories = [
        (0,   50,  "#2ecc71", "Good\n(0-50)"),
        (50,  100, "#f1c40f", "Moderate\n(51-100)"),
        (100, 150, "#e67e22", "Sensitive\n(101-150)"),
        (150, 200, "#e74c3c", "Unhealthy\n(151-200)"),
        (200, 300, "#8e44ad", "Very Unhealthy\n(201-300)"),
        (300, 500, "#2c3e50", "Hazardous\n(301+)"),
    ]
    counts = [((df["aqi"] >= lo) & (df["aqi"] < hi)).sum()
              for lo, hi, _, __ in categories]
    labels = [label for _, __, ___, label in categories]
    colors = [color for _, __, color, ___ in categories]

    axes[1].bar(labels, counts, color=colors, edgecolor="none")
    axes[1].set_xlabel("AQI Category")
    axes[1].set_ylabel("Hours")
    axes[1].set_title("Hours per AQI Category")
    axes[1].tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "1_aqi_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 2. AQI Over Time
# --------------------------------------------------
def plot_aqi_over_time(df):
    daily = df.groupby("date")["aqi"].mean().reset_index()
    daily["date"]      = pd.to_datetime(daily["date"])
    daily["rolling7"]  = daily["aqi"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.fill_between(daily["date"], daily["aqi"], alpha=0.3, color="#3498db")
    ax.plot(daily["date"], daily["aqi"],
            color="#3498db", linewidth=0.8, alpha=0.6, label="Daily avg AQI")
    ax.plot(daily["date"], daily["rolling7"],
            color="#e74c3c", linewidth=2, label="7-day rolling avg")

    for level, label, color in [
        (50,  "Good",      "#2ecc71"),
        (100, "Moderate",  "#f1c40f"),
        (150, "Sensitive", "#e67e22"),
    ]:
        ax.axhline(level, color=color, linestyle=":", alpha=0.5, linewidth=1)
        ax.text(daily["date"].iloc[-1], level + 2,
                label, color=color, fontsize=8)

    ax.set_title("AQI Over Time — Karachi (Dec 2025 → Present)", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("AQI")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "2_aqi_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 3. Hourly Patterns
# --------------------------------------------------
def plot_hourly_patterns(df):
    hourly = df.groupby("hour")["aqi"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(
        hourly["hour"],
        hourly["mean"] - hourly["std"],
        hourly["mean"] + hourly["std"],
        alpha=0.2, color="#3498db", label="±1 std"
    )
    ax.plot(hourly["hour"], hourly["mean"],
            color="#3498db", linewidth=2.5, marker="o", markersize=5)

    for rh in [7, 8, 9, 17, 18, 19]:
        ax.axvline(rh, color="#e74c3c", linestyle="--", alpha=0.3)

    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day (UTC)")
    ax.set_ylabel("Average AQI")
    ax.set_title("Average AQI by Hour of Day", fontsize=13)
    ax.text(8, hourly["mean"].max() * 0.95,
            "Rush hours", color="#e74c3c", fontsize=9)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "3_hourly_patterns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 4. Weekly Patterns
# --------------------------------------------------
def plot_weekly_patterns(df):
    day_order  = ["Monday", "Tuesday", "Wednesday",
                  "Thursday", "Friday", "Saturday", "Sunday"]
    day_colors = ["#3498db"] * 5 + ["#2ecc71"] * 2

    weekly = (df.groupby("day_name")["aqi"]
                .mean()
                .reindex(day_order)
                .reset_index())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(weekly["day_name"], weekly["aqi"],
                  color=day_colors, edgecolor="none")

    for bar, val in zip(bars, weekly["aqi"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average AQI")
    ax.set_title("Average AQI by Day of Week", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "4_weekly_patterns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 5. Correlation Heatmap
# --------------------------------------------------
def plot_correlation_heatmap(df):
    feature_cols = [
        "aqi", "pm25", "pm10", "co", "no2", "o3",
        "temperature", "humidity", "wind_speed", "pressure",
        "pm25_lag_1h", "pm25_lag_3h", "pm10_lag_1h",
        "temp_lag_1h", "humidity_lag_1h", "wind_lag_1h",
        "aqi_next_hour"
    ]
    cols = [c for c in feature_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "5_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 6. Seasonal Patterns
# --------------------------------------------------
def plot_seasonal_patterns(df):
    month_order = ["Dec", "Jan", "Feb", "Mar", "Apr", "May",
                   "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]
    month_order = [m for m in month_order
                   if m in df["month_name"].unique()]

    monthly = (df.groupby("month_name")["aqi"]
                 .agg(["mean", "min", "max"])
                 .reindex(month_order)
                 .reset_index())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(range(len(monthly)),
                    monthly["min"], monthly["max"],
                    alpha=0.2, color="#3498db", label="Min-Max range")
    ax.plot(range(len(monthly)), monthly["mean"],
            color="#3498db", linewidth=2.5,
            marker="o", markersize=6, label="Monthly avg")

    for i, (_, row) in enumerate(monthly.iterrows()):
        ax.text(i, row["mean"] + 1, f"{row['mean']:.0f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(monthly)))
    ax.set_xticklabels(monthly["month_name"])
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI")
    ax.set_title("AQI Seasonal Patterns by Month", fontsize=13)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "6_seasonal_patterns.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# 7. Weather vs AQI
# --------------------------------------------------
def plot_weather_vs_aqi(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Weather vs AQI — Karachi", fontsize=13, fontweight="bold")

    weather_features = [
        ("temperature", "Temperature (°C)",  "#e74c3c"),
        ("humidity",    "Humidity (%)",       "#3498db"),
        ("wind_speed",  "Wind Speed (m/s)",   "#2ecc71"),
        ("pressure",    "Pressure (hPa)",     "#9b59b6"),
    ]

    for ax, (col, label, color) in zip(axes.flatten(), weather_features):
        if col not in df.columns:
            continue
        sample = df[[col, "aqi"]].dropna().sample(
            min(2000, len(df)), random_state=42
        )
        ax.scatter(sample[col], sample["aqi"],
                   alpha=0.3, s=8, color=color)
        z  = np.polyfit(sample[col], sample["aqi"], 1)
        p  = np.poly1d(z)
        xs = np.linspace(sample[col].min(), sample[col].max(), 100)
        ax.plot(xs, p(xs), color="white", linewidth=2, linestyle="--")
        corr = sample[col].corr(sample["aqi"])
        ax.set_xlabel(label)
        ax.set_ylabel("AQI")
        ax.set_title(f"{label} vs AQI  (r={corr:.3f})")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "7_weather_vs_aqi.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {path}")


# --------------------------------------------------
# Summary statistics
# --------------------------------------------------
def print_summary(df):
    print("\n" + "=" * 55)
    print("📊 EDA SUMMARY — KARACHI AQI")
    print("=" * 55)
    print(f"Total hours  : {len(df)}")
    print(f"Date range   : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"AQI mean     : {df['aqi'].mean():.1f}")
    print(f"AQI median   : {df['aqi'].median():.1f}")
    print(f"AQI std      : {df['aqi'].std():.1f}")
    print(f"AQI min      : {df['aqi'].min()}")
    print(f"AQI max      : {df['aqi'].max()}")
    print(f"Worst hour   : {df.groupby('hour')['aqi'].mean().idxmax()}:00")
    print(f"Best hour    : {df.groupby('hour')['aqi'].mean().idxmin()}:00")
    print(f"Worst day    : {df.groupby('day_name')['aqi'].mean().idxmax()}")
    print(f"Best day     : {df.groupby('day_name')['aqi'].mean().idxmin()}")
    print("=" * 55)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting EDA\n")
    df = load_data()
    print_summary(df)
    print("\n📊 Generating plots...")
    plot_aqi_distribution(df)
    plot_aqi_over_time(df)
    plot_hourly_patterns(df)
    plot_weekly_patterns(df)
    plot_correlation_heatmap(df)
    plot_seasonal_patterns(df)
    plot_weather_vs_aqi(df)
    print(f"\n✅ All plots saved to: {PLOT_DIR}")
    print("🎉 EDA complete")