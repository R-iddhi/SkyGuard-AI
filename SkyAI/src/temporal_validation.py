import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"
df = pd.read_csv(file_path)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CALCULATE CHANGES
# ==========================================

df["temperature_change"] = df["temperature"].diff()

df["pressure_change"] = df["pressure"].diff()

df["humidity_change"] = df["humidity"].diff()

df["wind_speed_change"] = df["wind_speed"].diff()


# ==========================================
# 3. DEFINE TEMPORAL THRESHOLDS
# ==========================================

temperature_threshold = 10
pressure_threshold = 10
humidity_threshold = 20
wind_speed_threshold = 30


# ==========================================
# 4. CHECK FOR SUDDEN CHANGES
# ==========================================

temperature_anomaly = (
    df["temperature_change"].abs() > temperature_threshold
)

pressure_anomaly = (
    df["pressure_change"].abs() > pressure_threshold
)

humidity_anomaly = (
    df["humidity_change"].abs() > humidity_threshold
)

wind_speed_anomaly = (
    df["wind_speed_change"].abs() > wind_speed_threshold
)


# ==========================================
# 5. COMBINE TEMPORAL ANOMALIES
# ==========================================

df["temporal_anomaly"] = (
    temperature_anomaly |
    pressure_anomaly |
    humidity_anomaly |
    wind_speed_anomaly
)


# ==========================================
# 6. DISPLAY SUMMARY
# ==========================================

print("TEMPORAL ANOMALY DETECTION")

print("\nTotal observations:")
print(len(df))

print("\nTemperature sudden changes:")
print(temperature_anomaly.sum())

print("\nPressure sudden changes:")
print(pressure_anomaly.sum())

print("\nHumidity sudden changes:")
print(humidity_anomaly.sum())

print("\nWind speed sudden changes:")
print(wind_speed_anomaly.sum())

print("\nTotal temporal anomalies:")
print(df["temporal_anomaly"].sum())


# ==========================================
# 7. DISPLAY TEMPORAL ANOMALIES
# ==========================================

print("\n\nTEMPORAL ANOMALIES")

temporal_anomalies = df[
    df["temporal_anomaly"] == True
]

print(
    temporal_anomalies[
        [
            "timestamp",
            "temperature",
            "temperature_change",
            "pressure",
            "pressure_change",
            "humidity",
            "humidity_change",
            "wind_speed",
            "wind_speed_change",
            "is_anomaly",
            "temporal_anomaly"
        ]
    ].to_string(index=False)
)