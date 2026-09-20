import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"
df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. PHYSICAL VALIDATION
# ==========================================

humidity_invalid = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100)
)

wind_speed_invalid = (
    df["wind_speed"] < 0
)

wind_direction_invalid = (
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)

df["physical_anomaly"] = (
    humidity_invalid |
    wind_speed_invalid |
    wind_direction_invalid
)


# ==========================================
# 3. TEMPORAL VALIDATION
# ==========================================

df["temperature_change"] = df["temperature"].diff()
df["pressure_change"] = df["pressure"].diff()
df["humidity_change"] = df["humidity"].diff()
df["wind_speed_change"] = df["wind_speed"].diff()


temperature_anomaly = (
    df["temperature_change"].abs() > 10
)

pressure_anomaly = (
    df["pressure_change"].abs() > 10
)

humidity_anomaly = (
    df["humidity_change"].abs() > 20
)

wind_speed_anomaly = (
    df["wind_speed_change"].abs() > 30
)


df["temporal_anomaly"] = (
    temperature_anomaly |
    pressure_anomaly |
    humidity_anomaly |
    wind_speed_anomaly
)


# ==========================================
# 4. COMBINE BOTH DETECTORS
# ==========================================

df["combined_anomaly"] = (
    df["physical_anomaly"] |
    df["temporal_anomaly"]
)


# ==========================================
# 5. DISPLAY RESULTS
# ==========================================

print("COMBINED ANOMALY DETECTION")

print("\nTotal observations:")
print(len(df))

print("\nPhysical anomalies:")
print(df["physical_anomaly"].sum())

print("\nTemporal anomalies:")
print(df["temporal_anomaly"].sum())

print("\nCombined anomalies:")
print(df["combined_anomaly"].sum())


# ==========================================
# 6. CHECK INJECTED ANOMALIES
# ==========================================

print("\n\nINJECTED ANOMALIES")

injected = df[
    df["is_anomaly"] == 1
]

print(
    injected[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "is_anomaly",
            "physical_anomaly",
            "temporal_anomaly",
            "combined_anomaly"
        ]
    ].to_string(index=False)
)