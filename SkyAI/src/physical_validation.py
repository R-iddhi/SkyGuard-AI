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
# 2. PHYSICAL VALIDATION RULES
# ==========================================

# Humidity must be between 0 and 100
humidity_invalid = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100)
)

# Wind speed cannot be negative
wind_speed_invalid = (
    df["wind_speed"] < 0
)

# Wind direction must be between 0 and 360
wind_direction_invalid = (
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
)


# ==========================================
# 3. COMBINE PHYSICAL VALIDATION RESULTS
# ==========================================

df["physical_anomaly"] = (
    humidity_invalid |
    wind_speed_invalid |
    wind_direction_invalid
)


# ==========================================
# 4. DISPLAY SUMMARY
# ==========================================

print("PHYSICAL SENSOR VALIDATION")

print("\nTotal observations:")
print(len(df))

print("\nHumidity violations:")
print(humidity_invalid.sum())

print("\nWind speed violations:")
print(wind_speed_invalid.sum())

print("\nWind direction violations:")
print(wind_direction_invalid.sum())

print("\nTotal physical anomalies:")
print(df["physical_anomaly"].sum())


# ==========================================
# 5. DISPLAY PHYSICAL ANOMALIES
# ==========================================

print("\n\nPHYSICAL ANOMALIES")

physical_anomalies = df[
    df["physical_anomaly"] == True
]

print(
    physical_anomalies[
        [
            "timestamp",
            "temperature",
            "pressure",
            "wind_speed",
            "wind_direction",
            "humidity",
            "is_anomaly",
            "physical_anomaly"
        ]
    ].to_string(index=False)
)