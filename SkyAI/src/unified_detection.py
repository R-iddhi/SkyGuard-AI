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
# 4. FROZEN TEMPERATURE DETECTION
# ==========================================

df["temperature_same"] = (
    df["temperature"] == df["temperature"].shift(1)
)

temperature_group = (
    (~df["temperature_same"]).cumsum()
)

group_sizes = (
    df.groupby(temperature_group)["temperature"]
    .transform("size")
)

frozen_threshold = 8

df["frozen_temperature"] = (
    group_sizes >= frozen_threshold
)


# ==========================================
# 5. MULTIVARIATE VALIDATION
# ==========================================

multivariate_anomaly = (
    (
        temperature_anomaly &
        humidity_anomaly
    )
    |
    (
        temperature_anomaly &
        pressure_anomaly
    )
    |
    humidity_invalid
)

df["multivariate_anomaly"] = (
    multivariate_anomaly
)


# ==========================================
# 6. UNIFIED ANOMALY FLAG
# ==========================================

df["unified_anomaly"] = (
    df["physical_anomaly"] |
    df["temporal_anomaly"] |
    df["frozen_temperature"] |
    df["multivariate_anomaly"]
)


# ==========================================
# 7. DISPLAY SUMMARY
# ==========================================

print("UNIFIED ANOMALY DETECTION")

print("\nTotal observations:")
print(len(df))

print("\nPhysical anomalies:")
print(df["physical_anomaly"].sum())

print("\nTemporal anomalies:")
print(df["temporal_anomaly"].sum())

print("\nFrozen temperature observations:")
print(df["frozen_temperature"].sum())

print("\nMultivariate anomalies:")
print(df["multivariate_anomaly"].sum())

print("\nUnified anomalies:")
print(df["unified_anomaly"].sum())


# ==========================================
# 8. CHECK INJECTED ANOMALIES
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
            "frozen_temperature",
            "multivariate_anomaly",
            "unified_anomaly"
        ]
    ].to_string(index=False)
)