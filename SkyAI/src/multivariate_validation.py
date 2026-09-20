import pandas as pd

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. CALCULATE CHANGES
# ==========================================

df["temperature_change"] = df["temperature"].diff()

df["pressure_change"] = df["pressure"].diff()

df["humidity_change"] = df["humidity"].diff()


# ==========================================
# 3. DEFINE MULTIVARIATE CONDITIONS
# ==========================================

# Large temperature change
temperature_change_flag = (
    df["temperature_change"].abs() > 10
)

# Large pressure change
pressure_change_flag = (
    df["pressure_change"].abs() > 10
)

# Large humidity change
humidity_change_flag = (
    df["humidity_change"].abs() > 20
)

# Physically impossible humidity
humidity_range_flag = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100)
)


# ==========================================
# 4. COMBINE CONDITIONS
# ==========================================

df["multivariate_anomaly"] = (
    (
        temperature_change_flag &
        humidity_change_flag
    )
    |
    (
        temperature_change_flag &
        pressure_change_flag
    )
    |
    humidity_range_flag
)


# ==========================================
# 5. DISPLAY SUMMARY
# ==========================================

print("MULTIVARIATE ANOMALY DETECTION")

print("\nTotal observations:")
print(len(df))

print("\nMultivariate anomalies:")
print(df["multivariate_anomaly"].sum())


# ==========================================
# 6. DISPLAY INJECTED ANOMALIES
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
            "temperature_change",
            "pressure_change",
            "humidity_change",
            "is_anomaly",
            "multivariate_anomaly"
        ]
    ].to_string(index=False)
)