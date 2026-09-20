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

df["temperature_change"] = (
    df["temperature"].diff()
)

df["pressure_change"] = (
    df["pressure"].diff()
)

df["humidity_change"] = (
    df["humidity"].diff()
)


# ==========================================
# 3. PHYSICAL VALIDATION
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
# 4. TEMPORAL VALIDATION
# ==========================================

temperature_anomaly = (
    df["temperature_change"].abs() > 10
)

pressure_anomaly = (
    df["pressure_change"].abs() > 10
)

humidity_anomaly = (
    df["humidity_change"].abs() > 20
)

df["temporal_anomaly"] = (
    temperature_anomaly |
    pressure_anomaly |
    humidity_anomaly
)


# ==========================================
# 5. FROZEN SENSOR DETECTION
# ==========================================

df["temperature_same"] = (
    df["temperature"] ==
    df["temperature"].shift(1)
)

temperature_group = (
    (~df["temperature_same"]).cumsum()
)

group_sizes = (
    df.groupby(temperature_group)["temperature"]
    .transform("size")
)

df["frozen_sensor"] = (
    group_sizes >= 8
)


# ==========================================
# 6. MULTIVARIATE VALIDATION
# ==========================================

df["multivariate_anomaly"] = (
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


# ==========================================
# 7. CALCULATE ANOMALY RATE
# ==========================================

total_observations = len(df)

physical_rate = (
    df["physical_anomaly"].sum()
    / total_observations
)

temporal_rate = (
    df["temporal_anomaly"].sum()
    / total_observations
)

frozen_rate = (
    df["frozen_sensor"].sum()
    / total_observations
)

multivariate_rate = (
    df["multivariate_anomaly"].sum()
    / total_observations
)


# ==========================================
# 8. SENSOR HEALTH SCORE
# ==========================================

health_score = 100

health_score -= physical_rate * 100
health_score -= temporal_rate * 30
health_score -= frozen_rate * 100
health_score -= multivariate_rate * 50

health_score = max(0, min(100, health_score))


# ==========================================
# 9. HEALTH STATUS
# ==========================================

if health_score >= 90:
    health_status = "Healthy"

elif health_score >= 70:
    health_status = "Good"

elif health_score >= 50:
    health_status = "Warning"

else:
    health_status = "Critical"


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("SENSOR HEALTH ANALYSIS")

print("\nTotal observations:")
print(total_observations)

print("\nPhysical anomalies:")
print(df["physical_anomaly"].sum())

print("\nTemporal anomalies:")
print(df["temporal_anomaly"].sum())

print("\nFrozen sensor observations:")
print(df["frozen_sensor"].sum())

print("\nMultivariate anomalies:")
print(df["multivariate_anomaly"].sum())

print("\n------------------------------")

print("\nSensor Health Score:")
print(round(health_score, 2))

print("\nSensor Health Status:")
print(health_status)