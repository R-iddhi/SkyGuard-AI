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
df["wind_speed_change"] = df["wind_speed"].diff()


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
# 5. FROZEN SENSOR DETECTION
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

df["frozen_temperature"] = (
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
# 7. ROOT-CAUSE CLASSIFICATION
# ==========================================

df["root_cause"] = "Normal / No Anomaly"


# Invalid physical value
df.loc[
    df["physical_anomaly"],
    "root_cause"
] = "Sensor Fault"


# Frozen sensor
df.loc[
    df["frozen_temperature"],
    "root_cause"
] = "Frozen Sensor"


# Multivariate inconsistency
df.loc[
    df["multivariate_anomaly"],
    "root_cause"
] = "Multivariate Inconsistency"


# Sudden changes
df.loc[
    df["temporal_anomaly"],
    "root_cause"
] = "Sudden Sensor/Data Change"


# ==========================================
# 8. PRIORITY RULES
# ==========================================

# The most specific causes should have priority.

df.loc[
    df["frozen_temperature"],
    "root_cause"
] = "Frozen Sensor"

df.loc[
    df["physical_anomaly"],
    "root_cause"
] = "Sensor Fault"

df.loc[
    df["multivariate_anomaly"],
    "root_cause"
] = "Multivariate Inconsistency"


# ==========================================
# 9. DISPLAY DISTRIBUTION
# ==========================================

print("ROOT-CAUSE CLASSIFICATION")

print("\nRoot-cause distribution:")

print(
    df["root_cause"].value_counts()
)


# ==========================================
# 10. DISPLAY INJECTED ANOMALIES
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
            "root_cause"
        ]
    ].to_string(index=False)
)