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

df["temporal_anomaly"] = (
    (df["temperature_change"].abs() > 10) |
    (df["pressure_change"].abs() > 10) |
    (df["humidity_change"].abs() > 20) |
    (df["wind_speed_change"].abs() > 30)
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
        df["temperature_change"].abs() > 10
    ) &
    (
        df["humidity_change"].abs() > 20
    )
) | (
    (
        df["temperature_change"].abs() > 10
    ) &
    (
        df["pressure_change"].abs() > 10
    )
) | humidity_invalid


# ==========================================
# 7. UNIFIED ANOMALY
# ==========================================

df["unified_anomaly"] = (
    df["physical_anomaly"] |
    df["temporal_anomaly"] |
    df["frozen_temperature"] |
    df["multivariate_anomaly"]
)


# ==========================================
# 8. CREATE SEVERITY
# ==========================================

df["severity"] = "Normal"


# ------------------------------------------
# LOW SEVERITY
# ------------------------------------------

df.loc[
    df["unified_anomaly"],
    "severity"
] = "Low"


# ------------------------------------------
# MEDIUM SEVERITY
# ------------------------------------------

medium_condition = (
    df["temporal_anomaly"] |
    df["frozen_temperature"] |
    df["multivariate_anomaly"]
)

df.loc[
    medium_condition,
    "severity"
] = "Medium"


# ------------------------------------------
# HIGH SEVERITY
# ------------------------------------------

high_condition = (
    df["physical_anomaly"] |
    (df["temperature"].abs() > 50)
)

df.loc[
    high_condition,
    "severity"
] = "High"


# ==========================================
# 9. DISPLAY SUMMARY
# ==========================================

print("ANOMALY SEVERITY CLASSIFICATION")

print("\nSeverity distribution:")

print(
    df["severity"].value_counts()
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
            "unified_anomaly",
            "severity"
        ]
    ].to_string(index=False)
)