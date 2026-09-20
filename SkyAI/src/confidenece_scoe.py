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
        (df["temperature_change"].abs() > 10) &
        (df["humidity_change"].abs() > 20)
    )
    |
    (
        (df["temperature_change"].abs() > 10) &
        (df["pressure_change"].abs() > 10)
    )
    |
    humidity_invalid
)


# ==========================================
# 7. CONFIDENCE SCORE
# ==========================================

df["confidence_score"] = 0.0


# Physical violation = strong evidence
df.loc[
    df["physical_anomaly"],
    "confidence_score"
] += 40


# Sudden change = evidence
df.loc[
    df["temporal_anomaly"],
    "confidence_score"
] += 20


# Frozen sensor = strong temporal evidence
df.loc[
    df["frozen_temperature"],
    "confidence_score"
] += 30


# Multivariate inconsistency = strong evidence
df.loc[
    df["multivariate_anomaly"],
    "confidence_score"
] += 30


# ==========================================
# 8. LIMIT SCORE TO 100
# ==========================================

df["confidence_score"] = (
    df["confidence_score"].clip(upper=100)
)


# ==========================================
# 9. UNIFIED ANOMALY
# ==========================================

df["unified_anomaly"] = (
    df["confidence_score"] > 0
)


# ==========================================
# 10. DISPLAY RESULTS
# ==========================================

print("CONFIDENCE SCORE ANALYSIS")

print("\nTotal observations:")
print(len(df))

print("\nObservations with anomaly evidence:")
print(df["unified_anomaly"].sum())


# ==========================================
# 11. INJECTED ANOMALIES
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
            "confidence_score"
        ]
    ].to_string(index=False)
)


# ==========================================
# 12. MOST CONFIDENT ANOMALIES
# ==========================================

print("\n\nHIGHEST CONFIDENCE ANOMALIES")

high_confidence = (
    df[df["unified_anomaly"]]
    .sort_values(
        "confidence_score",
        ascending=False
    )
    .head(15)
)

print(
    high_confidence[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "confidence_score"
        ]
    ].to_string(index=False)
)