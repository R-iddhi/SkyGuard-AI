import pandas as pd
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ==========================================
# 2. FEATURES
# ==========================================

features = [
    "temperature",
    "pressure",
    "wind_speed",
    "wind_direction",
    "humidity"
]

X = df[features]


# ==========================================
# 3. TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

test_df = df.iloc[split_index:].copy()


# ==========================================
# 4. ISOLATION FOREST
# ==========================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.001,
    random_state=42
)

model.fit(X_train)


# ==========================================
# 5. ML PREDICTION
# ==========================================

ml_predictions = model.predict(X_test)

test_df["ml_anomaly"] = (
    ml_predictions == -1
)


# ==========================================
# 6. RULE-BASED PHYSICAL VALIDATION
# ==========================================

humidity_invalid = (
    (test_df["humidity"] < 0) |
    (test_df["humidity"] > 100)
)

wind_speed_invalid = (
    test_df["wind_speed"] < 0
)

wind_direction_invalid = (
    (test_df["wind_direction"] < 0) |
    (test_df["wind_direction"] > 360)
)

test_df["physical_anomaly"] = (
    humidity_invalid |
    wind_speed_invalid |
    wind_direction_invalid
)


# ==========================================
# 7. TEMPORAL VALIDATION
# ==========================================

test_df["temperature_change"] = (
    test_df["temperature"].diff()
)

test_df["pressure_change"] = (
    test_df["pressure"].diff()
)

test_df["humidity_change"] = (
    test_df["humidity"].diff()
)

test_df["wind_speed_change"] = (
    test_df["wind_speed"].diff()
)

test_df["temporal_anomaly"] = (
    (test_df["temperature_change"].abs() > 10) |
    (test_df["pressure_change"].abs() > 10) |
    (test_df["humidity_change"].abs() > 20)
)


# ==========================================
# 8. FROZEN SENSOR
# ==========================================

test_df["temperature_same"] = (
    test_df["temperature"] ==
    test_df["temperature"].shift(1)
)

temperature_group = (
    (~test_df["temperature_same"]).cumsum()
)

group_sizes = (
    test_df.groupby(temperature_group)["temperature"]
    .transform("size")
)

test_df["frozen_sensor"] = (
    group_sizes >= 8
)


# ==========================================
# 9. MULTIVARIATE VALIDATION
# ==========================================

test_df["multivariate_anomaly"] = (
    (
        (test_df["temperature_change"].abs() > 10) &
        (test_df["humidity_change"].abs() > 20)
    )
    |
    (
        (test_df["temperature_change"].abs() > 10) &
        (test_df["pressure_change"].abs() > 10)
    )
    |
    humidity_invalid
)


# ==========================================
# 10. COMBINE RULES
# ==========================================

test_df["rule_anomaly"] = (
    test_df["physical_anomaly"] |
    test_df["temporal_anomaly"] |
    test_df["frozen_sensor"] |
    test_df["multivariate_anomaly"]
)


# ==========================================
# 11. FINAL COMBINED DETECTION
# ==========================================

test_df["combined_anomaly"] = (
    test_df["ml_anomaly"] |
    test_df["rule_anomaly"]
)


# ==========================================
# 12. RESULTS
# ==========================================

print("COMBINED ML + RULE-BASED DETECTION")

print("\nTraining observations:")
print(len(X_train))

print("\nTesting observations:")
print(len(X_test))

print("\nML anomalies:")
print(test_df["ml_anomaly"].sum())

print("\nRule-based anomalies:")
print(test_df["rule_anomaly"].sum())

print("\nCombined anomalies:")
print(test_df["combined_anomaly"].sum())


# ==========================================
# 13. KNOWN TEST ANOMALIES
# ==========================================

print("\n\nKNOWN TEST ANOMALIES")

known = test_df[
    test_df["is_anomaly"] == 1
]

print(
    known[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "is_anomaly",
            "ml_anomaly",
            "rule_anomaly",
            "combined_anomaly"
        ]
    ].to_string(index=False)
)