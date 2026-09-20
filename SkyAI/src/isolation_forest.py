import pandas as pd
from sklearn.ensemble import IsolationForest

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
# 2. SELECT FEATURES
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
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

# Keep complete test information
test_df = df.iloc[split_index:].copy()


# ==========================================
# 4. CREATE ISOLATION FOREST MODEL
# ==========================================

model = IsolationForest(
    n_estimators=100,
    contamination=0.001,
    random_state=42
)


# ==========================================
# 5. TRAIN MODEL
# ==========================================

model.fit(X_train)


# ==========================================
# 6. PREDICT TEST DATA
# ==========================================

predictions = model.predict(X_test)


# ==========================================
# 7. CALCULATE ANOMALY SCORES
# ==========================================

scores = model.decision_function(X_test)

test_df["prediction"] = predictions
test_df["anomaly_score"] = scores


# ==========================================
# 8. DISPLAY BASIC RESULTS
# ==========================================

anomalies = (predictions == -1).sum()
normal = (predictions == 1).sum()

print("ISOLATION FOREST RESULTS")

print("\nTraining observations:")
print(len(X_train))

print("\nTesting observations:")
print(len(X_test))

print("\nPredicted normal observations:")
print(normal)

print("\nPredicted anomalies:")
print(anomalies)


# ==========================================
# 9. SHOW KNOWN TEST ANOMALIES
# ==========================================

print("\n\nKNOWN TEST ANOMALIES")

known_test_anomalies = test_df[
    test_df["is_anomaly"] == 1
]

print(
    known_test_anomalies[
        [
            "timestamp",
            "temperature",
            "pressure",
            "wind_speed",
            "wind_direction",
            "humidity",
            "is_anomaly",
            "prediction",
            "anomaly_score"
        ]
    ].to_string(index=False)
)


# ==========================================
# 10. SHOW MOST ANOMALOUS OBSERVATIONS
# ==========================================

print("\n\nMOST ANOMALOUS OBSERVATIONS")

most_anomalous = test_df.sort_values(
    "anomaly_score"
).head(15)

print(
    most_anomalous[
        [
            "timestamp",
            "temperature",
            "pressure",
            "wind_speed",
            "wind_direction",
            "humidity",
            "is_anomaly",
            "prediction",
            "anomaly_score"
        ]
    ].to_string(index=False)
)