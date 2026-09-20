import pandas as pd

# -----------------------------------
# LOAD ANOMALY DATASET
# -----------------------------------

file_path = "data/aws_anomaly_dataset.csv"

df = pd.read_csv(file_path)


# -----------------------------------
# PREPARE TIMESTAMP
# -----------------------------------

df["timestamp"] = pd.to_datetime(df["timestamp"])

# Make sure observations are in chronological order
df = df.sort_values("timestamp").reset_index(drop=True)


# -----------------------------------
# SELECT FEATURES
# -----------------------------------

features = [
    "temperature",
    "pressure",
    "wind_speed",
    "wind_direction",
    "humidity"
]

X = df[features]

# Known ground-truth labels
y = df["is_anomaly"]


# -----------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------

# Use the first 80% for training
# and the last 20% for testing.

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]


# -----------------------------------
# DISPLAY RESULTS
# -----------------------------------

print("TRAIN / TEST SPLIT")

print("\nTotal observations:")
print(len(df))

print("\nTraining observations:")
print(len(X_train))

print("Training anomalies:")
print(y_train.sum())

print("\nTesting observations:")
print(len(X_test))

print("Testing anomalies:")
print(y_test.sum())


# -----------------------------------
# TIME RANGES
# -----------------------------------

print("\nTraining time range:")
print(df["timestamp"].iloc[0])
print("to")
print(df["timestamp"].iloc[split_index - 1])

print("\nTesting time range:")
print(df["timestamp"].iloc[split_index])
print("to")
print(df["timestamp"].iloc[-1])