from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aws_benchmark_dataset.csv"

FEATURES = [
    "temperature",
    "pressure",
    "humidity",
    "wind_speed",
    "wind_direction"
]

df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["is_anomaly"] = 0
df["anomaly_type"] = "Normal"

rng = np.random.default_rng(42)

normal_indices = np.arange(100, len(df) - 100)

selected = rng.choice(normal_indices, size=90, replace=False)

groups = np.array_split(selected, 9)

for index in groups[0]:
    df.loc[index, "temperature"] += rng.choice([25, 30, 35])

for index in groups[1]:
    df.loc[index, "temperature"] -= rng.choice([25, 30, 35])

for index in groups[2]:
    df.loc[index, "humidity"] = rng.choice([105, 110, 120, 135])

for index in groups[3]:
    df.loc[index, "humidity"] = rng.choice([-5, -10, -20])

for index in groups[4]:
    df.loc[index, "pressure"] += rng.choice([20, 25, 30])

for index in groups[5]:
    df.loc[index, "pressure"] -= rng.choice([20, 25, 30])

for index in groups[6]:
    base_value = df.loc[index, "temperature"]
    df.loc[index:index + 7, "temperature"] = base_value

for index in groups[7]:
    df.loc[index, "temperature"] = 50
    df.loc[index, "humidity"] = 25
    df.loc[index, "pressure"] = 950

for index in groups[8]:
    df.loc[index, "temperature"] = 45
    df.loc[index, "humidity"] = 98

anomaly_indices = np.concatenate(groups)

df.loc[anomaly_indices, "is_anomaly"] = 1

for index in groups[0]:
    df.loc[index, "anomaly_type"] = "Temperature Spike"

for index in groups[1]:
    df.loc[index, "anomaly_type"] = "Temperature Drop"

for index in groups[2]:
    df.loc[index, "anomaly_type"] = "Humidity High"

for index in groups[3]:
    df.loc[index, "anomaly_type"] = "Humidity Low"

for index in groups[4]:
    df.loc[index, "anomaly_type"] = "Pressure Spike"

for index in groups[5]:
    df.loc[index, "anomaly_type"] = "Pressure Drop"

for index in groups[6]:
    df.loc[index:index + 7, "is_anomaly"] = 1
    df.loc[index:index + 7, "anomaly_type"] = "Frozen Sensor"

for index in groups[7]:
    df.loc[index, "anomaly_type"] = "Multivariate Inconsistency"

for index in groups[8]:
    df.loc[index, "anomaly_type"] = "Temperature-Humidity Inconsistency"

df.to_csv(OUTPUT_PATH, index=False)

print("BENCHMARK DATASET CREATED")
print()
print(f"Total observations: {len(df)}")
print(f"Anomaly observations: {int(df['is_anomaly'].sum())}")
print(f"Normal observations: {int((df['is_anomaly'] == 0).sum())}")
print()
print("ANOMALY TYPES")
print(df[df["is_anomaly"] == 1]["anomaly_type"].value_counts())
print()
print(f"Saved to: {OUTPUT_PATH}")