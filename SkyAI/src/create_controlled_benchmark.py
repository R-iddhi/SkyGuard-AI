from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["is_anomaly"] = 0
df["anomaly_type"] = "Normal"
df["event_id"] = ""

rng = np.random.default_rng(42)

start = 1000
spacing = 250
events = []

def add_event(event_id, start_index, anomaly_type, size=1):
    end_index = start_index + size
    events.append((event_id, start_index, end_index, anomaly_type))

add_event("TEMP_SPIKE_01", start + spacing * 0, "Temperature Spike")
add_event("TEMP_DROP_01", start + spacing * 1, "Temperature Drop")
add_event("HUM_HIGH_01", start + spacing * 2, "Humidity High")
add_event("HUM_LOW_01", start + spacing * 3, "Humidity Low")
add_event("PRESS_SPIKE_01", start + spacing * 4, "Pressure Spike")
add_event("PRESS_DROP_01", start + spacing * 5, "Pressure Drop")
add_event("FROZEN_01", start + spacing * 6, "Frozen Sensor", 10)
add_event("MULTI_01", start + spacing * 7, "Multivariate Inconsistency")
add_event("TEMP_HUM_01", start + spacing * 8, "Temperature-Humidity Inconsistency")

for event_id, index, end_index, anomaly_type in events:
    if anomaly_type == "Temperature Spike":
        df.loc[index, "temperature"] = 60

    elif anomaly_type == "Temperature Drop":
        df.loc[index, "temperature"] = -60

    elif anomaly_type == "Humidity High":
        df.loc[index, "humidity"] = 135

    elif anomaly_type == "Humidity Low":
        df.loc[index, "humidity"] = -10

    elif anomaly_type == "Pressure Spike":
        df.loc[index, "pressure"] = df.loc[index, "pressure"] + 40

    elif anomaly_type == "Pressure Drop":
        df.loc[index, "pressure"] = df.loc[index, "pressure"] - 40

    elif anomaly_type == "Frozen Sensor":
        temperature = df.loc[index, "temperature"]
        df.loc[index:end_index - 1, "temperature"] = temperature

    elif anomaly_type == "Multivariate Inconsistency":
        df.loc[index, "temperature"] = 50
        df.loc[index, "pressure"] = 950
        df.loc[index, "humidity"] = 25

    elif anomaly_type == "Temperature-Humidity Inconsistency":
        df.loc[index, "temperature"] = 45
        df.loc[index, "humidity"] = 98

    df.loc[index:end_index - 1, "is_anomaly"] = 1
    df.loc[index:end_index - 1, "anomaly_type"] = anomaly_type
    df.loc[index:end_index - 1, "event_id"] = event_id

df.to_csv(OUTPUT_PATH, index=False)

print("CONTROLLED BENCHMARK CREATED")
print()
print(f"Total observations: {len(df)}")
print(f"Normal observations: {int((df['is_anomaly'] == 0).sum())}")
print(f"Anomaly observations: {int(df['is_anomaly'].sum())}")
print()
print("ANOMALY EVENTS")

for event_id, index, end_index, anomaly_type in events:
    print(
        f"{event_id}: "
        f"{anomaly_type} | "
        f"rows {index}-{end_index - 1} | "
        f"observations {end_index - index}"
    )

print()
print("ANOMALY TYPE COUNTS")
print(df[df["is_anomaly"] == 1]["anomaly_type"].value_counts())
print()
print(f"Saved to: {OUTPUT_PATH}")