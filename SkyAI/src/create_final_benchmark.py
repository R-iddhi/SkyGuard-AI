from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "aws_anomaly_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(INPUT_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["is_anomaly"] = 0
df["anomaly_type"] = "Normal"
df["event_id"] = ""

events = []

def add_event(event_id, start, length, anomaly_type):
    end = start + length

    df.loc[start:end - 1, "is_anomaly"] = 1
    df.loc[start:end - 1, "anomaly_type"] = anomaly_type
    df.loc[start:end - 1, "event_id"] = event_id

    events.append(
        (event_id, start, length, anomaly_type)
    )


start = 1000
df.loc[start, "temperature"] = 60.0
add_event(
    "TEMP_SPIKE_01",
    start,
    1,
    "Temperature Spike"
)

start = 1250
df.loc[start, "temperature"] = -60.0
add_event(
    "TEMP_DROP_01",
    start,
    1,
    "Temperature Drop"
)

start = 1500
df.loc[start, "humidity"] = 135.0
add_event(
    "HUM_HIGH_01",
    start,
    1,
    "Humidity High"
)

start = 1750
df.loc[start, "humidity"] = -10.0
add_event(
    "HUM_LOW_01",
    start,
    1,
    "Humidity Low"
)

start = 2000
df.loc[start, "pressure"] = 1010.0
add_event(
    "PRESS_SPIKE_01",
    start,
    1,
    "Pressure Spike"
)

start = 2250
df.loc[start, "pressure"] = 920.0
add_event(
    "PRESS_DROP_01",
    start,
    1,
    "Pressure Drop"
)

start = 2500
end = start + 10

frozen_temperature = df.loc[start - 1, "temperature"]

df.loc[start:end - 1, "temperature"] = frozen_temperature

add_event(
    "FROZEN_01",
    start,
    10,
    "Frozen Sensor"
)

start = 2750

df.loc[start, "temperature"] = 50.0
df.loc[start, "pressure"] = 951.5
df.loc[start, "humidity"] = 34.0
df.loc[start, "wind_speed"] = 52.5
df.loc[start, "wind_direction"] = 312.0

add_event(
    "MULTI_01",
    start,
    1,
    "Multivariate Inconsistency"
)

start = 3000

df.loc[start, "temperature"] = 35.0
df.loc[start, "humidity"] = 20.0

add_event(
    "TEMP_HUM_01",
    start,
    1,
    "Temperature-Humidity Inconsistency"
)

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 70)
print("FINAL BENCHMARK CREATED")
print("=" * 70)

print(f"Dataset: {OUTPUT_PATH}")
print(f"Total observations: {len(df)}")
print(f"Normal observations: {(df['is_anomaly'] == 0).sum()}")
print(f"Anomaly observations: {(df['is_anomaly'] == 1).sum()}")
print(f"Anomaly events: {len(events)}")

print()
print("ANOMALY EVENTS")
print("-" * 70)

for event_id, start, length, anomaly_type in events:
    timestamp_start = df.loc[start, "timestamp"]
    timestamp_end = df.loc[start + length - 1, "timestamp"]

    print(
        f"{event_id}: "
        f"{anomaly_type} | "
        f"rows {start}-{start + length - 1} | "
        f"{timestamp_start} -> {timestamp_end} | "
        f"{length} observations"
    )

print()
print("ANOMALY TYPE COUNTS")
print("-" * 70)

print(
    df[df["is_anomaly"] == 1]["anomaly_type"]
    .value_counts()
)

print()
print("BENCHMARK SAVED SUCCESSFULLY")