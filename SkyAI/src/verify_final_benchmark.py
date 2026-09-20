from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

events = [
    "TEMP_SPIKE_01",
    "TEMP_DROP_01",
    "HUM_HIGH_01",
    "HUM_LOW_01",
    "PRESS_SPIKE_01",
    "PRESS_DROP_01",
    "FROZEN_01",
    "MULTI_01",
    "TEMP_HUM_01"
]

print("=" * 80)
print("SKYGUARD AI FINAL BENCHMARK VERIFICATION")
print("=" * 80)

print()
print(f"Total rows: {len(df)}")
print(f"Rows marked is_anomaly = 1: {(df['is_anomaly'] == 1).sum()}")
print(f"Rows marked is_anomaly = 0: {(df['is_anomaly'] == 0).sum()}")

print()
print("-" * 80)
print("EVENT LABEL COUNTS")
print("-" * 80)

for event in events:
    event_rows = df[df["event_id"] == event]

    print()
    print(event)
    print(f"Rows: {len(event_rows)}")
    print(
        f"is_anomaly=1: "
        f"{(event_rows['is_anomaly'] == 1).sum()}"
    )
    print(
        f"is_anomaly=0: "
        f"{(event_rows['is_anomaly'] == 0).sum()}"
    )

print()
print("-" * 80)
print("INJECTED EVENT VALUES")
print("-" * 80)

for event in events:
    event_rows = df[df["event_id"] == event]

    print()
    print(event)

    columns = [
        "timestamp",
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_direction",
        "is_anomaly",
        "event_id"
    ]

    print(
        event_rows[columns]
        .to_string(index=False)
    )

print()
print("-" * 80)
print("EVENT ID VALUES")
print("-" * 80)

print(
    df["event_id"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)