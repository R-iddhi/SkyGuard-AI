from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

event_ids = [
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

print("=" * 70)
print("FINAL BENCHMARK ANOMALY INSPECTION")
print("=" * 70)

for event_id in event_ids:
    event = df[df["event_id"] == event_id]

    print()
    print("-" * 70)
    print(event_id)
    print("-" * 70)

    print(
        event[
            [
                "timestamp",
                "temperature",
                "pressure",
                "humidity",
                "wind_speed",
                "wind_direction",
                "is_anomaly",
                "anomaly_type",
                "event_id"
            ]
        ].to_string(index=False)
    )

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    df[df["is_anomaly"] == 1][
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_direction",
            "anomaly_type",
            "event_id"
        ]
    ].to_string(index=False)
)