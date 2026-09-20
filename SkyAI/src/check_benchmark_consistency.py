from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

print("=" * 80)
print("SKYGUARD AI BENCHMARK CONSISTENCY CHECK")
print("=" * 80)

print()
print(f"Total rows: {len(df)}")
print(f"Anomaly rows: {(df['is_anomaly'] == 1).sum()}")
print(f"Normal rows: {(df['is_anomaly'] == 0).sum()}")

print()
print("-" * 80)
print("ALL LABELED ANOMALY ROWS")
print("-" * 80)

anomaly_rows = df[df["is_anomaly"] == 1]

print(
    anomaly_rows[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_direction",
            "event_id"
        ]
    ].to_string(index=True)
)

print()
print("-" * 80)
print("SEARCHING FOR EXPECTED INJECTED VALUES")
print("-" * 80)

tests = {
    "Temperature 60": df["temperature"] == 60,
    "Temperature -60": df["temperature"] == -60,
    "Humidity 135": df["humidity"] == 135,
    "Humidity -10": df["humidity"] == -10,
    "Pressure 1010": df["pressure"] == 1010,
    "Pressure 920": df["pressure"] == 920,
    "Temperature 50": df["temperature"] == 50,
    "Temperature 35 + Humidity 20": (
        (df["temperature"] == 35)
        & (df["humidity"] == 20)
    )
}

for name, condition in tests.items():
    matches = df[condition]

    print()
    print(name)
    print(f"Matches: {len(matches)}")

    if len(matches) > 0:
        print(
            matches[
                [
                    "timestamp",
                    "temperature",
                    "pressure",
                    "humidity",
                    "wind_speed",
                    "wind_direction",
                    "is_anomaly",
                    "event_id"
                ]
            ].to_string(index=True)
        )

print()
print("-" * 80)
print("FROZEN EVENT CHECK")
print("-" * 80)

frozen_value = -16.8

frozen_matches = df[
    df["temperature"] == frozen_value
]

print(
    f"Rows with temperature = {frozen_value}: "
    f"{len(frozen_matches)}"
)

if len(frozen_matches) > 0:
    print(
        frozen_matches[
            [
                "timestamp",
                "temperature",
                "pressure",
                "humidity",
                "is_anomaly",
                "event_id"
            ]
        ].to_string(index=True)
    )

print()
print("-" * 80)
print("UNLABELED EXTREME READINGS")
print("-" * 80)

unlabeled_extremes = df[
    (df["is_anomaly"] == 0)
    & (
        (df["temperature"].abs() >= 40)
        | (df["humidity"] < 0)
        | (df["humidity"] > 100)
        | (df["pressure"] < 925)
        | (df["pressure"] > 1005)
    )
]

print(
    f"Unlabeled extreme rows: "
    f"{len(unlabeled_extremes)}"
)

if len(unlabeled_extremes) > 0:
    print(
        unlabeled_extremes[
            [
                "timestamp",
                "temperature",
                "pressure",
                "humidity",
                "wind_speed",
                "wind_direction",
                "is_anomaly",
                "event_id"
            ]
        ].to_string(index=True)
    )

print()
print("=" * 80)
print("CONSISTENCY CHECK COMPLETE")
print("=" * 80)