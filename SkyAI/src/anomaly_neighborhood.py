from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

events = df[df["is_anomaly"] == 1]["event_id"].dropna().unique()

print("=" * 70)
print("SKYGUARD AI ANOMALY NEIGHBORHOOD ANALYSIS")
print("=" * 70)

for event in events:
    event_rows = df[df["event_id"] == event]

    start_index = event_rows.index.min()
    end_index = event_rows.index.max()

    start = max(0, start_index - 2)
    end = min(len(df), end_index + 3)

    print()
    print(f"EVENT: {event}")
    print("-" * 70)

    columns = [
        "timestamp",
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "is_anomaly",
        "event_id"
    ]

    print(
        df.loc[start:end - 1, columns]
        .to_string(index=True)
    )

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)