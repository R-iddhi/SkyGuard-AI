from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_controlled_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["frozen_detected"] = (
    df["temperature"]
    .rolling(8)
    .std()
    .fillna(999)
    .eq(0)
)

frozen_rows = df[df["anomaly_type"] == "Frozen Sensor"]

print("FROZEN SENSOR DETECTION ANALYSIS")
print()

print("=" * 70)
print("INJECTED FROZEN SENSOR OBSERVATIONS")
print("=" * 70)

print(
    frozen_rows[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "frozen_detected"
        ]
    ].to_string(index=False)
)

print()

print("=" * 70)
print("DETECTION SUMMARY")
print("=" * 70)

total = len(frozen_rows)
detected = int(frozen_rows["frozen_detected"].sum())

print(f"Total injected frozen observations: {total}")
print(f"Detected as frozen: {detected}")
print(f"Not yet detectable: {total - detected}")

if detected > 0:
    first_detection = frozen_rows[
        frozen_rows["frozen_detected"]
    ].iloc[0]

    print()
    print("First frozen detection:")
    print(f"Timestamp: {first_detection['timestamp']}")
    print(f"Temperature: {first_detection['temperature']}")

print()

print("=" * 70)
print("WHY EARLY OBSERVATIONS ARE NOT DETECTED")
print("=" * 70)

print(
    "The detector requires 8 consecutive observations "
    "with zero temperature variation."
)

print(
    "Therefore, the beginning of a frozen event is treated "
    "as insufficient evidence until enough history is available."
)