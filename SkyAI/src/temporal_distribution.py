from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "aws_final_benchmark.csv"

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

features = [
    "temperature",
    "pressure",
    "humidity"
]

print("=" * 70)
print("SKYGUARD AI TEMPORAL CHANGE DISTRIBUTION")
print("=" * 70)

for feature in features:
    change = df[feature].diff().abs().dropna()

    normal_change = change[
        df.loc[change.index, "is_anomaly"] == 0
    ]

    print()
    print(f"{feature.upper()}")
    print("-" * 70)

    print(f"Mean change:        {normal_change.mean():.4f}")
    print(f"Median change:      {normal_change.median():.4f}")
    print(f"Std deviation:      {normal_change.std():.4f}")
    print(f"95th percentile:    {np.percentile(normal_change, 95):.4f}")
    print(f"99th percentile:    {np.percentile(normal_change, 99):.4f}")
    print(f"99.5th percentile:  {np.percentile(normal_change, 99.5):.4f}")
    print(f"99.9th percentile:  {np.percentile(normal_change, 99.9):.4f}")
    print(f"Maximum change:     {normal_change.max():.4f}")

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
