from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "aws_data.nc"
OUTPUT_PATH = PROJECT_ROOT / "data" / "aws_clean_final_benchmark.csv"

ds = xr.open_dataset(RAW_PATH)

df = ds.to_dataframe().reset_index()

df = df.rename(columns={
    "obstime": "timestamp",
    "tempr": "temperature",
    "rh": "pressure",
    "ws": "wind_speed",
    "wd": "wind_direction",
    "ap": "humidity"
})

df = df[
    [
        "timestamp",
        "temperature",
        "pressure",
        "wind_speed",
        "wind_direction",
        "humidity"
    ]
].copy()

df = df.sort_values("timestamp").reset_index(drop=True)

df["is_anomaly"] = 0
df["event_id"] = pd.Series(index=df.index, dtype="object")

events = [
    {
        "index": 1000,
        "event_id": "TEMP_SPIKE_01",
        "temperature": 60.0
    },
    {
        "index": 1250,
        "event_id": "TEMP_DROP_01",
        "temperature": -60.0
    },
    {
        "index": 1500,
        "event_id": "HUM_HIGH_01",
        "humidity": 135.0
    },
    {
        "index": 1750,
        "event_id": "HUM_LOW_01",
        "humidity": -10.0
    },
    {
        "index": 2000,
        "event_id": "PRESS_SPIKE_01",
        "pressure": 1010.0
    },
    {
        "index": 2250,
        "event_id": "PRESS_DROP_01",
        "pressure": 920.0
    },
    {
        "start": 2500,
        "end": 2510,
        "event_id": "FROZEN_01"
    },
    {
        "index": 2750,
        "event_id": "MULTI_01",
        "temperature": 50.0,
        "pressure": 951.5,
        "humidity": 34.0,
        "wind_speed": 52.5,
        "wind_direction": 312.0
    },
    {
        "index": 3000,
        "event_id": "TEMP_HUM_01",
        "temperature": 35.0,
        "humidity": 20.0
    }
]

for event in events:
    if "start" in event:
        start = event["start"]
        end = event["end"]

        frozen_value = df.loc[start - 1, "temperature"]

        df.loc[start:end - 1, "temperature"] = frozen_value
        df.loc[start:end - 1, "is_anomaly"] = 1
        df.loc[start:end - 1, "event_id"] = event["event_id"]

    else:
        index = event["index"]

        if "temperature" in event:
            df.loc[index, "temperature"] = event["temperature"]

        if "pressure" in event:
            df.loc[index, "pressure"] = event["pressure"]

        if "humidity" in event:
            df.loc[index, "humidity"] = event["humidity"]

        if "wind_speed" in event:
            df.loc[index, "wind_speed"] = event["wind_speed"]

        if "wind_direction" in event:
            df.loc[index, "wind_direction"] = event["wind_direction"]

        df.loc[index, "is_anomaly"] = 1
        df.loc[index, "event_id"] = event["event_id"]

df.to_csv(OUTPUT_PATH, index=False)

print("=" * 80)
print("CLEAN FINAL BENCHMARK CREATED")
print("=" * 80)

print(f"Source: {RAW_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"Total rows: {len(df)}")
print(f"Anomaly rows: {int(df['is_anomaly'].sum())}")
print(f"Normal rows: {int((df['is_anomaly'] == 0).sum())}")
print(f"Anomaly events: {df['event_id'].nunique()}")

print("\nEVENT SUMMARY")
print("-" * 80)

for event_id, group in df[df["is_anomaly"] == 1].groupby("event_id"):
    print(
        f"{event_id}: "
        f"{len(group)} rows | "
        f"{group['timestamp'].min()} -> {group['timestamp'].max()}"
    )

print("\nCLEAN BENCHMARK CREATION COMPLETE")
print("=" * 80)