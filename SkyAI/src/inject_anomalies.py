import xarray as xr
import pandas as pd
import numpy as np

# -----------------------------------
# LOAD ORIGINAL DATASET
# -----------------------------------

file_path = "data/raw/aws_data.nc"

data = xr.open_dataset(file_path)

# Convert to DataFrame
df = data.to_dataframe().reset_index()

# Rename variables
df = df.rename(columns={
    "obstime": "timestamp",
    "tempr": "temperature",
    "rh": "pressure",
    "ws": "wind_speed",
    "wd": "wind_direction",
    "ap": "humidity"
})

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

# Create anomaly label
df["is_anomaly"] = 0

# Make a copy
anomaly_df = df.copy()


# -----------------------------------
# SPIKE ANOMALY
# -----------------------------------

spike_index = 5000

original_temperature = anomaly_df.loc[spike_index, "temperature"]

anomaly_df.loc[spike_index, "temperature"] = 60.0
anomaly_df.loc[spike_index, "is_anomaly"] = 1

print("SPIKE ANOMALY")

print("\nTimestamp:")
print(anomaly_df.loc[spike_index, "timestamp"])

print("\nOriginal temperature:")
print(original_temperature)

print("\nNew temperature:")
print(anomaly_df.loc[spike_index, "temperature"])

print("\nAnomaly label:")
print(anomaly_df.loc[spike_index, "is_anomaly"])

print("\nNearby observations:")
print(
    anomaly_df.loc[
        spike_index - 2:spike_index + 2,
        ["timestamp", "temperature", "is_anomaly"]
    ].to_string(index=False)
)


# -----------------------------------
# DROP ANOMALY
# -----------------------------------

drop_index = 7000

original_drop_temperature = anomaly_df.loc[drop_index, "temperature"]

anomaly_df.loc[drop_index, "temperature"] = -60.0
anomaly_df.loc[drop_index, "is_anomaly"] = 1

print("\n\nDROP ANOMALY")

print("\nTimestamp:")
print(anomaly_df.loc[drop_index, "timestamp"])

print("\nOriginal temperature:")
print(original_drop_temperature)

print("\nNew temperature:")
print(anomaly_df.loc[drop_index, "temperature"])

print("\nAnomaly label:")
print(anomaly_df.loc[drop_index, "is_anomaly"])

print("\nNearby observations:")
print(
    anomaly_df.loc[
        drop_index - 2:drop_index + 2,
        ["timestamp", "temperature", "is_anomaly"]
    ].to_string(index=False)
)


# -----------------------------------
# FROZEN SENSOR ANOMALY
# -----------------------------------

frozen_start = 9000
frozen_length = 10

frozen_value = anomaly_df.loc[frozen_start, "temperature"]

for i in range(frozen_length):
    index = frozen_start + i

    anomaly_df.loc[index, "temperature"] = frozen_value
    anomaly_df.loc[index, "is_anomaly"] = 1

print("\n\nFROZEN SENSOR ANOMALY")

print("\nFrozen temperature value:")
print(frozen_value)

print("\nNumber of frozen readings:")
print(frozen_length)

print("\nFrozen observations:")
print(
    anomaly_df.loc[
        frozen_start:frozen_start + frozen_length - 1,
        ["timestamp", "temperature", "is_anomaly"]
    ].to_string(index=False)
)


# -----------------------------------
# RANGE ANOMALY
# -----------------------------------

# IMPORTANT:
# This is now in the TESTING period.

range_index = 14000

original_humidity = anomaly_df.loc[range_index, "humidity"]

anomaly_df.loc[range_index, "humidity"] = 135.0
anomaly_df.loc[range_index, "is_anomaly"] = 1

print("\n\nRANGE ANOMALY")

print("\nTimestamp:")
print(anomaly_df.loc[range_index, "timestamp"])

print("\nOriginal humidity:")
print(original_humidity)

print("\nNew humidity:")
print(anomaly_df.loc[range_index, "humidity"])

print("\nAnomaly label:")
print(anomaly_df.loc[range_index, "is_anomaly"])

print("\nNearby observations:")
print(
    anomaly_df.loc[
        range_index - 2:range_index + 2,
        ["timestamp", "humidity", "is_anomaly"]
    ].to_string(index=False)
)


# -----------------------------------
# MULTIVARIATE INCONSISTENCY
# -----------------------------------

# IMPORTANT:
# This is also now in the TESTING period.

multi_index = 15000

original_multi_temperature = anomaly_df.loc[multi_index, "temperature"]

anomaly_df.loc[multi_index, "temperature"] = 50.0
anomaly_df.loc[multi_index, "is_anomaly"] = 1

print("\n\nMULTIVARIATE INCONSISTENCY ANOMALY")

print("\nTimestamp:")
print(anomaly_df.loc[multi_index, "timestamp"])

print("\nOriginal temperature:")
print(original_multi_temperature)

print("\nNew temperature:")
print(anomaly_df.loc[multi_index, "temperature"])

print("\nPressure:")
print(anomaly_df.loc[multi_index, "pressure"])

print("\nHumidity:")
print(anomaly_df.loc[multi_index, "humidity"])

print("\nWind speed:")
print(anomaly_df.loc[multi_index, "wind_speed"])

print("\nAnomaly label:")
print(anomaly_df.loc[multi_index, "is_anomaly"])

print("\nNearby observations:")
print(
    anomaly_df.loc[
        multi_index - 2:multi_index + 2,
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "is_anomaly"
        ]
    ].to_string(index=False)
)


# -----------------------------------
# ANOMALY SUMMARY
# -----------------------------------

print("\n\nANOMALY SUMMARY")

print("\nTotal observations:")
print(len(anomaly_df))

print("\nNormal observations:")
print((anomaly_df["is_anomaly"] == 0).sum())

print("\nAnomaly observations:")
print((anomaly_df["is_anomaly"] == 1).sum())

print("\nAnomaly percentage:")
print(
    (anomaly_df["is_anomaly"].sum() / len(anomaly_df)) * 100
)


# -----------------------------------
# SAVE ANOMALY DATASET
# -----------------------------------

output_file = "data/aws_anomaly_dataset.csv"

anomaly_df.to_csv(output_file, index=False)

print("\n\nDATASET SAVED")
print("File:", output_file)
print("Rows:", len(anomaly_df))
print("Columns:", len(anomaly_df.columns))