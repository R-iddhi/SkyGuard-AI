import xarray as xr
import pandas as pd

# Load dataset
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

# Select the suspicious period
start_time = "2016-09-09 12:00:00"
end_time = "2016-09-10 00:00:00"

event = df[
    (df["timestamp"] >= start_time) &
    (df["timestamp"] <= end_time)
]

print("SUSPICIOUS EVENT INSPECTION")
print("\nPeriod:")
print(start_time, "to", end_time)

print("\nSensor readings:")
print(
    event[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity",
            "wind_speed",
            "wind_direction"
        ]
    ].to_string(index=False)
)