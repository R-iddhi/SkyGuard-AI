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

# Calculate changes
df["temperature_change"] = df["temperature"].diff()
df["pressure_change"] = df["pressure"].diff()
df["humidity_change"] = df["humidity"].diff()
df["wind_speed_change"] = df["wind_speed"].diff()

# Find the 10 largest absolute changes
print("10 LARGEST TEMPERATURE CHANGES")
print(
    df.loc[
        df["temperature_change"].abs().nlargest(10).index,
        ["timestamp", "temperature", "temperature_change"]
    ].sort_values("timestamp")
)

print("\n10 LARGEST PRESSURE CHANGES")
print(
    df.loc[
        df["pressure_change"].abs().nlargest(10).index,
        ["timestamp", "pressure", "pressure_change"]
    ].sort_values("timestamp")
)

print("\n10 LARGEST HUMIDITY CHANGES")
print(
    df.loc[
        df["humidity_change"].abs().nlargest(10).index,
        ["timestamp", "humidity", "humidity_change"]
    ].sort_values("timestamp")
)

print("\n10 LARGEST WIND SPEED CHANGES")
print(
    df.loc[
        df["wind_speed_change"].abs().nlargest(10).index,
        ["timestamp", "wind_speed", "wind_speed_change"]
    ].sort_values("timestamp")
)