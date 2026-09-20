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

# Calculate change from previous observation
df["temperature_change"] = df["temperature"].diff()
df["pressure_change"] = df["pressure"].diff()
df["humidity_change"] = df["humidity"].diff()
df["wind_speed_change"] = df["wind_speed"].diff()

# Display results
print("HOURLY CHANGE ANALYSIS")

print("\nTemperature change:")
print(df["temperature_change"].describe())

print("\nPressure change:")
print(df["pressure_change"].describe())

print("\nHumidity change:")
print(df["humidity_change"].describe())

print("\nWind speed change:")
print(df["wind_speed_change"].describe())