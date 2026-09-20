import xarray as xr
import matplotlib.pyplot as plt

# Load raw dataset
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

# Temperature
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["temperature"])
plt.title("Temperature Over Time")
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# Pressure
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["pressure"])
plt.title("Pressure Over Time")
plt.xlabel("Time")
plt.ylabel("Pressure (hPa)")
plt.tight_layout()
plt.show()

# Humidity
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["humidity"])
plt.title("Humidity Over Time")
plt.xlabel("Time")
plt.ylabel("Humidity (%)")
plt.tight_layout()
plt.show()

# Wind speed
plt.figure(figsize=(12, 5))
plt.plot(df["timestamp"], df["wind_speed"])
plt.title("Wind Speed Over Time")
plt.xlabel("Time")
plt.ylabel("Wind Speed")
plt.tight_layout()
plt.show()