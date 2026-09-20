import xarray as xr
import pandas as pd

# 1. Load the original raw dataset
file_path = "data/raw/aws_data.nc"
data = xr.open_dataset(file_path)

# 2. Convert the NetCDF dataset into a DataFrame
df = data.to_dataframe().reset_index()

# 3. Rename columns to meaningful names
df = df.rename(columns={
    "obstime": "timestamp",
    "tempr": "temperature",
    "rh": "pressure",
    "ws": "wind_speed",
    "wd": "wind_direction",
    "ap": "humidity"
})

# 4. Sort data by time
df = df.sort_values("timestamp").reset_index(drop=True)

# 5. Display the first 5 rows
print("CLEAN DATASET")
print(df.head())

# 6. Display column names
print("\nCOLUMNS:")
print(df.columns.tolist())

# 7. Display dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# 8. Check physically invalid values
print("\nPHYSICAL VALIDITY CHECK")

print("\nInvalid humidity values:")
print(((df["humidity"] < 0) | (df["humidity"] > 100)).sum())

print("\nInvalid wind speed values:")
print((df["wind_speed"] < 0).sum())

print("\nInvalid wind direction values:")
print(((df["wind_direction"] < 0) | (df["wind_direction"] > 360)).sum())

print("\nTemperature range:")
print(df["temperature"].min(), "to", df["temperature"].max())

print("\nPressure range:")
print(df["pressure"].min(), "to", df["pressure"].max())

# 8. Physical validity checks

print("\nPHYSICAL VALIDITY CHECK")

# Humidity should be between 0% and 100%
invalid_humidity = (
    (df["humidity"] < 0) |
    (df["humidity"] > 100)
).sum()

print("\nInvalid humidity values:")
print(invalid_humidity)


# Wind speed cannot be negative
invalid_wind_speed = (
    df["wind_speed"] < 0
).sum()

print("\nInvalid wind speed values:")
print(invalid_wind_speed)


# Wind direction should be between 0° and 360°
invalid_wind_direction = (
    (df["wind_direction"] < 0) |
    (df["wind_direction"] > 360)
).sum()

print("\nInvalid wind direction values:")
print(invalid_wind_direction)


# Display temperature range
print("\nTemperature range:")
print(df["temperature"].min(), "to", df["temperature"].max())


# Display pressure range
print("\nPressure range:")
print(df["pressure"].min(), "to", df["pressure"].max())