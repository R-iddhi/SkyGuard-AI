import xarray as xr
import matplotlib.pyplot as plt

file_path = "data/raw/aws_data.nc"

data = xr.open_dataset(file_path)

variables = ["tempr", "rh", "ws", "wd", "ap"]

for variable in variables:

    plt.figure(figsize=(10, 5))

    plt.hist(data[variable].values, bins=50)

    plt.title(f"Distribution of {variable}")
    plt.xlabel(variable)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.show()