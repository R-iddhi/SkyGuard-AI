from typing import List, Dict, Optional


# ---------------------------------------------------------
# Spatial consistency configuration
# ---------------------------------------------------------

TEMPERATURE_THRESHOLD = 8.0
PRESSURE_THRESHOLD = 8.0
HUMIDITY_THRESHOLD = 20.0


# ---------------------------------------------------------
# Feature names
# ---------------------------------------------------------

FEATURE_NAMES = {
    "temperature": "Temperature",
    "pressure": "Atmospheric pressure",
    "humidity": "Relative humidity"
}


# ---------------------------------------------------------
# Compare station with neighboring stations
# ---------------------------------------------------------

def check_spatial_consistency(
    station_temperature: float,
    station_pressure: float,
    station_humidity: float,
    neighbors: Optional[List[Dict]] = None
):

    # No neighboring stations available.
    if not neighbors:

        return {
            "spatial_anomaly": False,
            "neighbor_count": 0,
            "temperature_difference": None,
            "pressure_difference": None,
            "humidity_difference": None,
            "anomalous_features": [],
            "summary": (
                "No neighboring station data available "
                "for spatial comparison."
            ),
            "details": []
        }


    # -----------------------------------------------------
    # Remove incomplete neighbor readings
    # -----------------------------------------------------

    valid_neighbors = []

    for neighbor in neighbors:

        if not all(
            key in neighbor
            for key in [
                "temperature",
                "pressure",
                "humidity"
            ]
        ):
            continue

        try:

            valid_neighbors.append({
                "temperature": float(
                    neighbor["temperature"]
                ),

                "pressure": float(
                    neighbor["pressure"]
                ),

                "humidity": float(
                    neighbor["humidity"]
                )
            })

        except (
            TypeError,
            ValueError
        ):

            continue


    if not valid_neighbors:

        return {
            "spatial_anomaly": False,
            "neighbor_count": 0,
            "temperature_difference": None,
            "pressure_difference": None,
            "humidity_difference": None,
            "anomalous_features": [],
            "summary": (
                "No valid neighboring station "
                "readings available."
            ),
            "details": []
        }


    # -----------------------------------------------------
    # Calculate neighbor averages
    # -----------------------------------------------------

    average_temperature = sum(
        neighbor["temperature"]
        for neighbor in valid_neighbors
    ) / len(valid_neighbors)


    average_pressure = sum(
        neighbor["pressure"]
        for neighbor in valid_neighbors
    ) / len(valid_neighbors)


    average_humidity = sum(
        neighbor["humidity"]
        for neighbor in valid_neighbors
    ) / len(valid_neighbors)


    # -----------------------------------------------------
    # Differences from neighbor average
    # -----------------------------------------------------

    temperature_difference = abs(
        station_temperature
        - average_temperature
    )


    pressure_difference = abs(
        station_pressure
        - average_pressure
    )


    humidity_difference = abs(
        station_humidity
        - average_humidity
    )


    # -----------------------------------------------------
    # Detect spatial inconsistencies
    # -----------------------------------------------------

    anomalous_features = []


    if temperature_difference > TEMPERATURE_THRESHOLD:

        anomalous_features.append(
            "temperature"
        )


    if pressure_difference > PRESSURE_THRESHOLD:

        anomalous_features.append(
            "pressure"
        )


    if humidity_difference > HUMIDITY_THRESHOLD:

        anomalous_features.append(
            "humidity"
        )


    spatial_anomaly = (
        len(anomalous_features) > 0
    )


    # -----------------------------------------------------
    # Detailed comparison
    # -----------------------------------------------------

    details = [

        {
            "feature": "temperature",
            "feature_name": FEATURE_NAMES["temperature"],
            "station_value": round(
                station_temperature,
                2
            ),
            "neighbor_average": round(
                average_temperature,
                2
            ),
            "difference": round(
                temperature_difference,
                2
            ),
            "threshold": TEMPERATURE_THRESHOLD,
            "anomaly": (
                temperature_difference
                > TEMPERATURE_THRESHOLD
            )
        },

        {
            "feature": "pressure",
            "feature_name": FEATURE_NAMES["pressure"],
            "station_value": round(
                station_pressure,
                2
            ),
            "neighbor_average": round(
                average_pressure,
                2
            ),
            "difference": round(
                pressure_difference,
                2
            ),
            "threshold": PRESSURE_THRESHOLD,
            "anomaly": (
                pressure_difference
                > PRESSURE_THRESHOLD
            )
        },

        {
            "feature": "humidity",
            "feature_name": FEATURE_NAMES["humidity"],
            "station_value": round(
                station_humidity,
                2
            ),
            "neighbor_average": round(
                average_humidity,
                2
            ),
            "difference": round(
                humidity_difference,
                2
            ),
            "threshold": HUMIDITY_THRESHOLD,
            "anomaly": (
                humidity_difference
                > HUMIDITY_THRESHOLD
            )
        }

    ]


    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    if spatial_anomaly:

        feature_names = [
            FEATURE_NAMES[feature]
            for feature in anomalous_features
        ]

        summary = (
            "Spatial inconsistency detected in: "
            + ", ".join(feature_names)
        )

    else:

        summary = (
            "Station reading is consistent "
            "with neighboring stations."
        )


    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {

        "spatial_anomaly": spatial_anomaly,

        "neighbor_count": len(
            valid_neighbors
        ),

        "neighbor_average": {
            "temperature": round(
                average_temperature,
                2
            ),

            "pressure": round(
                average_pressure,
                2
            ),

            "humidity": round(
                average_humidity,
                2
            )
        },

        "temperature_difference": round(
            temperature_difference,
            2
        ),

        "pressure_difference": round(
            pressure_difference,
            2
        ),

        "humidity_difference": round(
            humidity_difference,
            2
        ),

        "anomalous_features": anomalous_features,

        "summary": summary,

        "details": details
    }


# ---------------------------------------------------------
# Standalone tests
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("SPATIAL CONSISTENCY TESTS")
    print("=" * 70)


    # -----------------------------------------------------
    # TEST 1: Normal spatially consistent reading
    # -----------------------------------------------------

    print("\nTEST 1: Spatially consistent reading")

    neighbors = [

        {
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        },

        {
            "temperature": 21.0,
            "pressure": 971.0,
            "humidity": 52.0
        },

        {
            "temperature": 19.0,
            "pressure": 969.0,
            "humidity": 48.0
        }

    ]


    result = check_spatial_consistency(

        station_temperature=20.5,
        station_pressure=970.0,
        station_humidity=51.0,

        neighbors=neighbors
    )


    print(
        "Spatial anomaly:",
        result["spatial_anomaly"]
    )

    print(
        "Summary:",
        result["summary"]
    )


    # -----------------------------------------------------
    # TEST 2: Temperature spatial anomaly
    # -----------------------------------------------------

    print("\nTEST 2: Temperature spatial anomaly")

    neighbors = [

        {
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        },

        {
            "temperature": 21.0,
            "pressure": 971.0,
            "humidity": 52.0
        },

        {
            "temperature": 19.0,
            "pressure": 969.0,
            "humidity": 48.0
        }

    ]


    result = check_spatial_consistency(

        station_temperature=40.0,
        station_pressure=970.0,
        station_humidity=50.0,

        neighbors=neighbors
    )


    print(
        "Spatial anomaly:",
        result["spatial_anomaly"]
    )

    print(
        "Anomalous features:",
        result["anomalous_features"]
    )

    print(
        "Summary:",
        result["summary"]
    )


    # -----------------------------------------------------
    # TEST 3: Multiple spatial anomalies
    # -----------------------------------------------------

    print("\nTEST 3: Multiple spatial anomalies")

    result = check_spatial_consistency(

        station_temperature=45.0,
        station_pressure=990.0,
        station_humidity=90.0,

        neighbors=neighbors
    )


    print(
        "Spatial anomaly:",
        result["spatial_anomaly"]
    )

    print(
        "Anomalous features:",
        result["anomalous_features"]
    )

    print(
        "Summary:",
        result["summary"]
    )


    # -----------------------------------------------------
    # TEST 4: No neighbors
    # -----------------------------------------------------

    print("\nTEST 4: No neighboring stations")

    result = check_spatial_consistency(

        station_temperature=20.0,
        station_pressure=970.0,
        station_humidity=50.0,

        neighbors=[]
    )


    print(
        "Spatial anomaly:",
        result["spatial_anomaly"]
    )

    print(
        "Neighbor count:",
        result["neighbor_count"]
    )

    print(
        "Summary:",
        result["summary"]
    )


    print("\n" + "=" * 70)
    print("SPATIAL CONSISTENCY TESTS COMPLETE")
    print("=" * 70)