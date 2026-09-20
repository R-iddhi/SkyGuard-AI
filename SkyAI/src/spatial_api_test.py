import requests


URL = "http://127.0.0.1:8000/detect"


payload = {
    "station_id": "SPATIAL-TEST",
    "timestamp": "2026-09-20T10:00:00",
    "temperature": 40.0,
    "pressure": 970.0,
    "humidity": 50.0,
    "wind_speed": 20.0,
    "wind_direction": 180.0,

    "neighbors": [
        {
            "station_id": "NEIGHBOR-1",
            "temperature": 20.0,
            "pressure": 970.0,
            "humidity": 50.0
        },
        {
            "station_id": "NEIGHBOR-2",
            "temperature": 21.0,
            "pressure": 971.0,
            "humidity": 51.0
        },
        {
            "station_id": "NEIGHBOR-3",
            "temperature": 19.0,
            "pressure": 969.0,
            "humidity": 49.0
        }
    ]
}


response = requests.post(
    URL,
    json=payload
)


print()
print("=" * 70)
print("SPATIAL API TEST")
print("=" * 70)

print()
print("HTTP status:", response.status_code)

if response.status_code != 200:

    print()
    print("❌ Spatial API test failed")
    print(response.text)

else:

    result = response.json()

    print()
    print("Station ID:", result["station_id"])
    print("Status:", result["status"])
    print("Anomaly Type:", result["anomaly_type"])
    print("Severity:", result["severity"])

    print()
    print("Spatial Analysis:")
    print(
        "Spatial anomaly:",
        result["spatial_analysis"]["spatial_anomaly"]
    )

    print(
        "Neighbor count:",
        result["spatial_analysis"]["neighbor_count"]
    )

    print(
        "Anomalous features:",
        result["spatial_analysis"]["anomalous_features"]
    )

    print(
        "Neighbor average:",
        result["spatial_analysis"]["neighbor_average"]
    )

    print(
        "Temperature difference:",
        result["spatial_analysis"]["temperature_difference"]
    )

    print()
    print(
        "Summary:",
        result["spatial_analysis"]["summary"]
    )

    print()
    print("Reasons:")

    for reason in result["reasons"]:
        print("-", reason)

    print()
    print("=" * 70)

    if (
        result["spatial_analysis"]["spatial_anomaly"]
        and "temperature"
        in result["spatial_analysis"]["anomalous_features"]
    ):
        print("✅ SPATIAL API TEST PASSED")
    else:
        print("❌ SPATIAL API TEST FAILED")

    print("=" * 70)