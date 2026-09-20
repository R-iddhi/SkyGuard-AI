import requests


API_URL = "http://127.0.0.1:8000/detect"


def send_test(name, data, expected_status, expected_type):
    response = requests.post(API_URL, json=data)

    if response.status_code != 200:
        print(f"❌ {name} FAILED - HTTP {response.status_code}")
        return False

    result = response.json()

    status_ok = result["status"] == expected_status
    type_ok = result["anomaly_type"] == expected_type

    if status_ok and type_ok:
        print(f"✅ {name} PASSED")
        return True

    print(f"❌ {name} FAILED")
    print(f"Expected: {expected_status} / {expected_type}")
    print(
        f"Received: "
        f"{result['status']} / {result['anomaly_type']}"
    )

    return False


normal = {
    "station_id": "AUTO_NORMAL",
    "timestamp": "2026-09-06T23:40:00",
    "temperature": 20,
    "pressure": 970,
    "humidity": 50,
    "wind_speed": 20,
    "wind_direction": 180
}


temperature_normal = {
    "station_id": "AUTO_TEMP",
    "timestamp": "2026-09-06T23:41:00",
    "temperature": 20,
    "pressure": 970,
    "humidity": 50,
    "wind_speed": 20,
    "wind_direction": 180
}


temperature_anomaly = {
    "station_id": "AUTO_TEMP",
    "timestamp": "2026-09-06T23:42:00",
    "temperature": 40,
    "pressure": 970,
    "humidity": 50,
    "wind_speed": 20,
    "wind_direction": 180
}


invalid_humidity = {
    "station_id": "AUTO_RANGE",
    "timestamp": "2026-09-06T23:43:00",
    "temperature": 20,
    "pressure": 970,
    "humidity": 135,
    "wind_speed": 20,
    "wind_direction": 180
}


multivariate_normal = {
    "station_id": "AUTO_MULTI",
    "timestamp": "2026-09-06T23:44:00",
    "temperature": 20,
    "pressure": 1012,
    "humidity": 60,
    "wind_speed": 20,
    "wind_direction": 180
}


multivariate_anomaly = {
    "station_id": "AUTO_MULTI",
    "timestamp": "2026-09-06T23:45:00",
    "temperature": 50,
    "pressure": 951,
    "humidity": 34,
    "wind_speed": 52.5,
    "wind_direction": 312
}


passed = 0
total = 6


if send_test(
    "Normal Reading",
    normal,
    "NORMAL",
    "Normal"
):
    passed += 1


if send_test(
    "Temperature Baseline",
    temperature_normal,
    "NORMAL",
    "Normal"
):
    passed += 1


if send_test(
    "Sudden Temperature Change",
    temperature_anomaly,
    "SUSPICIOUS",
    "Sudden Change"
):
    passed += 1


if send_test(
    "Invalid Humidity",
    invalid_humidity,
    "ANOMALY",
    "Physical/Range Error"
):
    passed += 1


if send_test(
    "Multivariate Anomaly",
    multivariate_normal,
    "NORMAL",
    "Normal"
):
    passed += 1


if send_test(
    "Multivariate Detection",
    multivariate_anomaly,
    "ANOMALY",
    "Multivariate Inconsistency"
):
    passed += 1


print()
print("=" * 50)
print(f"Tests passed: {passed}/{total}")
print("=" * 50)