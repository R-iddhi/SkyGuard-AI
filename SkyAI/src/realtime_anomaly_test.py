import time

import requests


# ==================================================
# API CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000/detect"

STATION_ID = "ANOMALY_TEST_STATION"

DELAY_SECONDS = 0.5


# ==================================================
# TEST DATA
# ==================================================

test_readings = [

    # --------------------------------------------------
    # 1. TEMPERATURE SPIKE
    # --------------------------------------------------

    {
        "name": "Temperature Spike",

        "temperature": 60.0,

        "pressure": 970.0,

        "humidity": 50.0,

        "wind_speed": 20.0,

        "wind_direction": 180.0
    },


    # --------------------------------------------------
    # 2. TEMPERATURE DROP
    # --------------------------------------------------

    {
        "name": "Temperature Drop",

        "temperature": -60.0,

        "pressure": 970.0,

        "humidity": 50.0,

        "wind_speed": 20.0,

        "wind_direction": 180.0
    },


    # --------------------------------------------------
    # 3. INVALID HUMIDITY
    # --------------------------------------------------

    {
        "name": "Invalid Humidity",

        "temperature": 20.0,

        "pressure": 970.0,

        "humidity": 135.0,

        "wind_speed": 20.0,

        "wind_direction": 180.0
    },


    # --------------------------------------------------
    # 4. MULTIVARIATE ANOMALY
    # --------------------------------------------------

    {
        "name": "Multivariate Anomaly",

        "temperature": 50.0,

        "pressure": 951.5,

        "humidity": 34.0,

        "wind_speed": 52.5,

        "wind_direction": 312.0
    }

]


# ==================================================
# HEADER
# ==================================================

print("=" * 70)

print("SKYGUARD AI - REAL-TIME ANOMALY TEST")

print("=" * 70)

print()

print(
    f"Testing {len(test_readings)} anomaly scenarios..."
)

print()


# ==================================================
# TEST COUNTERS
# ==================================================

passed = 0

failed = 0


# ==================================================
# SEND TEST READINGS
# ==================================================

for number, reading in enumerate(
    test_readings,
    start=1
):

    payload = {

        "station_id": STATION_ID,

        "timestamp": (
            f"2026-09-07T12:{number:02d}:00"
        ),

        "temperature": reading["temperature"],

        "pressure": reading["pressure"],

        "humidity": reading["humidity"],

        "wind_speed": reading["wind_speed"],

        "wind_direction": reading["wind_direction"]

    }


    print(
        f"TEST {number}: {reading['name']}"
    )


    try:

        response = requests.post(

            API_URL,

            json=payload,

            timeout=10

        )


        if response.status_code != 200:

            print(
                f"❌ API ERROR: {response.status_code}"
            )

            failed += 1

            continue


        result = response.json()


        # --------------------------------------------------
        # Check whether anomaly was detected
        # --------------------------------------------------

        if result["status"] == "ANOMALY":

            print(
                "✅ ANOMALY DETECTED"
            )

            print(
                f"   Type: {result['anomaly_type']}"
            )

            print(
                f"   Severity: {result['severity']}"
            )

            print(
                f"   Evidence Score: "
                f"{result['evidence_score']}"
            )

            print(
                f"   ML Detection: "
                f"{result['ml_detection']}"
            )


            # --------------------------------------------------
            # SHAP information
            # --------------------------------------------------

            if result["shap_top_feature"]:

                print(
                    f"   SHAP Top Feature: "
                    f"{result['shap_top_feature']}"
                )

                print(
                    f"   SHAP Summary: "
                    f"{result['shap_summary']}"
                )


            passed += 1


        else:

            print(
                "❌ ANOMALY NOT DETECTED"
            )

            failed += 1


    except requests.exceptions.ConnectionError:

        print(
            "❌ Could not connect to FastAPI."
        )

        print(
            "   Make sure Uvicorn is running."
        )

        failed += 1


    except Exception as e:

        print(
            f"❌ ERROR: {e}"
        )

        failed += 1


    print()

    time.sleep(
        DELAY_SECONDS
    )


# ==================================================
# FINAL RESULT
# ==================================================

print("=" * 70)

print("TEST SUMMARY")

print("=" * 70)

print()

print(
    f"Tests passed: {passed}"
)

print(
    f"Tests failed: {failed}"
)

print()

if failed == 0:

    print(
        "🎉 ALL ANOMALY SCENARIOS DETECTED SUCCESSFULLY"
    )

else:

    print(
        "⚠️ SOME ANOMALY SCENARIOS WERE NOT DETECTED"
    )

print()

print("=" * 70)