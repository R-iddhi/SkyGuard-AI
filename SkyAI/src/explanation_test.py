import requests


# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "http://127.0.0.1:8000/detect"


# ==================================================
# TEST READING
# ==================================================

payload = {

    "station_id": "EXPLANATION_TEST",

    "timestamp": "2016-09-09T17:00:00",

    "temperature": 41.6,

    "pressure": 946.7,

    "humidity": 100.0,

    "wind_speed": 69.8,

    "wind_direction": 50.0
}


# ==================================================
# SEND REQUEST
# ==================================================

print("=" * 70)
print("SKYGUARD AI - EXPLANATION TEST")
print("=" * 70)
print()

print("Sending suspicious AWS observation...")
print()

response = requests.post(
    API_URL,
    json=payload,
    timeout=10
)


# ==================================================
# CHECK RESPONSE
# ==================================================

if response.status_code != 200:

    print(
        f"API ERROR: {response.status_code}"
    )

    print(response.text)

else:

    result = response.json()


    # ==================================================
    # BASIC RESULT
    # ==================================================

    print("DETECTION RESULT")
    print("-" * 70)

    print(
        f"Status: {result['status']}"
    )

    print(
        f"Anomaly Type: {result['anomaly_type']}"
    )

    print(
        f"Severity: {result['severity']}"
    )

    print(
        f"Root Cause: {result['root_cause']}"
    )

    print(
        f"Evidence Score: {result['evidence_score']}"
    )

    print()


    # ==================================================
    # DETECTION REASONS
    # ==================================================

    print("DETECTION REASONS")
    print("-" * 70)

    for reason in result["reasons"]:

        print(
            f"✓ {reason}"
        )

    print()


    # ==================================================
    # DETAILED EVIDENCE
    # ==================================================

    print("DETAILED EVIDENCE")
    print("-" * 70)

    for evidence in result["evidence_details"]:

        print(
            f"✓ {evidence}"
        )

    print()


    # ==================================================
    # MACHINE LEARNING
    # ==================================================

    print("MACHINE LEARNING")
    print("-" * 70)

    print(
        f"Isolation Forest anomaly: "
        f"{result['ml_detection']}"
    )

    print(
        f"Isolation Forest score: "
        f"{result['ml_score']}"
    )

    print()


    # ==================================================
    # SHAP EXPLANATION
    # ==================================================

    print("SHAP EXPLANATION")
    print("-" * 70)

    if result["shap_top_feature"]:

        print(
            f"Strongest feature: "
            f"{result['shap_top_feature']}"
        )

        print()

        print(
            f"Summary: "
            f"{result['shap_summary']}"
        )

        print()

        for item in result["shap_explanation"]:

            print(
                f"{item['feature_name']}: "
                f"SHAP={item['shap_value']:.6f} | "
                f"Importance={item['importance']:.6f} | "
                f"{item['contribution']}"
            )

    else:

        print(
            "SHAP explanation is not available "
            "because Isolation Forest did not "
            "classify this observation as an anomaly."
        )


print()
print("=" * 70)
print("EXPLANATION TEST COMPLETED")
print("=" * 70)