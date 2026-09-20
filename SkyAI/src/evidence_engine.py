from typing import Dict, List


def calculate_evidence(
    physical_anomaly: bool = False,
    temporal_anomaly: bool = False,
    frozen_anomaly: bool = False,
    multivariate_anomaly: bool = False,
    communication_anomaly: bool = False,
    seasonal_anomaly: bool = False,
    spatial_anomaly: bool = False,
    ml_anomaly: bool = False
) -> Dict:

    evidence_score = 0

    evidence_details: List[str] = []

    # ---------------------------------------------------------
    # 1. Physical validation
    # ---------------------------------------------------------

    if physical_anomaly:

        evidence_score += 40

        evidence_details.append(
            "Physical/range violation detected."
        )

    # ---------------------------------------------------------
    # 2. Temporal validation
    # ---------------------------------------------------------

    if temporal_anomaly:

        evidence_score += 10

        evidence_details.append(
            "Sudden temporal change detected."
        )

    # ---------------------------------------------------------
    # 3. Frozen sensor
    # ---------------------------------------------------------

    if frozen_anomaly:

        evidence_score += 40

        evidence_details.append(
            "Sensor value remained unchanged for "
            "multiple consecutive observations."
        )

    # ---------------------------------------------------------
    # 4. Multivariate consistency
    # ---------------------------------------------------------

    if multivariate_anomaly:

        evidence_score += 40

        evidence_details.append(
            "Temperature, pressure, and humidity "
            "showed an inconsistent combination."
        )

    # ---------------------------------------------------------
    # 5. Communication
    # ---------------------------------------------------------

    if communication_anomaly:

        evidence_score += 40

        evidence_details.append(
            "Communication/data transmission gap detected."
        )

    # ---------------------------------------------------------
    # 6. Seasonal pattern
    # ---------------------------------------------------------

    if seasonal_anomaly:

        evidence_score += 10

        evidence_details.append(
            "Reading deviates from the learned seasonal pattern."
        )

    # ---------------------------------------------------------
    # 7. Spatial consistency
    # ---------------------------------------------------------

    if spatial_anomaly:

        evidence_score += 20

        evidence_details.append(
            "Station reading is inconsistent with "
            "neighboring stations."
        )

    # ---------------------------------------------------------
    # 8. Machine learning
    # ---------------------------------------------------------

    if ml_anomaly:

        evidence_score += 20

        evidence_details.append(
            "Isolation Forest identified the reading "
            "as unusual."
        )

    # ---------------------------------------------------------
    # Limit score
    # ---------------------------------------------------------

    evidence_score = min(
        evidence_score,
        100
    )

    # ---------------------------------------------------------
    # Determine evidence level
    # ---------------------------------------------------------

    if evidence_score >= 70:

        evidence_level = "HIGH"

    elif evidence_score >= 40:

        evidence_level = "MEDIUM"

    elif evidence_score > 0:

        evidence_level = "LOW"

    else:

        evidence_level = "NONE"

    # ---------------------------------------------------------
    # Determine interpretation
    # ---------------------------------------------------------

    if (
        physical_anomaly
        or frozen_anomaly
        or communication_anomaly
        or multivariate_anomaly
    ):

        interpretation = (
            "Strong evidence of a sensor or data-quality problem."
        )

    elif (
        spatial_anomaly
        and temporal_anomaly
    ):

        interpretation = (
            "Reading shows both temporal and spatial inconsistency."
        )

    elif spatial_anomaly:

        interpretation = (
            "Reading differs from neighboring stations "
            "and may represent a station-specific problem."
        )

    elif temporal_anomaly:

        interpretation = (
            "Sudden change detected; additional context "
            "is required to determine whether it is a genuine "
            "weather event."
        )

    elif seasonal_anomaly:

        interpretation = (
            "Reading differs from the learned seasonal pattern, "
            "but seasonal deviation alone is not sufficient "
            "to declare a sensor fault."
        )

    elif ml_anomaly:

        interpretation = (
            "Machine learning detected an unusual observation, "
            "but additional evidence is required."
        )

    else:

        interpretation = (
            "No strong evidence of a sensor or data-quality problem."
        )

    return {

        "evidence_score":
            evidence_score,

        "evidence_level":
            evidence_level,

        "evidence_details":
            evidence_details,

        "interpretation":
            interpretation
    }


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SKYGUARD AI - EVIDENCE ENGINE TESTS")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: Completely normal reading
    # ---------------------------------------------------------

    result = calculate_evidence()

    print()
    print("TEST 1: Normal Reading")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 0
    assert result["evidence_level"] == "NONE"

    print("✅ TEST 1 PASSED")

    # ---------------------------------------------------------
    # TEST 2: Sudden temporal change only
    # ---------------------------------------------------------

    result = calculate_evidence(
        temporal_anomaly=True
    )

    print()
    print("TEST 2: Sudden Change")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 10
    assert result["evidence_level"] == "LOW"

    print("✅ TEST 2 PASSED")

    # ---------------------------------------------------------
    # TEST 3: Invalid physical value
    # ---------------------------------------------------------

    result = calculate_evidence(
        physical_anomaly=True
    )

    print()
    print("TEST 3: Physical Anomaly")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 40
    assert result["evidence_level"] == "MEDIUM"

    print("✅ TEST 3 PASSED")

    # ---------------------------------------------------------
    # TEST 4: Frozen sensor
    # ---------------------------------------------------------

    result = calculate_evidence(
        frozen_anomaly=True
    )

    print()
    print("TEST 4: Frozen Sensor")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 40
    assert result["evidence_level"] == "MEDIUM"

    print("✅ TEST 4 PASSED")

    # ---------------------------------------------------------
    # TEST 5: Multiple independent signals
    # ---------------------------------------------------------

    result = calculate_evidence(
        temporal_anomaly=True,
        spatial_anomaly=True,
        ml_anomaly=True
    )

    print()
    print("TEST 5: Multiple Evidence Sources")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 50
    assert result["evidence_level"] == "MEDIUM"

    print("✅ TEST 5 PASSED")

    # ---------------------------------------------------------
    # TEST 6: Strong sensor fault evidence
    # ---------------------------------------------------------

    result = calculate_evidence(
        physical_anomaly=True,
        frozen_anomaly=True,
        communication_anomaly=True
    )

    print()
    print("TEST 6: Strong Sensor Fault Evidence")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 100
    assert result["evidence_level"] == "HIGH"

    print("✅ TEST 6 PASSED")

    # ---------------------------------------------------------
    # TEST 7: Spatial + temporal inconsistency
    # ---------------------------------------------------------

    result = calculate_evidence(
        temporal_anomaly=True,
        spatial_anomaly=True
    )

    print()
    print("TEST 7: Spatial + Temporal Evidence")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 30
    assert result["evidence_level"] == "LOW"

    print("✅ TEST 7 PASSED")

    # ---------------------------------------------------------
    # TEST 8: Seasonal deviation alone
    # ---------------------------------------------------------

    result = calculate_evidence(
        seasonal_anomaly=True
    )

    print()
    print("TEST 8: Seasonal Deviation")
    print("Evidence score:", result["evidence_score"])
    print("Evidence level:", result["evidence_level"])
    print("Interpretation:")
    print(result["interpretation"])

    assert result["evidence_score"] == 10
    assert result["evidence_level"] == "LOW"

    print("✅ TEST 8 PASSED")

    print()
    print("=" * 70)
    print("ALL EVIDENCE ENGINE TESTS PASSED")
    print("=" * 70)