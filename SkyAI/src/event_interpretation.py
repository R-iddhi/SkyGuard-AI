from typing import Dict, List


def interpret_event(
    status: str,
    evidence_score: int,
    physical_anomaly: bool = False,
    temporal_anomaly: bool = False,
    frozen_anomaly: bool = False,
    multivariate_anomaly: bool = False,
    communication_anomaly: bool = False,
    seasonal_anomaly: bool = False,
    spatial_anomaly: bool = False,
    ml_anomaly: bool = False
) -> Dict:

    supporting_evidence: List[str] = []

    # ---------------------------------------------------------
    # Strong direct evidence of sensor/data problems
    # ---------------------------------------------------------

    if physical_anomaly:

        supporting_evidence.append(
            "Physical/range violation"
        )

    if frozen_anomaly:

        supporting_evidence.append(
            "Frozen sensor behavior"
        )

    if communication_anomaly:

        supporting_evidence.append(
            "Communication/data transmission gap"
        )

    # ---------------------------------------------------------
    # Strong consistency evidence
    # ---------------------------------------------------------

    if multivariate_anomaly:

        supporting_evidence.append(
            "Multivariate inconsistency"
        )

    if spatial_anomaly:

        supporting_evidence.append(
            "Spatial inconsistency with neighboring stations"
        )

    # ---------------------------------------------------------
    # Temporal evidence
    # ---------------------------------------------------------

    if temporal_anomaly:

        supporting_evidence.append(
            "Sudden temporal change"
        )

    # ---------------------------------------------------------
    # Seasonal evidence
    # ---------------------------------------------------------

    if seasonal_anomaly:

        supporting_evidence.append(
            "Seasonal pattern deviation"
        )

    # ---------------------------------------------------------
    # Machine-learning evidence
    # ---------------------------------------------------------

    if ml_anomaly:

        supporting_evidence.append(
            "Machine-learning anomaly detection"
        )

    # ---------------------------------------------------------
    # Decision logic
    # ---------------------------------------------------------

    # A physical violation, frozen sensor, or communication
    # failure is direct evidence of a data/sensor problem.

    if (
        physical_anomaly
        or frozen_anomaly
        or communication_anomaly
    ):

        classification = "LIKELY_SENSOR_OR_DATA_PROBLEM"

        interpretation = (
            "The reading contains direct evidence of a "
            "sensor or data-quality problem."
        )

        warning = (
            "The observation should be reviewed before "
            "being treated as a genuine weather event."
        )

    # Multivariate inconsistency indicates that the
    # measured variables do not agree with each other.

    elif multivariate_anomaly:

        classification = "LIKELY_SENSOR_OR_DATA_PROBLEM"

        interpretation = (
            "The combination of temperature, pressure, "
            "and humidity is inconsistent with the expected "
            "multivariate relationship."
        )

        warning = (
            "Check the sensor and data stream before "
            "accepting the reading as a meteorological event."
        )

    # Spatial inconsistency is particularly useful when
    # neighboring stations are available.

    elif spatial_anomaly:

        classification = "POSSIBLE_SENSOR_OR_LOCAL_EVENT"

        interpretation = (
            "The station differs from neighboring observations. "
            "This may indicate a station-specific sensor problem "
            "or a localized meteorological event."
        )

        warning = (
            "Additional temporal and meteorological context "
            "is required before determining the cause."
        )

    # A sudden change by itself does not prove sensor failure.

    elif temporal_anomaly:

        classification = "POSSIBLE_METEOROLOGICAL_EVENT"

        interpretation = (
            "A sudden change was detected, but there is no "
            "direct evidence of a sensor or data-quality failure."
        )

        warning = (
            "The change may represent a genuine weather event; "
            "additional observations should be monitored."
        )

    # Seasonal deviation alone is weak evidence.

    elif seasonal_anomaly:

        classification = "UNUSUAL_WEATHER_CONDITION"

        interpretation = (
            "The observation differs from the learned seasonal "
            "pattern, but no direct sensor-failure evidence "
            "was detected."
        )

        warning = (
            "Seasonal deviation alone should not be treated "
            "as proof of sensor failure."
        )

    # ML anomaly alone is also insufficient to identify
    # a sensor fault.

    elif ml_anomaly:

        classification = "UNUSUAL_OBSERVATION"

        interpretation = (
            "The machine-learning model identified an unusual "
            "observation, but additional physical and temporal "
            "evidence is unavailable."
        )

        warning = (
            "Machine-learning anomaly detection alone does not "
            "establish whether the cause is weather or sensor error."
        )

    else:

        classification = "LIKELY_NORMAL"

        interpretation = (
            "No significant evidence of a sensor/data problem "
            "or unusual observation was detected."
        )

        warning = (
            "Continue normal monitoring."
        )

    # ---------------------------------------------------------
    # Evidence-strength summary
    # ---------------------------------------------------------

    if evidence_score >= 70:

        evidence_strength = "HIGH"

    elif evidence_score >= 40:

        evidence_strength = "MEDIUM"

    elif evidence_score > 0:

        evidence_strength = "LOW"

    else:

        evidence_strength = "NONE"

    return {

        "classification":
            classification,

        "interpretation":
            interpretation,

        "evidence_strength":
            evidence_strength,

        "evidence_score":
            evidence_score,

        "supporting_evidence":
            supporting_evidence,

        "warning":
            warning
    }


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("SKYGUARD AI - EVENT INTERPRETATION TESTS")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: Normal observation
    # ---------------------------------------------------------

    result = interpret_event(
        status="NORMAL",
        evidence_score=0
    )

    print()
    print("TEST 1: Normal Observation")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])
    print("Interpretation:", result["interpretation"])

    assert (
        result["classification"]
        == "LIKELY_NORMAL"
    )

    assert (
        result["evidence_strength"]
        == "NONE"
    )

    print("✅ TEST 1 PASSED")

    # ---------------------------------------------------------
    # TEST 2: Physical sensor/data problem
    # ---------------------------------------------------------

    result = interpret_event(
        status="ANOMALY",
        evidence_score=40,
        physical_anomaly=True
    )

    print()
    print("TEST 2: Physical/Data Problem")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])
    print("Interpretation:", result["interpretation"])

    assert (
        result["classification"]
        == "LIKELY_SENSOR_OR_DATA_PROBLEM"
    )

    print("✅ TEST 2 PASSED")

    # ---------------------------------------------------------
    # TEST 3: Frozen sensor
    # ---------------------------------------------------------

    result = interpret_event(
        status="ANOMALY",
        evidence_score=40,
        frozen_anomaly=True
    )

    print()
    print("TEST 3: Frozen Sensor")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])

    assert (
        result["classification"]
        == "LIKELY_SENSOR_OR_DATA_PROBLEM"
    )

    print("✅ TEST 3 PASSED")

    # ---------------------------------------------------------
    # TEST 4: Communication failure
    # ---------------------------------------------------------

    result = interpret_event(
        status="ANOMALY",
        evidence_score=40,
        communication_anomaly=True
    )

    print()
    print("TEST 4: Communication Failure")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])

    assert (
        result["classification"]
        == "LIKELY_SENSOR_OR_DATA_PROBLEM"
    )

    print("✅ TEST 4 PASSED")

    # ---------------------------------------------------------
    # TEST 5: Multivariate inconsistency
    # ---------------------------------------------------------

    result = interpret_event(
        status="ANOMALY",
        evidence_score=40,
        multivariate_anomaly=True
    )

    print()
    print("TEST 5: Multivariate Inconsistency")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])

    assert (
        result["classification"]
        == "LIKELY_SENSOR_OR_DATA_PROBLEM"
    )

    print("✅ TEST 5 PASSED")

    # ---------------------------------------------------------
    # TEST 6: Spatial inconsistency
    # ---------------------------------------------------------

    result = interpret_event(
        status="SUSPICIOUS",
        evidence_score=20,
        spatial_anomaly=True
    )

    print()
    print("TEST 6: Spatial Inconsistency")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])
    print("Interpretation:", result["interpretation"])

    assert (
        result["classification"]
        == "POSSIBLE_SENSOR_OR_LOCAL_EVENT"
    )

    print("✅ TEST 6 PASSED")

    # ---------------------------------------------------------
    # TEST 7: Sudden change only
    # ---------------------------------------------------------

    result = interpret_event(
        status="SUSPICIOUS",
        evidence_score=10,
        temporal_anomaly=True
    )

    print()
    print("TEST 7: Sudden Change Only")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])
    print("Interpretation:", result["interpretation"])

    assert (
        result["classification"]
        == "POSSIBLE_METEOROLOGICAL_EVENT"
    )

    print("✅ TEST 7 PASSED")

    # ---------------------------------------------------------
    # TEST 8: Seasonal deviation only
    # ---------------------------------------------------------

    result = interpret_event(
        status="NORMAL",
        evidence_score=10,
        seasonal_anomaly=True
    )

    print()
    print("TEST 8: Seasonal Deviation")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])

    assert (
        result["classification"]
        == "UNUSUAL_WEATHER_CONDITION"
    )

    print("✅ TEST 8 PASSED")

    # ---------------------------------------------------------
    # TEST 9: ML anomaly only
    # ---------------------------------------------------------

    result = interpret_event(
        status="NORMAL",
        evidence_score=20,
        ml_anomaly=True
    )

    print()
    print("TEST 9: ML Anomaly Only")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])

    assert (
        result["classification"]
        == "UNUSUAL_OBSERVATION"
    )

    print("✅ TEST 9 PASSED")

    # ---------------------------------------------------------
    # TEST 10: Strong combined evidence
    # ---------------------------------------------------------

    result = interpret_event(
        status="ANOMALY",
        evidence_score=100,
        physical_anomaly=True,
        temporal_anomaly=True,
        spatial_anomaly=True,
        ml_anomaly=True
    )

    print()
    print("TEST 10: Strong Combined Evidence")
    print("Classification:", result["classification"])
    print("Evidence strength:", result["evidence_strength"])
    print("Supporting evidence:")
    
    for evidence in result["supporting_evidence"]:
        print("-", evidence)

    assert (
        result["classification"]
        == "LIKELY_SENSOR_OR_DATA_PROBLEM"
    )

    assert (
        result["evidence_strength"]
        == "HIGH"
    )

    print("✅ TEST 10 PASSED")

    print()
    print("=" * 70)
    print("ALL EVENT INTERPRETATION TESTS PASSED")
    print("=" * 70)