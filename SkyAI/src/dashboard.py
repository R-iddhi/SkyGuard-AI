from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SkyGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS / API
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "aws_clean_final_benchmark.csv"
)

API_URL = "http://127.0.0.1:8000"

DETECT_URL = f"{API_URL}/detect"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07111f;
        color: #e8eef7;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1728;
        border-right: 1px solid #1c2d44;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
        letter-spacing: -1px;
        color: #ffffff;
        margin-bottom: 2px;
    }

    .main-subtitle {
        font-size: 14px;
        color: #8fa4bd;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 3px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #7f94ad;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #7890aa;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .module-item {
        font-size: 13px;
        color: #c8d5e5;
        padding: 5px 0;
    }

    .panel {
        background-color: #0c1a2c;
        border: 1px solid #1d3048;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 10px;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .panel-subtitle {
        font-size: 12px;
        color: #8298b1;
        margin-bottom: 8px;
    }

    .online-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        background-color: #0d3529;
        color: #58e6a7;
        font-size: 12px;
        font-weight: 700;
    }

    .offline-pill {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 20px;
        background-color: #3a1717;
        color: #ff7777;
        font-size: 12px;
        font-weight: 700;
    }

    .alert-box {
        background-color: #30151a;
        border: 1px solid #7f2734;
        border-left: 5px solid #ff4d5e;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .alert-title {
        color: #ff6675;
        font-size: 18px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .alert-text {
        color: #dce5ef;
        font-size: 14px;
        line-height: 1.8;
    }

    .status-normal {
        background-color: #0d3529;
        border: 1px solid #1c7254;
        color: #63e7ae;
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .status-suspicious {
        background-color: #3b3012;
        border: 1px solid #806c20;
        color: #ffd95a;
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
        margin-bottom: 15px;
    }

    .pipeline {
        background-color: #0c1a2c;
        border: 1px solid #1d3048;
        border-radius: 12px;
        padding: 15px;
        overflow-x: auto;
        white-space: nowrap;
    }

    .pipeline-step {
        display: inline-block;
        background-color: #14263c;
        border: 1px solid #29435f;
        color: #cbd8e7;
        padding: 7px 10px;
        border-radius: 7px;
        font-size: 11px;
    }

    .pipeline-arrow {
        color: #58718d;
        padding: 0 7px;
        font-size: 13px;
    }

    .info-row {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        padding: 9px 0;
        border-bottom: 1px solid #1b2c42;
    }

    .info-label {
        color: #8095ad;
        font-size: 12px;
    }

    .info-value {
        color: #e5edf7;
        font-size: 12px;
        font-weight: 600;
        text-align: right;
    }

    .network-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #1b2c42;
    }

    .network-name {
        color: #d8e2ee;
        font-size: 13px;
    }

    .network-online {
        color: #58e6a7;
        font-size: 11px;
        font-weight: 800;
    }

    .network-warning {
        color: #ffd35c;
        font-size: 11px;
        font-weight: 800;
    }

    .footer {
        text-align: center;
        color: #61778f;
        font-size: 12px;
        padding: 30px 0;
        border-top: 1px solid #1a2b40;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "current_reading" not in st.session_state:
    st.session_state.current_reading = None

if "current_result" not in st.session_state:
    st.session_state.current_result = None

if "history" not in st.session_state:
    st.session_state.history = []

if "readings_processed" not in st.session_state:
    st.session_state.readings_processed = 0

if "alerts" not in st.session_state:
    st.session_state.alerts = 0

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "last_api_status" not in st.session_state:
    st.session_state.last_api_status = False


# ============================================================
# LOAD DATASET
# ============================================================

try:

    df = pd.read_csv(DATA_PATH)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce"
        )

except Exception:

    df = pd.DataFrame()


# ============================================================
# API FUNCTIONS
# ============================================================

def check_backend():

    try:

        response = requests.get(
            f"{API_URL}/docs",
            timeout=2
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def get_maintenance_monitoring():
    """
    Fetch dynamic sensor degradation and maintenance
    monitoring information from the FastAPI backend.
    """

    try:

        response = requests.get(
            f"{API_URL}/",
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        maintenance = data.get(
            "maintenance_monitoring"
        )

        if maintenance is None:

            return {
                "available": False
            }

        maintenance["available"] = True

        return maintenance

    except requests.RequestException:

        return {
            "available": False
        }

    except Exception:

        return {
            "available": False
        }


def send_reading(
    reading,
    station_id="AWS-DEMO-001"
):

    payload = {
        "station_id": station_id,
        "timestamp": reading["timestamp"],
        "temperature": float(reading["temperature"]),
        "pressure": float(reading["pressure"]),
        "humidity": float(reading["humidity"]),
        "wind_speed": float(reading["wind_speed"]),
        "wind_direction": float(reading["wind_direction"])
    }

    response = requests.post(
        DETECT_URL,
        json=payload,
        timeout=10
    )

    response.raise_for_status()

    result = response.json()

    st.session_state.readings_processed += 1

    if result.get("status") == "ANOMALY":
        st.session_state.alerts += 1

    return result


def add_history(
    reading,
    result
):

    history_row = {
        "timestamp": pd.to_datetime(
            reading["timestamp"]
        ),
        "temperature": float(
            reading["temperature"]
        ),
        "pressure": float(
            reading["pressure"]
        ),
        "humidity": float(
            reading["humidity"]
        ),
        "wind_speed": float(
            reading["wind_speed"]
        ),
        "wind_direction": float(
            reading["wind_direction"]
        ),
        "status": result.get(
            "status",
            "NORMAL"
        ),
        "anomaly_type": result.get(
            "anomaly_type",
            "Normal"
        )
    }

    st.session_state.history.append(
        history_row
    )

    if len(st.session_state.history) > 100:
        st.session_state.history = (
            st.session_state.history[-100:]
        )


# ============================================================
# READING GENERATORS
# ============================================================

def create_manual_reading(
    scenario
):

    timestamp = datetime.now().isoformat()

    if scenario == "Temperature Spike":

        return {
            "timestamp": timestamp,
            "temperature": 60.0,
            "pressure": 970.0,
            "humidity": 50.0,
            "wind_speed": 20.0,
            "wind_direction": 180.0
        }

    if scenario == "Temperature Drop":

        return {
            "timestamp": timestamp,
            "temperature": -60.0,
            "pressure": 970.0,
            "humidity": 50.0,
            "wind_speed": 20.0,
            "wind_direction": 180.0
        }

    if scenario == "Invalid Humidity":

        return {
            "timestamp": timestamp,
            "temperature": -15.0,
            "pressure": 970.0,
            "humidity": 135.0,
            "wind_speed": 20.0,
            "wind_direction": 180.0
        }

    if scenario == "Multivariate Anomaly":

        return {
            "timestamp": timestamp,
            "temperature": 50.0,
            "pressure": 951.0,
            "humidity": 95.0,
            "wind_speed": 50.0,
            "wind_direction": 310.0
        }

    return {
        "timestamp": timestamp,
        "temperature": -15.0,
        "pressure": 970.0,
        "humidity": 50.0,
        "wind_speed": 20.0,
        "wind_direction": 180.0
    }


def create_live_reading(
    index
):

    if not df.empty:

        row = df.iloc[
            index % len(df)
        ]

        return {
            "timestamp": pd.to_datetime(
                row["timestamp"]
            ).isoformat(),
            "temperature": float(
                row["temperature"]
            ),
            "pressure": float(
                row["pressure"]
            ),
            "humidity": float(
                row["humidity"]
            ),
            "wind_speed": float(
                row["wind_speed"]
            ),
            "wind_direction": float(
                row["wind_direction"]
            )
        }

    return {
        "timestamp": datetime.now().isoformat(),
        "temperature": -15.0,
        "pressure": 970.0,
        "humidity": 50.0,
        "wind_speed": 20.0,
        "wind_direction": 180.0
    }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🛡️ SkyGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-subtitle">Automatic Weather Station Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Station</div>',
        unsafe_allow_html=True
    )

    station = st.selectbox(
        "Station",
        [
            "AWS-DEMO-001",
            "AWS-NAS-021",
            "AWS-KOL-031"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-section">System Status</div>',
        unsafe_allow_html=True
    )

    backend_online = check_backend()

    st.session_state.last_api_status = backend_online

    if backend_online:

        st.success("🟢 Backend Online")

    else:

        st.error("🔴 Backend Offline")

    st.markdown(
        '<div class="sidebar-section">Detection Modules</div>',
        unsafe_allow_html=True
    )

    modules = [
        "Hybrid ML + Rule-Based",
        "Isolation Forest",
        "Temporal Validation",
        "Multivariate Consistency",
        "Seasonal Analysis",
        "Spatial Consistency",
        "SHAP Explainability"
    ]

    for module in modules:

        st.markdown(
            f'<div class="module-item">• {module}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="sidebar-section">Live Controls</div>',
        unsafe_allow_html=True
    )

    live_monitoring = st.toggle(
        "Enable live monitoring",
        value=False
    )

    refresh_seconds = st.selectbox(
        "Refresh interval",
        [2, 5, 10],
        index=0,
        format_func=lambda x: f"{x} seconds"
    )


# ============================================================
# MAIN HEADER
# ============================================================

header_left, header_right = st.columns(
    [4, 1]
)

with header_left:

    st.markdown(
        '<div class="main-title">SkyGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Intelligent Real-Time Anomaly Detection '
        'for Automatic Weather Stations'
        '</div>',
        unsafe_allow_html=True
    )


with header_right:

    if backend_online:

        st.markdown(
            '<div style="text-align:right;">'
            '<span class="online-pill">'
            '● LIVE · API CONNECTED'
            '</span>'
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div style="text-align:right;">'
            '<span class="offline-pill">'
            '● BACKEND OFFLINE'
            '</span>'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# STATION TABS
# ============================================================

st.tabs(
    [
        "🟢 AWS-DEMO-001",
        "🟠 AWS-NAS-021",
        "🟢 AWS-KOL-031"
    ]
)


# ============================================================
# REAL-TIME CONTROL PANEL
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '⚡ Real-Time Detection Controls'
    '</div>',
    unsafe_allow_html=True
)

control_left, control_right = st.columns(
    [4, 1]
)

with control_left:

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">Detection Control</div>'
        '<div class="panel-subtitle">'
        'Send a controlled sensor observation through '
        'the FastAPI detection pipeline.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    manual_anomaly = st.selectbox(
        "Manual anomaly scenario",
        [
            "Temperature Spike",
            "Temperature Drop",
            "Invalid Humidity",
            "Multivariate Anomaly",
            "Frozen Sensor"
        ],
        label_visibility="collapsed"
    )


with control_right:

    st.write("")

    manual_button = st.button(
        "🚨 Inject Anomaly",
        use_container_width=True
    )


# ============================================================
# MANUAL ANOMALY INJECTION
# ============================================================

if manual_button:

    if not backend_online:

        st.error(
            "FastAPI backend is offline. "
            "Start it with: "
            "python -m uvicorn src.api:app "
            "--host 127.0.0.1 --port 8000"
        )

    else:

        try:

            if manual_anomaly == "Frozen Sensor":

                frozen_result = None

                base_time = datetime.now()

                for i in range(8):

                    frozen_row = {
                        "timestamp": (
                            base_time
                            + timedelta(seconds=i)
                        ).isoformat(),
                        "temperature": 20.0,
                        "pressure": 970.0,
                        "humidity": 50.0,
                        "wind_speed": 20.0,
                        "wind_direction": 180.0
                    }

                    frozen_result = send_reading(
                        frozen_row,
                        station_id="AWS-MANUAL-FROZEN"
                    )

                    add_history(
                        frozen_row,
                        frozen_result
                    )

                    st.session_state.current_reading = (
                        frozen_row
                    )

                    st.session_state.current_result = (
                        frozen_result
                    )

                st.success(
                    "✅ Frozen sensor scenario completed "
                    "with 8 consecutive observations."
                )

            else:

                reading = create_manual_reading(
                    manual_anomaly
                )

                result = send_reading(
                    reading,
                    station_id="AWS-MANUAL-DEMO"
                )

                st.session_state.current_reading = reading

                st.session_state.current_result = result

                add_history(
                    reading,
                    result
                )

                st.success(
                    f"✅ {manual_anomaly} injected successfully."
                )

        except requests.RequestException as error:

            st.error(
                f"API request failed: {error}"
            )

        except Exception as error:

            st.error(
                f"Anomaly injection failed: {error}"
            )


# ============================================================
# LIVE MONITORING
# ============================================================

if live_monitoring and not manual_button:

    if backend_online:

        try:

            live_row = create_live_reading(
                st.session_state.current_index
            )

            live_result = send_reading(
                live_row,
                station_id="AWS-LIVE-DEMO"
            )

            st.session_state.current_reading = live_row

            st.session_state.current_result = live_result

            add_history(
                live_row,
                live_result
            )

            st.session_state.current_index += 1

        except requests.RequestException as error:

            st.error(
                f"Live API request failed: {error}"
            )

        except Exception as error:

            st.error(
                f"Live monitoring error: {error}"
            )

    else:

        st.warning(
            "Live monitoring requires the FastAPI backend."
        )


# ============================================================
# CURRENT TELEMETRY
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '📡 Current Sensor Telemetry'
    '</div>',
    unsafe_allow_html=True
)

reading = st.session_state.current_reading

if reading is not None:

    metric_1, metric_2, metric_3, metric_4, metric_5 = (
        st.columns(5)
    )

    with metric_1:

        st.metric(
            "🌡️ Temperature",
            f'{float(reading["temperature"]):.1f} °C'
        )

    with metric_2:

        st.metric(
            "◉ Pressure",
            f'{float(reading["pressure"]):.1f} hPa'
        )

    with metric_3:

        st.metric(
            "💧 Humidity",
            f'{float(reading["humidity"]):.1f} %'
        )

    with metric_4:

        st.metric(
            "💨 Wind Speed",
            f'{float(reading["wind_speed"]):.1f} m/s'
        )

    with metric_5:

        st.metric(
            "🧭 Wind Direction",
            f'{float(reading["wind_direction"]):.0f}°'
        )

else:

    st.info(
        "Enable live monitoring or inject an anomaly "
        "to display sensor telemetry."
    )


# ============================================================
# DETECTION RESULT
# ============================================================

result = st.session_state.current_result

if result is not None:

    st.markdown(
        '<div class="section-heading">'
        '🚨 AI Detection Result'
        '</div>',
        unsafe_allow_html=True
    )

    status = result.get(
        "status",
        "NORMAL"
    )

    anomaly_type = result.get(
        "anomaly_type",
        "Normal"
    )

    severity = result.get(
        "severity",
        "NONE"
    )

    confidence = result.get(
        "confidence_score",
        0
    )

    evidence_score = result.get(
        "evidence_score",
        0
    )

    root_cause = result.get(
        "root_cause",
        "None"
    )

    if status == "ANOMALY":

        st.markdown(
            f"""
            <div class="alert-box">
                <div class="alert-title">
                    🔴 ANOMALY DETECTED
                </div>
                <div class="alert-text">
                    <b>Type:</b> {anomaly_type}<br>
                    <b>Severity:</b> {severity}<br>
                    <b>Confidence:</b> {confidence}%<br>
                    <b>Evidence Score:</b> {evidence_score}<br>
                    <b>Root Cause:</b> {root_cause}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif status == "SUSPICIOUS":

        st.markdown(
            f"""
            <div class="status-suspicious">
                🟡 SUSPICIOUS READING<br>
                <span style="font-size:12px;font-weight:400;">
                    {anomaly_type}
                    · Severity {severity}
                    · Confidence {confidence}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            '<div class="status-normal">'
            '🟢 NORMAL — No significant sensor anomaly detected'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# SENSOR HEALTH
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '❤️ Sensor Health'
    '</div>',
    unsafe_allow_html=True
)

health_1, health_2, health_3, health_4 = st.columns(4)

with health_1:

    st.metric(
        "Health Score",
        "99.69"
    )

with health_2:

    st.metric(
        "Physical Anomalies",
        "1"
    )

with health_3:

    st.metric(
        "Temporal Anomalies",
        "114"
    )

with health_4:

    st.metric(
        "Frozen Observations",
        "10"
    )


# ============================================================
# SESSION NETWORK OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '🛰️ Session Network Overview'
    '</div>',
    unsafe_allow_html=True
)

network_1, network_2, network_3, network_4 = (
    st.columns(4)
)

with network_1:

    st.metric(
        "Stations Online",
        "1"
    )

with network_2:

    st.metric(
        "Readings Processed",
        str(
            st.session_state.readings_processed
        )
    )

with network_3:

    st.metric(
        "Alerts Raised",
        str(
            st.session_state.alerts
        )
    )

with network_4:

    st.metric(
        "Inference Service",
        "API"
    )


# ============================================================
# PIPELINE
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '⚙️ Detection Pipeline'
    '</div>',
    unsafe_allow_html=True
)

pipeline = [
    "AWS Data",
    "Validation",
    "Features",
    "Isolation Forest",
    "Rule Engine",
    "Classification",
    "Severity",
    "Confidence",
    "SHAP",
    "Alert"
]

pipeline_html = ""

for index, step in enumerate(pipeline):

    pipeline_html += (
        f'<span class="pipeline-step">'
        f'{step}'
        f'</span>'
    )

    if index < len(pipeline) - 1:

        pipeline_html += (
            '<span class="pipeline-arrow">→</span>'
        )

st.markdown(
    f'<div class="pipeline">{pipeline_html}</div>',
    unsafe_allow_html=True
)


# ============================================================
# TWO COLUMN AI ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '🧠 AI Analysis'
    '</div>',
    unsafe_allow_html=True
)

analysis_left, analysis_right = st.columns(2)


with analysis_left:

    reasons = []

    if result is not None:

        reasons = result.get(
            "reasons",
            []
        )

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">'
        '🔎 Detection Evidence'
        '</div>',
        unsafe_allow_html=True
    )

    if reasons:

        for reason in reasons:

            st.markdown(
                f"• {reason}"
            )

    else:

        st.write(
            "No anomaly evidence detected."
        )

    if result is not None:

        ml_detection = result.get(
            "ml_detection",
            False
        )

        ml_score = result.get(
            "ml_score",
            "N/A"
        )

        st.write(
            f"**Evidence Score:** {evidence_score}"
        )

        st.write(
            f"**Isolation Forest:** {ml_detection}"
        )

        st.write(
            f"**ML Score:** {ml_score}"
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with analysis_right:

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">'
        '🧩 SHAP Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    if result is not None:

        top_feature = result.get(
            "shap_top_feature",
            "Not available"
        )

        shap_summary = result.get(
            "shap_summary",
            "SHAP explanation is available for "
            "the current ML analysis."
        )

        st.write(
            f"**Top Feature:** {top_feature}"
        )

        st.write(
            shap_summary
        )

    else:

        st.write(
            "Run a detection to view SHAP explanation."
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# LIVE SENSOR HISTORY
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '📈 Session Sensor History'
    '</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    history_df = history_df.sort_values(
        "timestamp"
    )

    chart_data = history_df[
        [
            "timestamp",
            "temperature",
            "pressure",
            "humidity"
        ]
    ].copy()

    chart_data = chart_data.set_index(
        "timestamp"
    )

    st.line_chart(
        chart_data[["temperature"]],
        height=230
    )

    st.caption(
        "Temperature history"
    )

    st.line_chart(
        chart_data[["pressure"]],
        height=230
    )

    st.caption(
        "Atmospheric pressure history"
    )

    st.line_chart(
        chart_data[["humidity"]],
        height=230
    )

    st.caption(
        "Relative humidity history"
    )

    st.markdown(
        "**Latest readings**"
    )

    display_df = history_df.tail(
        10
    ).copy()

    display_df["timestamp"] = (
        display_df["timestamp"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No session history yet. "
        "Enable live monitoring or inject an anomaly."
    )


# ============================================================
# STATION CONFIGURATION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '🖥️ Station Configuration & Overview'
    '</div>',
    unsafe_allow_html=True
)

config_left, config_right = st.columns(
    [1.4, 1]
)


with config_left:

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">'
        '📡 Station Configuration'
        '</div>',
        unsafe_allow_html=True
    )

    configuration = {
        "Station ID": station,
        "Monitoring Mode": (
            "LIVE"
            if live_monitoring
            else "STANDBY"
        ),
        "Parameters": (
            "Temperature / Pressure / "
            "Relative Humidity / Wind"
        ),
        "Detection": "Hybrid ML + Rule Engine",
        "Explainability": "SHAP",
        "API": "FastAPI /detect"
    }

    for label, value in configuration.items():

        row_html = (
            '<div class="info-row">'
            f'<span class="info-label">{label}</span>'
            f'<span class="info-value">{value}</span>'
            '</div>'
        )

        st.markdown(
            row_html,
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with config_right:

    st.markdown(
        '<div class="panel">'
        '<div class="panel-title">'
        '🛰️ Network Status'
        '</div>',
        unsafe_allow_html=True
    )

    network_rows = [
        ("AWS-DEMO-001", "ONLINE"),
        ("AWS-NAS-021", "STANDBY"),
        ("AWS-KOL-031", "ONLINE")
    ]

    for network_name, network_status in network_rows:

        if network_status == "ONLINE":

            status_class = "network-online"

        else:

            status_class = "network-warning"

        network_html = (
            '<div class="network-row">'
            f'<span class="network-name">'
            f'{network_name}'
            f'</span>'
            f'<span class="{status_class}">'
            f'{network_status}'
            f'</span>'
            '</div>'
        )

        st.markdown(
            network_html,
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '📊 Dataset Information'
    '</div>',
    unsafe_allow_html=True
)

data_left, data_right = st.columns(2)

with data_left:

    if not df.empty:

        st.metric(
            "Observations",
            f"{len(df):,}"
        )

    else:

        st.metric(
            "Observations",
            "Unavailable"
        )


with data_right:

    st.metric(
        "Sensor Parameters",
        "5"
    )


# ============================================================
# DEGRADATION / MAINTENANCE
# ============================================================

st.markdown(
    '<div class="section-heading">'
    '🔧 Sensor Degradation & Maintenance'
    '</div>',
    unsafe_allow_html=True
)

# Fetch the latest maintenance analysis
# from the FastAPI backend.

maintenance_data = get_maintenance_monitoring()

maintenance_left, maintenance_right = st.columns(2)


if maintenance_data.get("available", False):

    risk_score = maintenance_data.get(
        "risk_score",
        0
    )

    risk_level = maintenance_data.get(
        "risk_level",
        "Unknown"
    )

    degradation_trend = maintenance_data.get(
        "degradation_trend",
        "Unknown"
    )

    historical_rate = maintenance_data.get(
        "historical_anomaly_rate",
        0
    )

    recent_rate = maintenance_data.get(
        "recent_anomaly_rate",
        0
    )

    change_rate = maintenance_data.get(
        "change_in_anomaly_rate",
        0
    )

    relative_change = maintenance_data.get(
        "relative_change",
        0
    )

    recommendation = maintenance_data.get(
        "recommendation",
        "No maintenance recommendation available."
    )

    risk_factors = maintenance_data.get(
        "risk_factors",
        []
    )


    # --------------------------------------------------------
    # SENSOR CONDITION
    # --------------------------------------------------------

    with maintenance_left:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">'
            'Sensor Condition'
            '</div>'
            '<div class="panel-subtitle">'
            'Dynamic degradation and maintenance-risk analysis'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Degradation Trend:** "
            f"{degradation_trend}"
        )

        st.write(
            f"**Maintenance Risk:** "
            f"{risk_level}"
        )

        st.write(
            f"**Maintenance Risk Score:** "
            f"{risk_score}/100"
        )

        st.write(
            f"**Historical Anomaly Rate:** "
            f"{historical_rate:.3f}%"
        )

        st.write(
            f"**Recent Anomaly Rate:** "
            f"{recent_rate:.3f}%"
        )

        st.write(
            f"**Change in Anomaly Rate:** "
            f"{change_rate:.3f} percentage points"
        )

        st.write(
            f"**Relative Change:** "
            f"{relative_change:.1f}%"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # MAINTENANCE RECOMMENDATION
    # --------------------------------------------------------

    with maintenance_right:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">'
            'Maintenance Recommendation'
            '</div>',
            unsafe_allow_html=True
        )

        if risk_level == "High":

            st.error(
                recommendation
            )

        elif risk_level == "Medium":

            st.warning(
                recommendation
            )

        elif risk_level == "Low":

            st.success(
                recommendation
            )

        else:

            st.info(
                recommendation
            )


        if risk_factors:

            st.markdown(
                "**Risk Factors**"
            )

            for factor in risk_factors:

                st.markdown(
                    f"• {factor}"
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


else:

    with maintenance_left:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">'
            'Sensor Condition'
            '</div>'
            '<div class="panel-subtitle">'
            'Maintenance monitoring'
            '</div>',
            unsafe_allow_html=True
        )

        st.warning(
            "Maintenance monitoring data is "
            "currently unavailable."
        )

        st.write(
            "Make sure the FastAPI backend is running."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


    with maintenance_right:

        st.markdown(
            '<div class="panel">'
            '<div class="panel-title">'
            'Backend Status'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The dashboard could not retrieve "
            "maintenance monitoring information "
            "from the FastAPI backend."
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    '🛡️ SkyGuard AI<br>'
    'Intelligent Real-Time Anomaly Detection '
    'for Automatic Weather Stations<br><br>'
    'FastAPI · Isolation Forest · Hybrid Rule Engine · SHAP'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

if live_monitoring:

    time.sleep(
        refresh_seconds
    )

    st.rerun()