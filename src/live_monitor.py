"""
VitalSense Live Monitoring Dashboard
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("vitalsense.live_monitor")

# Configuration (change paths/params if needed)
DATA_PROCESSED_PATH = r"D:\brototype\BW1\VitalSense\data\processed"
MODEL_PATH = r"D:\brototype\BW1\VitalSense\model"
PROCESSED_FILE = os.path.join(DATA_PROCESSED_PATH, "health_processed_dataset.csv")
UNIFIED_FILE = os.path.join(DATA_PROCESSED_PATH, "unified_health_dataset.csv")
DIABETES_MODEL_FILE = os.path.join(MODEL_PATH, "diabetes_xgb.pkl")
HEART_MODEL_FILE = os.path.join(MODEL_PATH, "heart_xgb.pkl")
SCALER_STATS_FILE = os.path.join(DATA_PROCESSED_PATH, "scaler_stats.json")

REFRESH_SECONDS = 5

# Clinical-ish threshold tuning (used for severity labels)
DIABETES_MODERATE = 0.07
DIABETES_HIGH = 0.20
HEART_MODERATE = 0.05
HEART_HIGH = 0.15

# Feature order used by models / scaler
FEATURE_ORDER = [
    "Age",
    "Gender",
    "BloodPressure",
    "Cholesterol",
    "Glucose",
    "BMI",
    "HbA1c",
]


# Cached loaders
@st.cache_data(ttl=3600)
def load_models(
    diabetes_path: str = DIABETES_MODEL_FILE,
    heart_path: str = HEART_MODEL_FILE,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Load trained models from disk. Uses joblib.

    Returns:
        Tuple[diabetes_model, heart_model]. Any load failures are logged and
        returned as None.
    """
    diabetes_model = heart_model = None

    try:
        diabetes_model = joblib.load(diabetes_path)
        logger.info("Diabetes model loaded from %s", diabetes_path)
    except Exception as exc:
        st.error(f"Failed to load diabetes model: {exc}")
        logger.exception("Failed to load diabetes model")

    try:
        heart_model = joblib.load(heart_path)
        logger.info("Heart model loaded from %s", heart_path)
    except Exception as exc:
        st.error(f"Failed to load heart model: {exc}")
        logger.exception("Failed to load heart model")

    return diabetes_model, heart_model


@st.cache_data(ttl=3600)
def load_scaler_stats(stats_path: str = SCALER_STATS_FILE) -> Dict[str, Dict[str, float]]:
    """
    Load scaler statistics (mean/std per feature) from JSON.

    Returns:
        dict mapping feature -> {"mean": float, "std": float}
    """
    try:
        with open(stats_path, "r") as fh:
            stats = json.load(fh)
        logger.info("Loaded scaler stats from %s", stats_path)
        return stats
    except Exception as exc:
        st.error(f"Failed to load scaler stats: {exc}")
        logger.exception("Failed to load scaler stats")
        # Fallback: zeros/ones so scaling doesn't crash
        return {f: {"mean": 0.0, "std": 1.0} for f in FEATURE_ORDER}


# Helpers: scaling, heuristics, severity
def scale_features(sample: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> np.ndarray:
    """
    Scale sample dict to model input array. Age and Gender are passed through
    unchanged. Other features are z-scored using
    stats.
    """
    values = []
    for f in FEATURE_ORDER:
        v = float(sample.get(f, 0.0))
        if f in ("Age", "Gender"):
            values.append(v)
            continue
        s = stats.get(f, {"mean": 0.0, "std": 1.0})
        mean = float(s.get("mean", 0.0))
        std = float(s.get("std") or 1.0)
        values.append((v - mean) / std)
    arr = np.array(values, dtype=float).reshape(1, -1)
    logger.debug("Scaled sample shape %s", arr.shape)
    return arr


def calculate_heuristic_risk(sample: Dict[str, float]) -> Tuple[float, float]:
    """
    Compute a simple, conservative heuristic risk for diabetes and heart disease.
    This acts as a safety floor for ML predictions.
    """
    # Diabetes heuristic
    dia_score = 0.0
    gl = sample.get("Glucose", 0.0)
    hba = sample.get("HbA1c", 0.0)
    bmi = sample.get("BMI", 0.0)
    age = sample.get("Age", 0.0)

    if gl > 140:
        dia_score += 0.30
    if gl > 200:
        dia_score += 0.40
    if hba > 5.7:
        dia_score += 0.20
    if hba > 6.4:
        dia_score += 0.40
    if bmi > 25:
        dia_score += 0.10
    if bmi > 30:
        dia_score += 0.20
    if age > 45:
        dia_score += 0.10
    dia_risk = min(0.95, dia_score)

    # Heart heuristic
    heart_score = 0.0
    bp = sample.get("BloodPressure", 0.0)
    chol = sample.get("Cholesterol", 0.0)

    if bp > 130:
        heart_score += 0.20
    if bp > 140:
        heart_score += 0.30
    if chol > 200:
        heart_score += 0.20
    if chol > 240:
        heart_score += 0.30
    if age > 50:
        heart_score += 0.10
    if bmi > 30:
        heart_score += 0.10
    if sample.get("Diabetes", False):
        heart_score += 0.20
    # Cap
    heart_risk = min(0.95, heart_score)

    logger.debug("Heuristic dia: %.3f heart: %.3f", dia_risk, heart_risk)
    return dia_risk, heart_risk


def severity_label(prob: float, mod_thres: float, high_thres: float) -> str:
    """Return 'low'|'moderate'|'high' given thresholds."""
    if prob >= high_thres:
        return "high"
    if prob >= mod_thres:
        return "moderate"
    return "low"


def severity_color(label: str) -> str:
    """Map severity label to CSS-friendly color name."""
    return {"high": "red", "moderate": "yellow", "low": "green"}.get(label, "green")


# UI Rendering helpers
def render_right_alert_box(container: st.delta_generator.DeltaGenerator, alert: Optional[Dict[str, Any]]) -> None:
    """Render compact alert panel used on right sidebar of main UI."""
    if not alert:
        container.info("No critical alerts.")
        return

    color = severity_color(alert["severity"])
    bgcolor = "#ffcccc" if color == "red" else "#fff3bf" if color == "yellow" else "#ccffcc"
    container.markdown(
        f"**{alert['timestamp']} — {alert['title']} ({alert['prob']*100:.1f}%)**"
    )
    container.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{bgcolor}'>Severity: <b>{alert['severity'].upper()}</b></div>",
        unsafe_allow_html=True,
    )


def render_alert_history(container: st.delta_generator.DeltaGenerator, history: list) -> None:
    """Render the bottom alert history list (newest first)."""
    if not history:
        container.info("No alerts in history.")
        return

    # Render up to 50 entries for performance
    for entry in history[:50]:
        c1, c2 = container.columns([1, 4])
        c1.markdown(f"**{entry['timestamp']}**")
        c2.markdown(
            f"- **{entry['title']}**  —  Severity: *{entry['severity'].upper()}* — {entry['prob']*100:.1f}%"
        )


# Core logic: calculation + simulation
def calculate_and_update_risk(
    diabetes_model: Optional[Any],
    heart_model: Optional[Any],
    scaler_stats: Dict[str, Dict[str, float]],
    *,
    use_simulated_values: bool = False,
    ui_update: bool = True,
) -> Tuple[float, float, str, str]:
    """
    Compute predictions (ML + heuristic), determine severities, and update UI
    cards and right-panel latest alert. Returns (dia_prob, heart_prob,
    dia_severity, heart_severity).
    """
    # Pull current inputs from session / sidebar variables
    age = st.session_state.get("ui_age", 50)
    gender = st.session_state.get("ui_gender", 0)
    bmi = st.session_state.get("ui_bmi", 26.0)
    hba1c = st.session_state.get("ui_hba1c", 5.7)
    cholesterol_input = st.session_state.get("ui_cholesterol", 190.0)

    if use_simulated_values and st.session_state.get("glucose_buf"):
        glucose_val = float(st.session_state.glucose_buf[-1])
        bp_val = float(st.session_state.bp_buf[-1])
        chol_val = float(st.session_state.current_cholesterol)
    else:
        glucose_val = float(max(80, min(200, 100 + (hba1c - 5) * 10)))
        bp_val = float(max(90, min(180, int(120 + (bmi - 24) * 2)))) # Estimate BP based on BMI instead of Cholesterol
        chol_val = float(cholesterol_input)
        # update stored cholesterol when not simulating
        st.session_state.current_cholesterol = chol_val

    sample = {
        "Age": float(age),
        "Gender": int(gender),
        "BloodPressure": float(bp_val),
        "Cholesterol": float(chol_val),
        "Glucose": float(glucose_val),
        "BMI": float(bmi),
        "HbA1c": float(hba1c),
    }

    # Scale and predict
    X_scaled = scale_features(sample, scaler_stats)

    dia_prob = heart_prob = 0.0
    try:
        if diabetes_model is not None:
            dia_prob = float(np.clip(diabetes_model.predict_proba(X_scaled)[:, 1][0], 0.0, 1.0))
    except Exception as exc:
        st.error(f"Diabetes model error: {exc}")
        logger.exception("Diabetes prediction failed")

    try:
        if heart_model is not None:
            heart_prob = float(np.clip(heart_model.predict_proba(X_scaled)[:, 1][0], 0.0, 1.0))
    except Exception as exc:
        st.error(f"Heart model error: {exc}")
        logger.exception("Heart prediction failed")

    # Hybrid approach: use heuristic safety floor
    h_dia, h_heart = calculate_heuristic_risk(sample)
    dia_prob = max(dia_prob, h_dia)
    heart_prob = max(heart_prob, h_heart)

    # Determine severity labels
    dia_sev = severity_label(dia_prob, DIABETES_MODERATE, DIABETES_HIGH)
    heart_sev = severity_label(heart_prob, HEART_MODERATE, HEART_HIGH)

    # Update UI cards and right panel latest alert if requested
    if ui_update:
        dc = st.session_state.get("diabetes_card")
        hc = st.session_state.get("heart_card")
        if dc:
            dc.markdown(f"### Diabetes Risk\n**{dia_prob*100:.1f}%**\n\nSeverity: **{dia_sev.upper()}**")
        if hc:
            hc.markdown(f"### Heart Disease Risk\n**{heart_prob*100:.1f}%**\n\nSeverity: **{heart_sev.upper()}**")

        # Latest alert: choose most severe current state to display
        current_alert = None
        if dia_sev == "high" or heart_sev == "high":
            if dia_sev == "high":
                current_alert = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "title": "HIGH Diabetes risk",
                    "severity": dia_sev,
                    "prob": dia_prob,
                }
            else:
                current_alert = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "title": "HIGH Heart risk",
                    "severity": heart_sev,
                    "prob": heart_prob,
                }
        elif dia_sev == "moderate" or heart_sev == "moderate":
            if dia_sev == "moderate":
                current_alert = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "title": "Moderate Diabetes risk",
                    "severity": dia_sev,
                    "prob": dia_prob,
                }
            else:
                current_alert = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "title": "Moderate Heart risk",
                    "severity": heart_sev,
                    "prob": heart_prob,
                }

        if current_alert:
            st.session_state.latest_alert = current_alert

        # Render right panel
        render_right_alert_box(st.session_state.get("right_alert_box"), st.session_state.get("latest_alert"))

    logger.debug(
        "Predictions: dia=%.3f (%s), heart=%.3f (%s)",
        dia_prob,
        dia_sev,
        heart_prob,
        heart_sev,
    )

    return dia_prob, heart_prob, dia_sev, heart_sev


def step_simulation(
    diabetes_model: Optional[Any],
    heart_model: Optional[Any],
    scaler_stats: Dict[str, Dict[str, float]],
) -> None:
    """
    Advance the simulation by one step:
      - generate glucose / BP / cholesterol new values
      - compute predictions
      - append alerts to history when thresholds crossed
      - update charts
    """
    # 1) simulate glucose and BP as noisy random walks with small drift
    last_g = float(st.session_state.glucose_buf[-1])
    g_noise = float(np.random.normal(loc=0.0, scale=2.0))
    g_drift = (st.session_state.ui_bmi - 24) * 0.05
    next_g = float(np.clip(last_g + g_noise + g_drift, 40, 500))
    st.session_state.glucose_buf.append(next_g)

    last_bp = float(st.session_state.bp_buf[-1])
    bp_noise = float(np.random.normal(loc=0.0, scale=3.0))
    bp_drift = (st.session_state.ui_bmi - 24) * 0.2
    next_bp = float(np.clip(last_bp + bp_noise + bp_drift, 60, 220))
    st.session_state.bp_buf.append(next_bp)

    # 2) cholesterol small drift
    last_chol = float(st.session_state.current_cholesterol)
    chol_noise = float(np.random.normal(loc=0.0, scale=0.5))
    chol_drift = (st.session_state.ui_bmi - 26) * 0.01
    next_chol = float(np.clip(last_chol + chol_noise + chol_drift, 100, 400))
    st.session_state.current_cholesterol = next_chol

    st.session_state.timestamp_buf.append(pd.Timestamp.now().strftime("%H:%M:%S"))

    # 3) calculate risks with simulated values
    dia_prob, heart_prob, dia_sev, heart_sev = calculate_and_update_risk(
        diabetes_model, heart_model, scaler_stats, use_simulated_values=True
    )

    # 4) append to alert history based on severity (only while monitoring)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    created = False
    if dia_sev == "high":
        entry = {"timestamp": ts, "title": "HIGH Diabetes risk", "severity": dia_sev, "prob": dia_prob}
        st.session_state.alert_history.insert(0, entry)
        st.session_state.latest_alert = entry
        created = True
    if heart_sev == "high":
        entry = {"timestamp": ts, "title": "HIGH Heart risk", "severity": heart_sev, "prob": heart_prob}
        st.session_state.alert_history.insert(0, entry)
        st.session_state.latest_alert = entry
        created = True

    # record moderate if no high
    if dia_sev == "moderate" and not created:
        entry = {"timestamp": ts, "title": "Moderate Diabetes risk", "severity": dia_sev, "prob": dia_prob}
        st.session_state.alert_history.insert(0, entry)
    if heart_sev == "moderate" and not created:
        entry = {"timestamp": ts, "title": "Moderate Heart risk", "severity": heart_sev, "prob": heart_prob}
        st.session_state.alert_history.insert(0, entry)

    # keep history bounded
    st.session_state.alert_history = st.session_state.alert_history[:200]

    # update charts area
    vitals_df = pd.DataFrame(
        {
            "time": list(st.session_state.timestamp_buf),
            "Glucose": list(st.session_state.glucose_buf),
            "SystolicBP": list(st.session_state.bp_buf),
        }
    ).set_index("time")

    with st.session_state.get("vitals_chart"):
        st.subheader("Glucose (mg/dL)")
        st.line_chart(vitals_df["Glucose"])

    with st.session_state.get("bp_chart"):
        st.subheader("Systolic Blood Pressure (mmHg)")
        st.line_chart(vitals_df["SystolicBP"])


# App layout & initialization
def init_ui() -> Tuple[Any, Any]:
    """Build UI and initialize session_state placeholders. Return (vitals_box, bp_box)."""
    st.set_page_config(page_title="VitalSense — Live Monitor", layout="wide")
    st.markdown("<h1>🏥 VitalSense — Live Monitoring Dashboard</h1>", unsafe_allow_html=True)
    st.write("Professional medical-style dashboard — simulated live vitals feeding your "
             "trained models.")

    # load resources
    diabetes_model, heart_model = load_models()
    scaler_stats = load_scaler_stats()

    # Sidebar controls (bind to st.session_state keys)
    with st.sidebar:
        st.header("Patient Settings")
        st.session_state.ui_age = st.slider("Age", 10, 100, st.session_state.get("ui_age", 50))
        gender_sel = st.selectbox("Gender", options=["Male", "Female"], index=0)
        st.session_state.ui_gender = 1 if gender_sel.lower().startswith("m") else 0
        st.session_state.ui_bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=st.session_state.get("ui_bmi", 26.0), step=0.1)
        st.session_state.ui_hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=st.session_state.get("ui_hba1c", 5.7), step=0.1)
        st.session_state.ui_cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100.0, max_value=400.0, value=float(st.session_state.get("ui_cholesterol", 190.0)), step=1.0)

        st.markdown("---")
        st.write("Simulation controls")
        if "running" not in st.session_state:
            st.session_state.running = False
        if st.button("Start Monitoring"):
            st.session_state.running = True
            logger.info("Monitoring started by user")
        if st.button("Stop Monitoring"):
            st.session_state.running = False
            logger.info("Monitoring stopped by user")
        st.write(f"Refresh every **{REFRESH_SECONDS} seconds**")

        # Debug toggle
        st.session_state.show_debug = st.checkbox("Show Debug Info", value=st.session_state.get("show_debug", False))

        st.markdown("## Latest Critical Alert")
        latest_alert = st.session_state.get("latest_alert")
        if latest_alert:
            st.markdown(f"**{latest_alert['timestamp']}**")
            st.markdown(f"- {latest_alert['title']}")
            st.markdown(f"- Severity: **{latest_alert['severity'].upper()}**")
            st.markdown(f"- Probability: **{latest_alert['prob']*100:.1f}%**")
        else:
            st.info("No critical alerts yet.")

    # Main layout: charts and status
    col1, col2 = st.columns([2, 1])
    with col1:
        vitals_chart = st.empty()
        bp_chart = st.empty()
    with col2:
        st.subheader("Current Status")
        diabetes_card = st.empty()
        heart_card = st.empty()
        st.markdown("### Alerts (Right Panel)")
        right_alert_box = st.empty()

    # bottom history container
    st.markdown("---")
    st.markdown("## Alert History")
    alert_log_container = st.container()

    # Save UI widgets into session_state for other functions to update
    st.session_state.update(
        {
            "diabetes_card": diabetes_card,
            "heart_card": heart_card,
            "right_alert_box": right_alert_box,
            "vitals_chart": vitals_chart,
            "bp_chart": bp_chart,
            "alert_log_container": alert_log_container,
        }
    )

    # initialize buffers if not present
    st.session_state.setdefault("glucose_buf", deque(maxlen=60))
    st.session_state.setdefault("bp_buf", deque(maxlen=60))
    st.session_state.setdefault("timestamp_buf", deque(maxlen=60))
    st.session_state.setdefault("alert_history", [])
    st.session_state.setdefault("latest_alert", None)
    st.session_state.setdefault("current_cholesterol", float(st.session_state.get("ui_cholesterol", 190.0)))

    # initial baseline values if empty
    if not st.session_state.glucose_buf:
        baseline_glucose = float(max(80, min(200, 100 + (st.session_state.ui_hba1c - 5) * 10)))
        st.session_state.glucose_buf.append(baseline_glucose)
    if not st.session_state.bp_buf:
        baseline_bp = float(max(90, min(180, int(st.session_state.ui_cholesterol / 1.0))))
        st.session_state.bp_buf.append(baseline_bp)
    if not st.session_state.timestamp_buf:
        st.session_state.timestamp_buf.append(pd.Timestamp.now().strftime("%H:%M:%S"))

    return diabetes_model, heart_model, scaler_stats


# Main app runner
def main() -> None:
    """Run the Streamlit app."""
    diabetes_model, heart_model, scaler_stats = init_ui()

    # If monitoring is on, simulate; else calculate from current settings
    if st.session_state.get("running", False):
        st.success("Monitoring: RUNNING")
        try:
            step_simulation(diabetes_model, heart_model, scaler_stats)
            time.sleep(REFRESH_SECONDS)
            st.experimental_rerun()
        except Exception as exc:
            st.error(f"Simulation error: {exc}")
            logger.exception("Simulation failed")
    else:
        st.warning("Monitoring: STOPPED — click Start Monitoring to begin.")
        # update predictions from inputs immediately when stopped
        calculate_and_update_risk(diabetes_model, heart_model, scaler_stats, use_simulated_values=False)

    # render bottom history every run
    render_alert_history(st.session_state.get("alert_log_container"), st.session_state.get("alert_history", []))


if __name__ == "__main__":
    main()