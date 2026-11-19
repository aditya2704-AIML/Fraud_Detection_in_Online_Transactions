from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PaySphere Fraud Scorer", page_icon="🛡️", layout="wide")

BUNDLE_PATH = Path("paysphere_results/model_bundle.joblib")


def to_log_odds(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


@st.cache_resource
def load_bundle(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def route_action(score: float, policy: dict) -> str:
    if policy.get("automatic_block_enabled") and score >= policy.get("block_threshold", 1.000001):
        return "Hard block (proposed)"
    if score >= policy.get("review_threshold", 0.30):
        return "Analyst review"
    if score >= policy.get("otp_threshold", 0.19):
        return "OTP challenge"
    return "Auto-approve"


st.title("PaySphere — Online Transaction Fraud Scoring Demo")
st.caption(
    "Interactive scoring interface backed by the calibrated model bundle (`model_bundle.joblib`). "
    "Intended for portfolio demonstration rather than live payment authorization."
)

bundle = load_bundle(BUNDLE_PATH)

if bundle is None:
    st.error(
        f"Could not find `{BUNDLE_PATH}`. Run `PaySphere_Fraud_Detection.ipynb` first "
        "to generate the trained model bundle and policy."
    )
    st.stop()

pipeline = bundle["pipeline"]
calibrator = bundle["calibrator"]
feature_cols = bundle["features"]
policy = bundle["policy"]

with st.sidebar:
    st.header("Model & Policy Info")
    st.write(f"**Selected Model:** {bundle.get('model', 'Histogram boosting')}")
    st.write(f"**Total Features:** {len(feature_cols)}")
    st.json(policy)

st.subheader("Transaction & Risk Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Transaction Amount (INR)", min_value=1.0, value=12500.0, step=100.0)
    avg_amount_24h = st.number_input("Avg Amount Last 24h (INR)", min_value=0.0, value=4500.0, step=100.0)
    payment_method = st.selectbox("Payment Method", ["UPI", "CARD", "NETBANKING", "WALLET"])
    merchant_category = st.selectbox(
        "Merchant Category",
        ["Electronics", "Fashion", "Gaming", "Grocery", "Travel", "Utilities"],
    )
    authentication_method = st.selectbox("Authentication Method", ["OTP", "3DS", "PIN", "NONE"])

with col2:
    ip_risk = st.slider("IP Address Risk Score", 0.0, 1.0, 0.35, 0.01)
    device_trust = st.slider("Device Trust Score", 0.0, 1.0, 0.65, 0.01)
    otp_success_rate = st.slider("Customer OTP Success Rate", 0.0, 1.0, 0.85, 0.01)
    merchant_fraud_rate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, 0.10, 0.01)
    hour_of_day = st.slider("Hour of Day (0–23)", 0, 23, 14, 1)

with col3:
    device_change = int(st.checkbox("Device Change Flag", value=False))
    location_change = int(st.checkbox("Location Change Flag", value=False))
    is_international = int(st.checkbox("International Transaction", value=False))
    is_weekend = int(st.checkbox("Weekend Flag", value=hour_of_day in {0} and False))
    prior_device_uses = st.number_input("Prior Device Uses (Customer)", min_value=0, value=2, step=1)
    prior_merchant_uses = st.number_input("Prior Merchant Uses (Customer)", min_value=0, value=1, step=1)
    prior_customer_txns = st.number_input("Prior Customer Transactions", min_value=0, value=5, step=1)
    velocity_1h = st.number_input("Sample Velocity Last 1h", min_value=0, value=1, step=1)
    minutes_since_prev = st.number_input("Minutes Since Previous Txn", min_value=0.0, value=120.0, step=5.0)

# Start with NaNs for all required features so SimpleImputer handles any unexposed columns
row_data = {col: np.nan for col in feature_cols}

# Populate raw and derived features where present in feature_cols
denom_txns = prior_customer_txns if prior_customer_txns > 0 else np.nan
derived_values = {
    "amount": amount,
    "avg_amount_last_24h": avg_amount_24h if avg_amount_24h > 0 else np.nan,
    "payment_method": payment_method,
    "merchant_category": merchant_category,
    "authentication_method": authentication_method,
    "ip_address_risk_score": ip_risk,
    "device_trust_score": device_trust,
    "otp_success_rate_customer": otp_success_rate,
    "merchant_historical_fraud_rate": merchant_fraud_rate,
    "hour_of_day": hour_of_day,
    "device_change_flag": device_change,
    "location_change_flag": location_change,
    "is_international": is_international,
    "is_weekend": is_weekend,
    "prior_device_uses": prior_device_uses,
    "prior_merchant_uses": prior_merchant_uses,
    "prior_customer_transactions": prior_customer_txns,
    "sample_velocity_1h": velocity_1h,
    "minutes_since_previous": minutes_since_prev,
    "new_device_flag": int(prior_device_uses == 0),
    "new_merchant_flag": int(prior_merchant_uses == 0),
    "device_familiarity": (prior_device_uses / denom_txns) if pd.notna(denom_txns) else np.nan,
    "amount_deviation_ratio": amount / (avg_amount_24h + 1.0),
    "amount_difference": amount - avg_amount_24h,
    "rapid_retry_flag": int(minutes_since_prev <= 5.0),
    "dual_shift_flag": int(device_change * location_change),
    "late_night_flag": int(0 <= hour_of_day <= 5),
    "log_amount": float(np.log1p(amount)),
    "combined_risk_index": (
        0.30 * ip_risk
        + 0.25 * device_change
        + 0.20 * location_change
        + 0.10 * is_international
        + 0.10 * merchant_fraud_rate
        + 0.05 * (1.0 - otp_success_rate)
    ),
}

for k, v in derived_values.items():
    if k in row_data:
        row_data[k] = v

input_df = pd.DataFrame([row_data], columns=feature_cols)

if st.button("Score Transaction", type="primary"):
    raw_prob = float(pipeline.predict_proba(input_df)[0, 1])
    cal_prob = float(calibrator.predict_proba(to_log_odds(np.array([raw_prob])))[0, 1])
    action = route_action(cal_prob, policy)
    binary_flag = cal_prob >= policy.get("binary_threshold", 0.235)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Calibrated Fraud Risk", f"{cal_prob:.2%}")
    m2.metric("Raw Model Score", f"{raw_prob:.2%}")
    m3.metric("Proposed Action", action)
    m4.metric("Binary Flag (≤10% FPR Policy)", "FLAGGED" if binary_flag else "NOT FLAGGED")
