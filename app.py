import streamlit as st
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Trend Waste Model (Research Edition)", layout="wide")

st.title("Online Aesthetic Trend Lifecycle Model (Research Version)")

trend = st.text_input("Enter a trend (e.g. coquette aesthetic)")


# -----------------------------
# SAFE PYTRENDS FETCH (FIX)
# -----------------------------
@st.cache_data(ttl=3600)
def fetch_trends(trend):
    """Cached + retry-safe data fetch"""

    pytrends = TrendReq(
        hl='en-US',
        tz=360,
        requests_args={
            "headers": {"User-Agent": "Mozilla/5.0"}
        }
    )

    for attempt in range(5):
        try:
            pytrends.build_payload([trend], timeframe="today 5-y")
            data = pytrends.interest_over_time()

            # small delay to avoid rapid-fire requests
            time.sleep(5)

            return data

        except TooManyRequestsError:
            wait = (attempt + 1) * 15
            time.sleep(wait)

    return pd.DataFrame()


# -----------------------------
# CORE METHODS
# -----------------------------
def smooth_series(values, window=6):
    return values.rolling(window=window).mean().dropna()


def normalize_peak(values):
    peak = values.max()
    if peak == 0:
        return values * 0
    return (values / peak) * 100


def detect_peak(values):
    smoothed = values.rolling(window=3).mean()
    return smoothed.idxmax()


def compute_decay(peak_value, residual_value, time_years):
    if peak_value <= 0 or residual_value <= 0 or time_years <= 0:
        return 0, 0

    decay_rate = (np.log(peak_value) - np.log(residual_value)) / time_years
    half_life = np.log(2) / decay_rate if decay_rate > 0 else 0

    return decay_rate, half_life


def classify_phase(v):
    if v < 31:
        return "Discovery"
    elif v < 70:
        return "Acceleration"
    elif v >= 100:
        return "Peak"
    elif v >= 70:
        return "Saturation"
    elif v < 40:
        return "Decline"
    else:
        return "Residual"


# -----------------------------
# MAIN PIPELINE
# -----------------------------
if trend:

    data = fetch_trends(trend)

    if data.empty:
        st.error("No data found or rate limited. Try again later.")
    else:

        raw = data[trend].copy()
        smoothed = smooth_series(raw, window=6)
        values = normalize_peak(smoothed)

        peak_index = detect_peak(values)
        peak_value = 100
        peak_date = peak_index

        post_peak = values.loc[peak_index:]
        stable_region = post_peak[post_peak < 40]

        if len(stable_region) > 8:
            residual_value = stable_region.tail(8).mean()
        else:
            residual_value = post_peak.tail(8).mean()

        time_years = (values.index[-1] - values.index[0]).days / 365

        decay_rate, half_life = compute_decay(
            peak_value,
            residual_value,
            time_years
        )

        phases = values.apply(classify_phase)
        num_significant_peaks = len(values[values >= 90])

        # -------------------------
        # VISUALIZATION
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Normalized Trend Lifecycle (Peak = 100)")

            fig, ax = plt.subplots()
            ax.plot(values.values, label="Normalized Trend")
            ax.axhline(100, linestyle="--", label="Peak (100)")
            ax.axhline(70, linestyle="--", alpha=0.3)
            ax.axhline(40, linestyle="--", alpha=0.3)

            ax.set_ylabel("GT Index (Normalized)")
            ax.legend()

            st.pyplot(fig)

        with col2:
            st.subheader("Model Outputs")

            st.write("Peak Value:", 100)
            st.write("Peak Date:", peak_date.date())
            st.write("Decay Rate:", round(decay_rate, 4))
            st.write("Half-Life (years):", round(half_life, 3))
            st.write("Residual Value:", round(residual_value, 2))
            st.write("Detected High-Intensity Peaks:", num_significant_peaks)

        # -------------------------
        # TWI
        # -------------------------
        st.subheader("Trend Waste Index (Research Model)")

        value_collapse = 1 - (residual_value / 100)
        peak_sharpness = min(1.0, num_significant_peaks / 5)
        time_proxy = min(1.0, half_life / 3) if half_life else 0

        twi = (
            value_collapse * 0.4 +
            peak_sharpness * 0.4 +
            time_proxy * 0.2
        )

        st.write("TWI Score:", round(twi, 3))

        if twi > 0.75:
            st.error("HIGH WASTE TREND")
        elif twi > 0.5:
            st.warning("MODERATE WASTE TREND")
        else:
            st.success("LOW WASTE TREND")

        # -------------------------
        # INTERPRETATION
        # -------------------------
        st.subheader("Interpretation (Model Output)")

        if half_life < 1:
            st.write("Classification: Fast-decay algorithmic microtrend")
        elif half_life < 2.5:
            st.write("Classification: Medium-cycle aesthetic trend")
        else:
            st.write("Classification: Long-tail / subcultural durability trend")

        if num_significant_peaks > 1:
            st.write("Note: Multi-peak behavior detected → possible revival cycle")