import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import folium
import re
from streamlit_folium import st_folium

from tensorflow.keras.models import load_model
from datetime import timedelta

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="ClimateVerse AI",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# LOAD MODEL + SCALER
# =====================================================

@st.cache_resource
def load_assets():

    model = load_model(
        "temperature_model.keras"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_assets()

# =====================================================
# FEATURES
# =====================================================

FEATURES = [
    "T2M",
    "WS2M_MAX",
    "RH2M",
    "QV2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
    "WD2M_sin",
    "WD2M_cos",
    "doy_sin",
    "doy_cos"
]

# =====================================================
# NASA POWER READER
# =====================================================

def read_nasa_power_csv(uploaded_file):

    uploaded_file.seek(0)

    raw = pd.read_csv(
        uploaded_file,
        header=None
    )

    header_row = None

    for i in range(len(raw)):

        row = raw.iloc[i].astype(str)

        if "YEAR" in row.values:
            header_row = i
            break

    if header_row is None:
        st.error(
            "NASA POWER header not found."
        )
        st.stop()

    uploaded_file.seek(0)

    df = pd.read_csv(
        uploaded_file,
        skiprows=header_row
    )

    return df

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess(df):

    df = df.copy()

    required_columns = [
        "YEAR",
        "DOY",
        "T2M",
        "WS2M_MAX",
        "WD2M",
        "RH2M",
        "QV2M",
        "PRECTOTCORR",
        "ALLSKY_SFC_SW_DWN"
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:

        st.error(
            f"Missing Columns: {missing}"
        )

        st.stop()

    # Handle NASA missing values

    df = df.replace(
        [-999, -999.0],
        np.nan
    )

    df = df.interpolate()

    df = df.dropna()

    # Create Date

    df["date"] = (
        pd.to_datetime(
            df["YEAR"].astype(str),
            format="%Y"
        )
        + pd.to_timedelta(
            df["DOY"] - 1,
            unit="D"
        )
    )

    # Cyclical DOY

    df["doy_sin"] = np.sin(
        2 * np.pi * df["DOY"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["DOY"] / 365
    )

    # Wind Direction Encoding

    df["WD2M_sin"] = np.sin(
        np.radians(df["WD2M"])
    )

    df["WD2M_cos"] = np.cos(
        np.radians(df["WD2M"])
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    if len(df) < 30:

        st.error(
            "Minimum 30 days of data required."
        )

        st.stop()

    return df

# =====================================================
# FORECAST
# =====================================================

def forecast(df):

    scaled = scaler.transform(
        df[FEATURES]
    )

    window = scaled[-30:]

    X = np.expand_dims(
        window,
        axis=0
    )

    preds = model.predict(
        X,
        verbose=0
    )

    t2m_min = scaler.data_min_[0]
    t2m_max = scaler.data_max_[0]

    preds_actual = (
        preds *
        (t2m_max - t2m_min)
        + t2m_min
    )

    return preds_actual[0]

# =====================================================
# HEADER
# =====================================================

st.title("🌍 ClimateVerse AI")

st.markdown(
    """
    ### AI Powered Climate Forecasting Dashboard
    
    Upload Daily Climate Dataset
    and generate a 7-day temperature forecast.
    """
)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "📂 Upload Dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# =====================================================
# MAIN
# =====================================================

if uploaded_file is not None:

    try:

        df = read_nasa_power_csv(
            uploaded_file
        )

        df = preprocess(df)

        st.success(
            "Dataset Loaded Successfully"
        )

        # ====================================
        # DATASET INFO
        # ====================================

        st.info(
            f"""
            Records: {len(df)}
            
            Start Date: {df['date'].min().date()}
            
            End Date: {df['date'].max().date()}
            """
        )

        latest = df.iloc[-1]

        # ====================================
        # METRICS
        # ====================================

        st.subheader(
            "📊 Current Climate Conditions"
        )

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            "Temperature",
            f"{latest['T2M']:.1f} °C"
        )

        c2.metric(
            "Humidity",
            f"{latest['RH2M']:.1f}%"
        )

        c3.metric(
            "Rainfall",
            f"{latest['PRECTOTCORR']:.1f} mm"
        )

        c4.metric(
            "Wind Speed",
            f"{latest['WS2M_MAX']:.1f} m/s"
        )

        c5.metric(
            "Solar Radiation",
            f"{latest['ALLSKY_SFC_SW_DWN']:.1f}"
        )

        st.divider()

        # ====================================
        # GEO MAP + DIGITAL TWIN
        # ====================================

        st.subheader(
            "🌍 Geospatial Climate Intelligence"
        )

        # Default coordinates

        lat = 28.61
        lon = 77.23

        try:

            filename = uploaded_file.name

            match = re.search(
                r'_(\d+)d(\d+)N_(\d+)d(\d+)E',
                filename
            )

            if match:

                lat = float(
                    match.group(1) + "." + match.group(2)
                )

                lon = float(
                    match.group(3) + "." + match.group(4)
                )

        except:
            pass


        # Climate State Logic

        temp = latest["T2M"]
        humidity = latest["RH2M"]
        rain = latest["PRECTOTCORR"]
        wind = latest["WS2M_MAX"]

        if temp >= 40:

            climate_state = "🔥 Extreme Heat"

        elif temp >= 35 and humidity < 40:

            climate_state = "☀️ Hot & Dry"

        elif rain > 5:

            climate_state = "🌧 Rainy"

        elif humidity > 70:

            climate_state = "💧 Humid"

        else:

            climate_state = "🌤 Pleasant"


        col1, col2 = st.columns([2, 1])

        with col1:

            m = folium.Map(
                location=[lat, lon],
                zoom_start=8
            )

            popup_text = f"""
            <b>ClimateVerse AI</b><br>
            Temp: {temp:.1f} °C<br>
            Humidity: {humidity:.1f}%<br>
            Rainfall: {rain:.1f} mm<br>
            Wind: {wind:.1f} m/s<br>
            State: {climate_state}
            """

            folium.Marker(
                [lat, lon],
                popup=popup_text,
                tooltip="Forecast Location"
            ).add_to(m)

            st_folium(
                m,
                height=450,
                use_container_width=True
            )

        with col2:

            st.markdown(
                "### 🌆 Digital Twin"
            )

            if climate_state == "🔥 Extreme Heat":

                twin = """
                ☀️☀️☀️

                🏢🏢🏢

                🌡 42°C+

                🔥 HEATWAVE
                """

            elif climate_state == "☀️ Hot & Dry":

                twin = """
                ☀️☀️

                🏢🏢🏢

                🌵🌵

                HOT & DRY
                """

            elif climate_state == "🌧 Rainy":

                twin = """
                ☁️☁️☁️

                🌧🌧🌧

                🏢🏢🏢

                RAINY
                """

            elif climate_state == "💧 Humid":

                twin = """
                ☁️☁️

                💧💧💧

                🏢🏢🏢

                HUMID
                """

            else:

                twin = """
                🌤

                🌳🌳🌳

                🏢🏢🏢

                PLEASANT
                """

            st.markdown(
                f"""
                ### {climate_state}

                ```
                {twin}
                ```
                """
            )

            

        st.divider()

        # ====================================
        # LAST 30 DAYS
        # ====================================

        hist = df.tail(30)

        st.subheader(
            "📈 Last 30 Days Temperature"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["T2M"],
                mode="lines+markers",
                name="Temperature"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ====================================
        # FORECAST
        # ====================================

        preds = forecast(df)

        future_dates = pd.date_range(
            start=df["date"].iloc[-1]
            + timedelta(days=1),
            periods=7
        )

        forecast_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Predicted Temperature (°C)":
                np.round(preds, 2)
            }
        )

        st.subheader(
            "🔮 7 Day Temperature Forecast"
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        # Forecast Summary

        fc1, fc2, fc3 = st.columns(3)

        fc1.metric(
            "Average Forecast",
            f"{np.mean(preds):.2f} °C"
        )

        fc2.metric(
            "Maximum Forecast",
            f"{np.max(preds):.2f} °C"
        )

        fc3.metric(
            "Minimum Forecast",
            f"{np.min(preds):.2f} °C"
        )

        # ====================================
        # FORECAST GRAPH
        # ====================================

        st.subheader(
            "📊 Historical vs Forecast"
        )

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=hist["date"],
                y=hist["T2M"],
                mode="lines",
                name="Historical"
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=future_dates,
                y=preds,
                mode="lines+markers",
                name="Forecast"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ====================================
        # ALERTS
        # ====================================

        st.subheader(
            "🚨 Climate Alerts"
        )

        if np.max(preds) >= 40:

            st.error(
                "🔥 Heatwave Alert: Temperature may exceed 40°C"
            )

        elif np.max(preds) >= 35:

            st.warning(
                "🌡 High Temperature Warning"
            )

        else:

            st.success(
                "✅ No extreme temperature expected"
            )

        st.divider()

        # ====================================
        # MODEL PERFORMANCE
        # ====================================

        st.subheader(
            "📉 Model Performance"
        )

        mae_df = pd.DataFrame(
            {
                "Day": [
                    "Day1",
                    "Day2",
                    "Day3",
                    "Day4",
                    "Day5",
                    "Day6",
                    "Day7"
                ],
                "MAE": [
                    0.86,
                    1.21,
                    1.40,
                    1.51,
                    1.62,
                    1.62,
                    1.73
                ]
            }
        )

        fig3 = go.Figure()

        fig3.add_trace(
            go.Scatter(
                x=mae_df["Day"],
                y=mae_df["MAE"],
                mode="lines+markers",
                name="MAE"
            )
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        # ====================================
        # DOWNLOAD
        # ====================================

        csv = forecast_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"temperature_forecast_{future_dates[0].date()}.csv",
            mime="text/csv"
        )

        st.divider()

        st.caption(
            """
            ClimateVerse AI
            | TensorFlow LSTM Forecasting
            | Forecast Horizon: 7 Days
            """
        )

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

else:

    st.info(
        "Upload CSV from the sidebar to begin forecasting."
    )
