
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from forecasting_models import (
    prepare_hourly_data,
    make_sarima_forecast
)

st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f5f8f6;
}
[data-testid="stSidebar"] {
    background-color: #102b3f;
}
[data-testid="stSidebar"] * {
    color: white;
}
h1, h2, h3 {
    color: #153447;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🌿 Air Quality Monitor")
st.sidebar.caption("Ashok Vihar | Delhi")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Forecast",
        "Model Performance",
        "About"
    ]
)

st.title("🌿 Air Quality Forecasting Dashboard")
st.caption(
    "Hourly PM2.5 Analysis and Forecasting | Ashok Vihar, Delhi"
)

# Load dataset
uploaded_file = st.sidebar.file_uploader(
    "Upload AshokVihar_Hourly.csv",
    type=["csv"]
)

default_path = Path("data/AshokVihar_Hourly.csv")

df = None

try:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    elif default_path.exists():
        raw = pd.read_csv(default_path)
    else:
        raw = None

    if raw is not None:
        df = prepare_hourly_data(raw)

except Exception as e:
    st.error(f"Dataset error: {e}")

# Load actual M2 metrics
metrics_path = Path("outputs/M2_Model_Comparison.csv")

if metrics_path.exists():
    metrics = pd.read_csv(metrics_path)
else:
    metrics = pd.DataFrame({
        "Model": ["SARIMA", "ARIMA", "Holt-Winters"],
        "MAE": [17.857971, 81.732639, 152.390514],
        "RMSE": [23.776891, 84.974393, 179.749539],
        "MAPE (%)": [93.563515, 481.283398, 705.978005]
    })


# DASHBOARD
if page == "Dashboard":

    st.header("Air Quality Overview")

    if df is None:

        st.info(
            "Upload AshokVihar_Hourly.csv using the sidebar."
        )

    else:

        series = df["PM2.5"].dropna()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Latest PM2.5",
            f"{series.iloc[-1]:.2f}"
        )

        c2.metric(
            "Average PM2.5",
            f"{series.mean():.2f}"
        )

        c3.metric(
            "Maximum PM2.5",
            f"{series.max():.2f}"
        )

        c4.metric(
            "Observations",
            f"{len(series):,}"
        )

        fig = px.line(
            df.reset_index(),
            x="timestamp",
            y="PM2.5",
            title="Hourly PM2.5 Concentration",
            labels={
                "timestamp": "Date",
                "PM2.5": "PM2.5"
            }
        )

        fig.update_layout(template="plotly_white")

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        hourly = (
            df.groupby(df.index.hour)["PM2.5"]
            .mean()
            .reset_index()
        )

        hourly.columns = ["Hour", "Average PM2.5"]

        fig2 = px.line(
            hourly,
            x="Hour",
            y="Average PM2.5",
            markers=True,
            title="Hourly Pollution Pattern"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )


# FORECAST
elif page == "Forecast":

    st.header("PM2.5 Forecasting")

    if df is None:

        st.warning(
            "Upload the Ashok Vihar dataset first."
        )

    else:

        horizon = st.slider(
            "Forecast horizon (hours)",
            6, 72, 24, step=6
        )

        train_size = st.slider(
            "Recent training observations",
            500, 3000, 1000, step=250
        )

        if st.button(
            "Generate Forecast",
            type="primary"
        ):

            with st.spinner("Fitting SARIMA model..."):

                try:

                    forecast, fitted = make_sarima_forecast(
                        df["PM2.5"],
                        steps=horizon,
                        max_train=train_size
                    )

                    history = df["PM2.5"].tail(300)

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=history.index,
                            y=history.values,
                            name="Historical PM2.5"
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=forecast.index,
                            y=forecast.values,
                            name="SARIMA Forecast"
                        )
                    )

                    fig.update_layout(
                        title="Historical Data and Future Forecast",
                        xaxis_title="Time",
                        yaxis_title="PM2.5",
                        template="plotly_white"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                    st.dataframe(
                        forecast.to_frame(),
                        use_container_width=True
                    )

                    st.download_button(
                        "Download Forecast CSV",
                        forecast.to_csv().encode("utf-8"),
                        file_name="PM25_Forecast.csv"
                    )

                except Exception as e:

                    st.error(str(e))


# MODEL PERFORMANCE
elif page == "Model Performance":

    st.header("Model Performance Comparison")

    st.dataframe(
        metrics,
        use_container_width=True
    )

    best = metrics.sort_values("RMSE").iloc[0]

    st.success(
        f"Best model by RMSE: {best['Model']} "
        f"| RMSE: {best['RMSE']:.2f}"
    )

    fig = px.bar(
        metrics,
        x="Model",
        y="RMSE",
        color="Model",
        title="RMSE Comparison (Lower is Better)"
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.warning(
        "MAPE can be unstable when actual PM2.5 values "
        "are close to zero. Interpret it alongside MAE and RMSE."
    )


# ABOUT
else:

    st.header("About the Project")

    st.markdown("""
    **Project:** Multivariate Time-Series Analysis and
    Forecasting of PM2.5 Concentration Using Air Pollutant
    and Meteorological Variables at Ashok Vihar, Delhi.

    **Domain:** Environmental Data Science

    **Station:** Ashok Vihar, Delhi

    **Frequency:** Hourly

    **Target variable:** PM2.5

    **Milestone 2 models:**
    - ARIMA
    - SARIMA
    - Holt-Winters

    **Dataset source:**

    https://data.mendeley.com/datasets/bzhzr9b64v/1

    Forecasts are research estimates and are not official
    air-quality warnings or health recommendations.
    """)
