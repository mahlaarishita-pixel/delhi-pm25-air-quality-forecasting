
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from forecasting_models import (
    prepare_hourly_data,
    make_sarima_forecast
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Delhi PM2.5 Forecasting",
    page_icon="🌿",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "AshokVihar_Hourly.csv"
M2_PATH = BASE_DIR / "M2_Model_Comparison.csv"
M3_METRICS_PATH = BASE_DIR / "outputs" / "Model_Comparison.csv"
TEST_PATH = BASE_DIR / "outputs" / "Test_Predictions.csv"
FUTURE_PATH = BASE_DIR / "outputs" / "Future_Forecast.csv"

# --------------------------------------------------
# CUSTOM STYLE
# --------------------------------------------------

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

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🌿 Air Quality Monitor")
st.sidebar.caption("Ashok Vihar | Delhi")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Forecast",
        "Model Performance",
        "Test Predictions",
        "About"
    ]
)

st.title("🌿 Air Quality Forecasting Dashboard")
st.caption(
    "Hourly PM2.5 Analysis and Forecasting | Ashok Vihar, Delhi"
)

# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload AshokVihar_Hourly.csv",
    type=["csv"]
)

df = None

try:
    if uploaded_file is not None:
        raw = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
    else:
        raw = None

    if raw is not None:
        df = prepare_hourly_data(raw)

except Exception as e:
    st.error(f"Dataset error: {e}")


def load_csv(path):
    """Safely load a CSV file."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load {path.name}: {e}")

    return pd.DataFrame()


# --------------------------------------------------
# LOAD M2 AND M3 RESULTS
# --------------------------------------------------

m2_metrics = load_csv(M2_PATH)
m3_metrics = load_csv(M3_METRICS_PATH)
test_predictions = load_csv(TEST_PATH)
future_forecast = load_csv(FUTURE_PATH)

# Add milestone labels and combine comparison results

if not m2_metrics.empty:
    m2_metrics["Milestone"] = "M2"

if not m3_metrics.empty:
    m3_metrics["Milestone"] = "M3"

available_metrics = [
    data for data in [m2_metrics, m3_metrics]
    if not data.empty
]

if available_metrics:
    metrics = pd.concat(
        available_metrics,
        ignore_index=True
    )
else:
    metrics = pd.DataFrame()


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def find_column(data, possible_names):
    """Find a column using possible column names."""
    for name in possible_names:
        for col in data.columns:
            if str(col).strip().lower() == name.lower():
                return col
    return None


def get_forecast_columns(data):
    """Identify forecast time and PM2.5 columns."""

    time_col = find_column(
        data,
        ["timestamp", "datetime", "date", "ds", "time"]
    )

    value_col = find_column(
        data,
        ["PM2.5", "PM2_5", "forecast", "predicted", "yhat",
         "PM2.5_Forecast"]
    )

    if value_col is None:
        numeric_cols = data.select_dtypes(
            include="number"
        ).columns.tolist()

        numeric_cols = [
            col for col in numeric_cols
            if not str(col).lower().startswith("unnamed")
        ]

        if numeric_cols:
            value_col = numeric_cols[-1]

    return time_col, value_col


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

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
            title="Hourly PM2.5 Concentration"
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

    st.divider()

    st.subheader("M3 Forecast Highlights")

    if not future_forecast.empty:

        time_col, value_col = get_forecast_columns(
            future_forecast
        )

        if value_col is not None:

            forecast_values = pd.to_numeric(
                future_forecast[value_col],
                errors="coerce"
            ).dropna()

            if not forecast_values.empty:

                a, b, c = st.columns(3)

                a.metric(
                    "Forecast Average",
                    f"{forecast_values.mean():.2f}"
                )

                b.metric(
                    "Forecast Maximum",
                    f"{forecast_values.max():.2f}"
                )

                c.metric(
                    "Forecast Minimum",
                    f"{forecast_values.min():.2f}"
                )

        else:
            st.info("Forecast file has no identifiable numeric column.")

    else:
        st.info("M3 future forecast file is not available.")


# --------------------------------------------------
# FORECAST
# --------------------------------------------------

elif page == "Forecast":

    st.header("PM2.5 Future Forecast")

    st.subheader("M3: Saved 24-Hour Forecast")

    if not future_forecast.empty:

        time_col, value_col = get_forecast_columns(
            future_forecast
        )

        if value_col is not None:

            plot_data = future_forecast.copy()

            if time_col is not None:
                plot_data[time_col] = pd.to_datetime(
                    plot_data[time_col],
                    errors="coerce"
                )
                x_axis = time_col
            else:
                plot_data["Forecast Hour"] = range(
                    1, len(plot_data) + 1
                )
                x_axis = "Forecast Hour"

            plot_data[value_col] = pd.to_numeric(
                plot_data[value_col],
                errors="coerce"
            )

            fig = px.line(
                plot_data,
                x=x_axis,
                y=value_col,
                markers=True,
                title="M3 Future PM2.5 Forecast"
            )

            fig.update_layout(
                template="plotly_white",
                xaxis_title="Forecast Time",
                yaxis_title="PM2.5"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                future_forecast,
                use_container_width=True
            )

            st.download_button(
                "Download M3 Future Forecast",
                future_forecast.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="M3_Future_Forecast.csv",
                mime="text/csv"
            )

        else:
            st.warning(
                "Could not identify the forecast values."
            )

    else:
        st.warning(
            "Future_Forecast.csv was not found in outputs."
        )

    # Optional original M2 SARIMA forecasting feature

    st.divider()

    with st.expander("Generate a fresh M2 SARIMA forecast"):

        st.caption(
            "This option fits the SARIMA model again using the uploaded dataset."
        )

        if df is None:
            st.warning(
                "Upload the dataset first to generate a fresh forecast."
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
                "Generate SARIMA Forecast",
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
                            title="Fresh SARIMA Forecast",
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

                    except Exception as e:
                        st.error(str(e))


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.header("Model Performance Comparison")

    if metrics.empty:

        st.warning(
            "No model comparison files were found."
        )

    else:

        st.subheader("M2 and M3 Model Metrics")

        st.dataframe(
            metrics,
            use_container_width=True
        )

        rmse_col = find_column(
            metrics,
            ["RMSE"]
        )

        model_col = find_column(
            metrics,
            ["Model", "Model Name"]
        )

        if rmse_col and model_col:

            metrics[rmse_col] = pd.to_numeric(
                metrics[rmse_col],
                errors="coerce"
            )

            valid = metrics.dropna(
                subset=[rmse_col]
            )

            if not valid.empty:

                best = valid.loc[
                    valid[rmse_col].idxmin()
                ]

                st.success(
                    f"Best model by RMSE: {best[model_col]} "
                    f"| RMSE: {best[rmse_col]:.3f}"
                )

                fig = px.bar(
                    valid,
                    x=model_col,
                    y=rmse_col,
                    color="Milestone" if "Milestone" in valid.columns else model_col,
                    title="M2 and M3 RMSE Comparison (Lower is Better)",
                    barmode="group"
                )

                fig.update_layout(
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        st.warning(
            "MAPE can be unstable when actual PM2.5 values "
            "are close to zero. Interpret it alongside MAE and RMSE."
        )

        st.download_button(
            "Download Combined Model Comparison",
            metrics.to_csv(index=False).encode("utf-8"),
            file_name="Combined_Model_Comparison.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# TEST PREDICTIONS
# --------------------------------------------------

elif page == "Test Predictions":

    st.header("M3 Test Set Predictions")

    if test_predictions.empty:

        st.warning(
            "Test_Predictions.csv was not found in the outputs folder."
        )

    else:

        st.subheader("Actual vs Predicted Values")

        st.dataframe(
            test_predictions.head(100),
            use_container_width=True
        )

        st.caption(
            "Showing the first 100 rows in the table."
        )

        actual_col = find_column(
            test_predictions,
            ["Actual", "Actual_PM2.5", "y_true", "PM2.5_Actual"]
        )

        if actual_col is not None:

            plot_data = test_predictions.copy()

            predicted_cols = [
                col for col in plot_data.columns
                if col != actual_col
                and not str(col).lower().startswith("unnamed")
                and pd.api.types.is_numeric_dtype(plot_data[col])
            ]

            if predicted_cols:

                plot_data["Observation"] = range(
                    1, len(plot_data) + 1
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=plot_data["Observation"],
                        y=plot_data[actual_col],
                        name="Actual PM2.5"
                    )
                )

                for col in predicted_cols:

                    fig.add_trace(
                        go.Scatter(
                            x=plot_data["Observation"],
                            y=plot_data[col],
                            name=str(col)
                        )
                    )

                fig.update_layout(
                    title="Actual vs Model Predictions",
                    xaxis_title="Test Observation",
                    yaxis_title="PM2.5",
                    template="plotly_white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info(
                    "No numeric prediction columns were identified."
                )

        else:
            st.info(
                "Actual-value column could not be identified. "
                "Please check the CSV column names."
            )

        st.download_button(
            "Download Test Predictions",
            test_predictions.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="M3_Test_Predictions.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# ABOUT
# --------------------------------------------------

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

    ### Milestone 2 Models
    - ARIMA
    - SARIMA
    - Holt-Winters

    ### Milestone 3 Models
    - SARIMA
    - XGBoost
    - LSTM

    ### M3 Methodology
    - Time-series preprocessing
    - Lag-based feature engineering
    - Recursive forecasting
    - Deep learning using LSTM
    - Model evaluation and comparison
    - 24-hour future forecasting

    **Dataset source:**

    https://data.mendeley.com/datasets/bzhzr9b64v/1

    Forecasts are research estimates and are not official
    air-quality warnings or health recommendations.
    """)

    st.divider()

    st.caption(
        "Delhi PM2.5 Forecasting Project | M2 + M3"
    )
