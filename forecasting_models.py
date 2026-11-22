
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def prepare_hourly_data(raw):

    data = raw.copy()

    data.columns = [str(c).strip() for c in data.columns]

    data = data.loc[
        :, ~pd.Index(data.columns).str.match(r"^Unnamed")
    ]

    required = ["year", "month", "day", "hour", "PM2.5"]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data["timestamp"] = pd.to_datetime(
        data[["year", "month", "day", "hour"]]
    )

    data["PM2.5"] = pd.to_numeric(
        data["PM2.5"], errors="coerce"
    )

    data = data.dropna(subset=["timestamp", "PM2.5"])

    data = data.sort_values("timestamp")

    data = data.drop_duplicates("timestamp")

    data = data.set_index("timestamp").asfreq("h")

    if data["PM2.5"].isna().any():
        raise ValueError(
            "Missing hourly PM2.5 observations found."
        )

    return data


def make_sarima_forecast(series, steps=24, max_train=1000):

    y = pd.to_numeric(series, errors="coerce").dropna()

    y = y.astype(float).tail(max_train)

    if len(y) < 200:
        raise ValueError(
            "At least 200 observations are required."
        )

    model = SARIMAX(
        y,
        order=(2, 0, 2),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False, maxiter=50)

    prediction = fitted.get_forecast(
        steps=steps
    ).predicted_mean

    prediction.index = pd.date_range(
        start=y.index[-1] + pd.Timedelta(hours=1),
        periods=steps,
        freq="h"
    )

    prediction.name = "Forecast_PM2.5"

    return prediction, fitted
