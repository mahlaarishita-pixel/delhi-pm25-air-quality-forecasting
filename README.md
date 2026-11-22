# Air Quality Forecasting Dashboard

Streamlit application for hourly PM2.5 analysis and forecasting
at Ashok Vihar, Delhi.

## Features

- Historical PM2.5 dashboard
- Hourly pollution visualization
- SARIMA forecasting
- Model performance comparison
- Forecast CSV download

## Dataset

Delhi Multi-Site Air-Quality Data Set:

https://data.mendeley.com/datasets/bzhzr9b64v/1

## Run locally

Install dependencies:

pip install -r requirements.txt

Start the app:

streamlit run app.py

## Deploy on Streamlit Community Cloud

1. Upload the extracted project files to a GitHub repository.
2. Open https://share.streamlit.io/
3. Select New App.
4. Choose your repository and branch.
5. Set the main file path to app.py.
6. Click Deploy.

## Models

The app displays the actual Milestone 2 evaluation metrics.
The forecast page fits the SARIMA(2,0,2) x (1,0,1,24)
specification on a recent training window.

Forecasts are research estimates, not official air-quality alerts.