import pandas as pd
from config import FORECAST_DAYS

def predict_next_30_days(model, last_date):
    """
    model: trained SARIMAXResults object
    last_date: last available date in dataset
    """

    forecast_obj = model.get_forecast(steps=FORECAST_DAYS)
    forecast_values = forecast_obj.predicted_mean

    future_dates = pd.date_range(
        start=last_date,
        periods=FORECAST_DAYS + 1,
        freq="D"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_1g_INR": forecast_values.values
    })

    forecast_df.to_csv("Outputs/forecast.csv", index=False)

    return forecast_df
