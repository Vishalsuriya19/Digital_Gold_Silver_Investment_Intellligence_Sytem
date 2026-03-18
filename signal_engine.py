import json

def generate_signal(current_price, forecast_df):
    avg_future_price = forecast_df["predicted_price"].mean()

    if avg_future_price > current_price * 1.02:
        signal = "BUY"
    elif avg_future_price < current_price * 0.98:
        signal = "SELL"
    else:
        signal = "HOLD"

    output = {
        "current_price": current_price,
        "avg_30_day_prediction": avg_future_price,
        "signal": signal
    }

    with open("outputs/signals.json", "w") as f:
        json.dump(output, f, indent=4)

    return output
