import pandas as pd
from sklearn.linear_model import LinearRegression

def generate_forecast(df, target_year, decline_adjustment_pct=0.0):
    """Project regional production for target year with optional decline stress."""
    forecast_results = []
    regions = df["Region"].unique()

    for region in regions:
        region_data = df[df["Region"] == region].copy()
        if len(region_data) < 2:
            continue

        X = region_data["Year"].values.reshape(-1, 1)
        y = region_data["Production_Volume"].values
        model = LinearRegression()
        model.fit(X, y)

        prediction = model.predict([[target_year]])[0]
        last_year = int(region_data["Year"].max())
        years_ahead = max(0, target_year - last_year)
        decline_factor = (1 - (decline_adjustment_pct / 100)) ** years_ahead
        stressed_projection = prediction * decline_factor

        forecast_results.append({
            "Region": region,
            "Selected_Year": target_year,
            "Projected_Production": max(0, prediction),
            "Stressed_Production": max(0, stressed_projection),
            "Years_Ahead": years_ahead,
        })

    return pd.DataFrame(forecast_results)