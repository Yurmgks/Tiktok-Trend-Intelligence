import os
import sys
import datetime
import logging
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DailyMetric

logger = logging.getLogger(__name__)

class TrendPredictor:
    def __init__(self):
        pass

    def calculate_trend_score(self, views_growth: float, comments_growth: float, shares_growth: float) -> tuple:
        """
        Computes Trend Score based on the formula:
        Trend Score = (views_growth * 0.4) + (comments_growth * 0.3) + (shares_growth * 0.3)
        Output:
            score (float)
            status (str): 'Trending Up', 'Stable', or 'Trending Down'
        """
        # Growth rates are expected as decimals (e.g. 0.15 for 15% growth)
        score = (views_growth * 0.4) + (comments_growth * 0.3) + (shares_growth * 0.3)
        
        # Categorize
        if score > 0.05:
            status = "Trending Up"
        elif score < -0.05:
            status = "Trending Down"
        else:
            status = "Stable"
            
        return float(score), status

    def forecast_7_days(self, dates: list, values: list) -> tuple:
        """
        Forecasts the next 7 days using linear trend extrapolation.
        Returns:
            future_dates (list of datetime): Dates for the next 7 days.
            forecast_values (list of float): Predicted values.
        """
        if len(dates) < 3 or len(values) < 3:
            # Insufficient data, return simple progression
            last_date = dates[-1] if dates else datetime.datetime.now()
            last_val = values[-1] if values else 1000.0
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
            forecast_values = [last_val * (1.0 + 0.02 * i) for i in range(1, 8)]
            return future_dates, forecast_values
            
        try:
            # Prepare data for regression
            y = np.array(values)
            # Days index (0, 1, 2...)
            x = np.arange(len(values))
            
            # Fit a simple linear trend: y = mx + c
            m, c = np.polyfit(x, y, 1)
            
            # Forecast next 7 steps
            future_x = np.arange(len(values), len(values) + 7)
            forecast_values = (m * future_x + c).tolist()
            
            # Prevent negative forecasts
            forecast_values = [max(v, 0.0) for v in forecast_values]
            
            # Generate future dates
            last_date = dates[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
            
            return future_dates, forecast_values
        except Exception as e:
            logger.error(f"Error forecasting trend: {e}")
            last_date = dates[-1]
            last_val = values[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, 8)]
            forecast_values = [last_val * (1.0 + 0.01 * i) for i in range(1, 8)]
            return future_dates, forecast_values

if __name__ == "__main__":
    predictor = TrendPredictor()
    # Calculate score
    score, status = predictor.calculate_trend_score(0.25, 0.10, 0.40)
    print(f"Trend Score: {score:.2f} -> Status: {status}")
    
    # Forecast
    dates = [datetime.datetime.now() - datetime.timedelta(days=i) for i in range(10, 0, -1)]
    values = [100, 110, 115, 120, 132, 140, 155, 160, 172, 185]
    f_dates, f_vals = predictor.forecast_7_days(dates, values)
    print("\n7-Day Forecast:")
    for d, v in zip(f_dates, f_vals):
        print(f"  Date: {d.strftime('%Y-%m-%d')} -> Predicted: {v:.1f}")
