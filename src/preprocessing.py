import pandas as pd
import numpy as np

def calculate_solar_zenith_proxy(df, timestamp_col='timestamp'):
    """Calculates a trigonometric proxy for solar zenith angle based on hour of the day."""
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    hours = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60.0
    # Trigonometric proxy peaking at solar noon (~12:00)
    df['solar_zenith_proxy'] = np.sin((hours - 6) * np.pi / 12).clip(lower=0)
    return df

def add_wind_spatial_lags(df, target_col='surface_solar_radiation'):
    """Simulates West-to-East directional spatial propagation lag across Berlin grids."""
    df['wind_spatial_lag'] = df[target_col].shift(1).fillna(method='bfill')
    return df
