import pandas as pd
from src.preprocessing import calculate_solar_zenith_proxy, add_wind_spatial_lags

def test_preprocessing_pipeline():
    # Create dummy dataframe
    data = {'timestamp': ['2026-06-01 12:00:00', '2026-06-01 18:00:00'],
            'surface_solar_radiation': [0.8, 0.2]}
    df = pd.DataFrame(data)
    
    # Apply transformations
    df = calculate_solar_zenith_proxy(df)
    df = add_wind_spatial_lags(df)
    
    # Assertions
    assert 'solar_zenith_proxy' in df.columns
    assert 'wind_spatial_lag' in df.columns
    assert df['solar_zenith_proxy'].iloc[0] > df['solar_zenith_proxy'].iloc[1]
    print("All preprocessing tests passed successfully!")

if __name__ == "__main__":
    test_preprocessing_pipeline()
