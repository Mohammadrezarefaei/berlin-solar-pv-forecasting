import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """Trains a regularized XGBoost model with predefined hyperparameters."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return model, preds, {"MAE": mae, "RMSE": rmse}
