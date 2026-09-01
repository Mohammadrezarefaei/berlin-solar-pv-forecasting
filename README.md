# ☀️ Berlin Solar PV Power Forecasting (Advanced Spatial & Physics-Informed Machine Learning)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://berlin-solar-pv-forecasting-ajffk5ttfnjklapdr4asak.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, production-ready machine learning pipeline designed to forecast photovoltaic (PV) power generation across spatial grids in Berlin. This project transitions from naive temporal baselines to an advanced, regularized **XGBoost** architecture augmented with meteorological physics and directional wind-driven spatial lags.

👉 **Explore the Live Dashboard:** [Streamlit App](https://berlin-solar-pv-forecasting-ajffk5ttfnjklapdr4asak.streamlit.app/)

---

## 📊 Model Performance Comparison

| Model Architecture | Features Utilized | MAE (MW) | RMSE (MW) | Status / Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Single-Point)** | Naive Persistence / Rolling Mean | 0.99 | 1.41 | High lag, prone to phase shifts |
| **Old Spatial (RF)** | Standard Random Forest (100 trees) | 0.54 | 0.72 | Moderate spatial awareness |
| **Advanced XGBoost (New)** | **3x3 Grid Spatial Lags + Solar Zenith Proxy** | **0.24** | **0.27** | **75% Error Reduction (Production Ready)** |

### Visual Performance Evaluation

![Model Performance Comparison](https://raw.githubusercontent.com/Mohammadrezarefaei/berlin-solar-pv-forecasting/main/berlin_models_comparison_final.png)

*Comparison of actual photovoltaic generation against baseline, random forest, and regularized XGBoost models.*

![Animated Evolution](https://raw.githubusercontent.com/Mohammadrezarefaei/berlin-solar-pv-forecasting/main/berlin_models_comparison_animated.gif)

*Dynamic 48-hour timeline tracking step-by-step model tracking and error minimization.*

---

## 🛠️ Core Methodology & Engineering

* **Zero Data Leakage:** Strict chronological train-test splitting (80/20) to mirror real-world deployment conditions and avoid look-ahead bias.
* **Physics-Informed Features:** Integration of a **Solar Zenith Angle Proxy** (`solar_zenith_proxy`) derived from trigonometric sun-path calculations to bound generation limits.
* **Wind-Driven Spatial Lag (`wind_spatial_lag`):** Capturing West-to-East meteorological propagation across Berlin grids to predict cloud cover and radiation shifts before they hit local sensors.
* **Regularized Hyperparameters:** Tuned via `reg_alpha=0.1` and `reg_lambda=1.0` to control variance and prevent overfitting on high-dimensional weather grids.

---

## 🔬 Model Explainability (SHAP)

To lift the "black-box" nature of gradient boosting, **SHAP (SHapley Additive exPlanations)** was implemented to validate that the model relies on true physical principles:
* **Global Importance:** `surface_solar_radiation` and `wind_spatial_lag` dominate the top tiers of feature contribution.
* **Directional Validation:** Beeswarm plots confirm that higher incoming solar radiation and positive spatial lag values scale up power generation output monotonically.

### SHAP Analysis Visualizations

![SHAP Feature Importance](https://raw.githubusercontent.com/Mohammadrezarefaei/berlin-solar-pv-forecasting/main/shap_feature_importance_bar.png)
*Global feature importance ranking via mean absolute SHAP values.*

![SHAP Beeswarm Plot](https://raw.githubusercontent.com/Mohammadrezarefaei/berlin-solar-pv-forecasting/main/shap_beeswarm.png)
*Directional feature impact and distribution across model predictions.*

---

## 📁 Repository Structure

```text
berlin-solar-pv-forecasting/
│
├── .streamlit/
│   └── config.toml
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── preprocessing.py
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py
│
├── app.py
├── requirements.txt
├── README.md
├── berlin_models_comparison_final.png
├── berlin_models_comparison_animated.gif
├── shap_feature_importance_bar.png
└── shap_beeswarm.png
