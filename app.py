import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="Berlin Solar PV Forecasting Dashboard",
    page_icon="☀️",
    layout="wide"
)

# Title and Overview
st.title("☀️ Berlin Solar PV Power Forecasting")
st.markdown("**Advanced Spatial & Physics-Informed Machine Learning Pipeline** for photovoltaic generation forecasting across Berlin grids.")

st.sidebar.header("Navigation & Controls")
section = st.sidebar.radio("Go to:", ["Model Performance", "SHAP Explainability", "Live Preview Data"])

def safe_display_image(filename, caption):
    if os.path.exists(filename):
        st.image(filename, caption=caption, use_container_width=True)
    else:
        st.error(f"⚠️ Image file `{filename}` not found in the directory. Please check file names in repository.")

if section == "Model Performance":
    st.subheader("📊 Comprehensive Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline MAE", "0.99 MW", "High Error")
    col2.metric("Old Spatial (RF) MAE", "0.54 MW", "Moderate")
    col3.metric("Advanced XGBoost (New)", "0.24 MW", "75% Reduction 🚀")
    
    st.markdown("---")
    st.subheader("Visual Evaluation Timeline")
    safe_display_image("berlin_models_comparison_final.png", "Actual vs. Predicted Solar Generation (25-Hour Evolution)")
        
    st.subheader("Animated Performance Tracking")
    safe_display_image("berlin_models_comparison_animated.gif", "Dynamic Timeline Tracking")

elif section == "SHAP Explainability":
    st.subheader("🔬 Model Explainability & Interpretability (SHAP)")
    st.markdown("Validating that the XGBoost model relies on true meteorological and physical principles rather than spurious correlations.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Global Feature Importance**")
        safe_display_image("shap_feature_importance_bar.png", "Mean Absolute SHAP Values")
            
    with col2:
        st.markdown("**Directional Feature Impact**")
        safe_display_image("shap_beeswarm.png", "SHAP Beeswarm Distribution")

elif section == "Live Preview Data":
    st.subheader("⚙️ Pipeline Architecture & Methodology")
    st.markdown("""
    - **Zero Data Leakage:** Strict chronological train-test splitting (80/20) mirroring real-world deployment.
    - **Physics-Informed Features:** Integration of a `solar_zenith_proxy` derived from trigonometric sun-path calculations.
    - **Wind-Driven Spatial Lags:** Capturing West-to-East meteorological propagation across Berlin grids (`wind_spatial_lag`).
    - **Regularized Hyperparameters:** Tuned via `reg_alpha=0.1` and `reg_lambda=1.0` to control variance.
    """)
    
    st.info("To run interactive notebook simulations or execute raw python pipelines, check out the repository files.")
