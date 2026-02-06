import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# =========================================================
# 1. Page Configuration & Model Loading
# =========================================================
st.set_page_config(page_title="IL-17A Prediction", layout="centered")

# Load the model
try:
    # Ensure your model file is named 'rf_model.pkl'
    model = joblib.load('rf_model.pkl') 
except FileNotFoundError:
    st.error("Error: Model file 'rf_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# =========================================================
# 2. Feature Definitions
# =========================================================
# NOTE: The order of keys must match the training data columns exactly.

feature_ranges = {
    # 1. BMI
    "BMI": {
        "type": "numerical", 
        "min": 10.0, "max": 50.0, "default": 24.0, 
        "label": "Body Mass Index (BMI)"
    },
    
    # 2. Biologics_History (0=No, 1=Yes)
    "Biologics_History": {
        "type": "categorical", 
        "options": [0, 1], "default": 0, 
        "label": "History of Biologics (0=No, 1=Yes)"
    },
    
    # 3. Baseline_PASI
    "Baseline_PASI": {
        "type": "numerical", 
        "min": 0.0, "max": 72.0, "default": 15.0, 
        "label": "Baseline PASI Score"
    },
    
    # 4. Hemoglobin (g/L)
    "Hemoglobin": {
        "type": "numerical", 
        "min": 50.0, "max": 200.0, "default": 130.0, 
        "label": "Hemoglobin (Hb, g/L)"
    },
    
    # 5. ALP (U/L)
    "ALP": {
        "type": "numerical", 
        "min": 10.0, "max": 300.0, "default": 70.0, 
        "label": "Alkaline Phosphatase (ALP, U/L)"
    },
    
    # 6. IBil (μmol/L)
    "IBil": {
        "type": "numerical", 
        "min": 0.0, "max": 50.0, "default": 10.0, 
        "label": "Indirect Bilirubin (IBil, μmol/L)"
    },
    
    # 7. SII
    "SII": {
        "type": "numerical", 
        "min": 0.0, "max": 5000.0, "default": 500.0, 
        "label": "Systemic Immune-Inflammation Index (SII)"
    }
}

# =========================================================
# 3. UI: Sidebar Input
# =========================================================
st.title("IL-17A Inhibitor Response Prediction")
st.markdown("### Clinical Prediction Model (Random Forest)")

st.sidebar.header("Patient Data Entry")

user_inputs = {}

# Generate input fields
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=properties["label"],
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
            key=feature
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=properties["label"],
            options=properties["options"],
            index=properties["options"].index(properties["default"]),
            key=feature
        )
    user_inputs[feature] = value

# Convert to DataFrame
input_df = pd.DataFrame([user_inputs])

# Display Input Confirmation
st.subheader("Input Confirmation")
st.dataframe(input_df, hide_index=True)

# =========================================================
# 4. Prediction & Visualization
# =========================================================
if st.button("Predict Response"):
    st.divider()
    st.subheader("Prediction Results")
    
    # --- Step A: Model Prediction ---
    try:
        # Pipeline handles scaling automatically
        predicted_proba = model.predict_proba(input_df)[0]
        probability_responder = predicted_proba[1] * 100  # Probability of Class 1
        
        # Result Logic
        if probability_responder > 50:
            result_text = "Responder"
            color_code = "#2ca02c" # Green
            advice = "High probability of favorable response to IL-17A treatment."
        else:
            result_text = "Non-Responder"
            color_code = "#d62728" # Red
            advice = "High risk of inadequate response. Monitor risk factors."

        # --- Step B: Text Visualization ---
        text = f"Predicted Probability: {probability_responder:.2f}%\nResult: {result_text}"
        
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, text, fontsize=18, ha='center', va='center',
                fontname='Times New Roman', fontweight='bold', color='black',
                transform=ax.transAxes)
        
        # Color the border based on result
        for spine in ax.spines.values():
            spine.set_edgecolor(color_code)
            spine.set_linewidth(3)
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
        st.pyplot(fig)
        
        st.info(f"Clinical Note: {advice}")

        # --- Step C: SHAP Visualization ---
        st.subheader("SHAP Interpretation")
        with st.spinner('Calculating feature contribution...'):
            # 1. Extract components
            rf_classifier = model.named_steps['classifier']
            scaler = model.named_steps['scaler']
            
            # 2. Scale input
            input_scaled = scaler.transform(input_df)
            
            # 3. Create explainer
            explainer = shap.TreeExplainer(rf_classifier)
            shap_values_raw = explainer.shap_values(input_scaled, check_additivity=False)
            
            # 4. Extract Class 1 SHAP values
            if isinstance(shap_values_raw, list):
                shap_values = shap_values_raw[1]
                base_value = explainer.expected_value[1]
            else:
                shap_values = shap_values_raw[:,:,1]
                base_value = explainer.expected_value[1]

            # 5. Plot
            plt.figure(figsize=(12, 4), dpi=150)
            shap.force_plot(
                base_value,
                shap_values[0],
                input_df.iloc[0],
                feature_names=input_df.columns,
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            st.pyplot(plt)
            st.caption("Note: Red bars push the prediction towards 'Responder'; Blue bars push towards 'Non-Responder'.")
            
    except Exception as e:
        st.error(f"Error: {e}")
        st.warning("Check: 1. Feature names match X_train columns exactly. 2. Feature order is correct.")