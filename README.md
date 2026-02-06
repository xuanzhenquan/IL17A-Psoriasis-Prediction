# IL17A-Psoriasis-Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

## 🏥 Overview
This repository contains the source code and web application for the study: **"An Explainable Machine Learning Model for Predicting Therapeutic Response to IL-17A Inhibitors in Psoriasis"**.

We developed a machine learning model using **Random Forest** to predict whether a psoriasis patient will achieve a PASI 75 response. The model integrates clinical features (e.g., BMI, PASI score) and biological markers, providing interpretability via **SHAP (SHapley Additive exPlanations)** values.

## ✨ Key Features
* **High Accuracy**: Robust prediction of therapeutic response.
* **Explainable AI**: Visualizes feature contributions using SHAP force plots.
* **Web Interface**: A user-friendly [Streamlit App](Your_App_Link_Here) for clinical use.

## 🛠️ Usage
1. Clone the repository.
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---
*For research purposes only.*
