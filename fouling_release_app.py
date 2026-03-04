# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from sklearn.preprocessing import StandardScaler

# --- Configuration ---
st.set_page_config(page_title="Algae Removal Predictor", page_icon="🌊", layout="wide")

# --- Applicability Domain Function ---
def applicability_domain(x_test_normalized, x_train_normalized):
    X_train = x_train_normalized.values
    X_test = x_test_normalized.values
    try:
        # Leverage math
        xtx_inv = np.linalg.pinv(X_train.T @ X_train)
        leverage_test = np.diagonal(X_test @ xtx_inv @ X_test.T).ravel()
        # Warning leverage threshold h* = 3(p+1)/n
        h_star = 3 * ((x_train_normalized.shape[1] + 1) / x_train_normalized.shape[0])
        return [valor < h_star for valor in leverage_test], h_star, leverage_test[0]
    except:
        return [False], 0, 0

def get_color(confidence):
    if confidence == "HIGH": return 'background-color: cornflowerblue; color: white'
    elif confidence == "MEDIUM": return 'background-color: lightblue; color: black'
    else: return 'background-color: #ff4b4b; color: white'

# --- Asset Preparation ---
@st.cache_resource
def prepare_model_and_scaler():
    try:
        with open("Model/Decision_Tree_Trained.pkl", "rb") as f:
            model = pickle.load(f)
        
        descriptors_df = pd.read_csv("Dataset/descriptors.csv")
        descriptors_df.columns = descriptors_df.columns.str.strip().str.replace('﻿', '')
        
        train_raw = pd.read_csv("Dataset/AlgeaRemoval110kpa.csv")
        train_raw.columns = train_raw.columns.str.strip().str.replace('﻿', '')
        
        train_subset = train_raw[train_raw['Train/Test'] == 'Train']
        train_descriptors_raw = [[val] for val in train_subset['Mor14s-PW5-TIC1'].values]

        scaler = StandardScaler()
        scaler.fit(train_descriptors_raw)
        
        df_train_normalized = pd.DataFrame(scaler.transform(train_descriptors_raw), columns=['Descriptor'])
        
        # --- FIXED YARDSTICK ---
        y_train_values = train_subset['AlgaeRemoval110kpa'].values
        mean_val = np.mean(y_train_values)
        std_val = np.std(y_train_values) # Use the spread of Y values
        
        return model, scaler, descriptors_df, df_train_normalized, mean_val, std_val

    except Exception as e:
        st.error(f"❌ Error during setup: {e}")
        st.stop()

model, scaler, descriptors_df, df_train_normalized, mean_value, train_std = prepare_model_and_scaler()

# --- UI Layout ---
header_path = os.path.join(os.path.dirname(__file__), "header.png")
if os.path.exists(header_path):
    st.image(header_path, use_container_width=True)

st.title("🌊 Algae (U. linza) Removal Predictor")
st.markdown("---")

st.header("🧪 Component Selection")
col1, col2, col3 = st.columns(3)
with col1: pdms_type = st.selectbox("PDMS Type", ["DMS-V22", "DMS-V31"])
with col2: siloxane_type = st.selectbox("Siloxane Crosslinker", ["HMS-151", "HMS-301"])
with col3: oil_type = st.selectbox("Silicone Oil", ["None", "DPDM-005-047", "PMDM-010-065", "PMDM-010-044", "PM-100-012"])

st.header("⚖️ Component Amounts (grams)")
c1, c2, c3 = st.columns(3)
with c1: g_pdms = st.number_input(f"{pdms_type} (g)", min_value=0.0, value=20.0)
with c2: g_sil = st.number_input(f"{siloxane_type} (g)", min_value=0.0, value=0.91 if "151" in siloxane_type else 1.251)
with c3: g_oil = st.number_input("Oil (g)", min_value=0.0, value=0.0 if oil_type == "None" else 1.0)

# --- Prediction Action ---
if st.button("🔮 Predict U. linza Removal", type="primary", use_container_width=True):
    try:
        # 1. Calculation
        m = descriptors_df.loc[descriptors_df['NAME'] == pdms_type, 'Mor14s'].values[0]
        p = descriptors_df.loc[descriptors_df['NAME'] == siloxane_type, 'PW5'].values[0]
        t = 0.0 if oil_type == "None" else descriptors_df.loc[descriptors_df['NAME'] == oil_type, 'TIC1'].values[0]
        tot = g_pdms + g_sil + g_oil
        raw_test = ((g_pdms/tot) * m) + ((g_sil/tot) * p) + ((g_oil/tot) * t)
        
        # 2. Scaling
        df_test_normalized = pd.DataFrame(scaler.transform([[raw_test]]), columns=['Descriptor'])
        
        # 3. Predict
        y_pred = model.predict(df_test_normalized)[0]
        
        # 4. Williams Plot Logic
        # Standardized Residual = (Prediction - Mean) / Training Std Dev
        std_residual = (y_pred - mean_value) / train_std
        
        # Leverage Logic
        lev_inside, h_star, h_val = applicability_domain(df_test_normalized, df_train_normalized)
        
        # 5. Determine Confidence
        res_inside = abs(std_residual) < 3
        inside_domain = lev_inside[0]
        
        if inside_domain and res_inside:
            confidence = "HIGH"
        elif not inside_domain and res_inside:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # 6. Results UI
        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.success(f"### Prediction: **{y_pred:.2f}%**")
        r2.info(f"### Confidence: **{confidence}**")

        summary_df = pd.DataFrame({
            "Metric": ["Standardized Residual", "Leverage (h)", "Threshold (h*)", "Confidence"],
            "Value": [f"{std_residual:.4f}", f"{h_val:.4f}", f"{h_star:.4f}", confidence]
        })

        # Apply coloring only to the Confidence row
        def color_row(row):
            return [get_color(confidence) if row['Metric'] == "Confidence" else "" for _ in row]

        st.table(summary_df.style.apply(color_row, axis=1))

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")