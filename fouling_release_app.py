# fouling_release_app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Streamlit app layout

# --- Load ML model ---
with open("Model/combinatorial_descision_tree.pkl", "rb") as f:
    model = pickle.load(f)


# --- Load descriptors ---
descriptors_df = pd.read_csv("Dataset/descriptors.csv")

# --- Scale the descriptors ---

X = descriptors_df[['PW5', 'TIC1', 'MWC08']].values

# Initialize scaler
scaler = StandardScaler()

# Fit and transform the descriptors
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame to retain names
descriptors_scaled = pd.DataFrame(X_scaled, index=descriptors_df.index, columns=['PW5', 'TIC1', 'MWC08'])
descriptors_scaled['NAME'] = descriptors_df['NAME']


# --- App title and image ---
st.title("Fouling Release Property Predictor")
st.image("header.png", use_container_width=True)  # Add your image path

st.header("Select Components and Fractions")


# ---- Component 1: PDMS ----
pdms_type = st.selectbox(
    "Select PDMS type",
    ["DMS-V22", "DMS-V31"]
)

f1 = st.slider("PDMS fraction (grams)", 15.0, 25.0, 20.0)

# ---- Component 2: Siloxane Copolymer ----
siloxane_type = st.selectbox(
    "Select Siloxane Copolymer Crosslinker type",
    ["HMS-151", "HMS-301"]
)
f2 = st.slider("Siloxane copolymer Crosslinker fraction (grams)", 0.5, 1.5, 1.0)

# ---- Component 3: Silicone Oil ----
silicone_type = st.selectbox(
    "Select Silicone Oil type",
    ["PMM-0025", "PMM-1015", "PMM-1021", "PDM-0421"]
)
f3 = st.slider("Silicone oil fraction (grams)",0.0, 1.0, 3.0, 2.0)


# --- Get descriptors ---
pw5 = descriptors_df.loc[descriptors_df['NAME'] == pdms_type, 'PW5'].values[0]
tic1 = descriptors_df.loc[descriptors_df['NAME'] == siloxane_type, 'TIC1'].values[0]
mwc08 = descriptors_df.loc[descriptors_df['NAME'] == silicone_type, 'MWC08'].values[0]

# --- Compute combinatorial descriptor ---
combinatorial_descriptor = np.array([[f1*pw5 + f2*tic1 + f3*mwc08]])


# --- Display sample info ---
st.subheader("Selected Sample Info")

sample_info = pd.DataFrame({
    "Component": [pdms_type, siloxane_type , silicone_type],
    "Fraction (g)": [f1, f2, f3]
})

st.dataframe(sample_info)

# st.write("Combinatorial descriptor:")
# st.write(combinatorial_descriptor )

# --- Predict button ---
if st.button("Predict Property"):
    prediction = model.predict(combinatorial_descriptor)[0]
    st.success(f"Predicted property value (Ulva. linza Removal): {prediction:.3f}")

    # -----------------------------
    # Allow CSV download
    # -----------------------------
    result_df = pd.DataFrame({
        "PDMS_type":[pdms_type],
        "PDMS_fraction":[f1],
        "Siloxane_type":[siloxane_type],
        "Siloxane_fraction":[f2],
        "Silicone_type":[silicone_type],
        "Silicone_fraction":[f3],
        "Combinatorial_descriptor":[combinatorial_descriptor],
        "Ulva_linza_Removal":[prediction]
    })

    csv = result_df.to_csv(index=False)

    st.download_button(
        label="Download prediction as CSV",
        data=csv,
        file_name="prediction.csv",
        mime="text/csv"
    )
