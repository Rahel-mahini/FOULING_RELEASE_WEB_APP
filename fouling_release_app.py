# # fouling_release_app.py

# import streamlit as st
# import pandas as pd
# import pickle
# import numpy as np
# #import plotly.graph_objects as go
# from sklearn.preprocessing import StandardScaler

# # Streamlit app layout

# # --- Load ML model ---
# with open("Model/combinatorial_descision_tree.pkl", "rb") as f:
#     model = pickle.load(f)


# # --- Load descriptors ---
# descriptors_df = pd.read_csv("Dataset/descriptors.csv")

# # --- Scale the descriptors ---

# X = descriptors_df[['PW5', 'TIC1', 'MWC08']].values

# # Initialize scaler
# scaler = StandardScaler()

# # Fit and transform the descriptors
# X_scaled = scaler.fit_transform(X)

# # Convert back to DataFrame to retain names
# descriptors_scaled = pd.DataFrame(X_scaled, index=descriptors_df.index, columns=['PW5', 'TIC1', 'MWC08'])
# descriptors_scaled['NAME'] = descriptors_df['NAME']


# # --- App title and image ---
# st.title("Fouling Release Property Predictor")
# st.image("header.png", use_container_width=True)  # Add your image path

# st.header("Select Components and Fractions")


# # ---- Component 1: PDMS ----
# pdms_type = st.selectbox(
#     "Select PDMS type",
#     ["DMS-V22", "DMS-V31"]
# )

# f1 = st.slider("PDMS fraction (grams)", 15.0, 25.0, 20.0)

# # ---- Component 2: Siloxane Copolymer ----
# siloxane_type = st.selectbox(
#     "Select Siloxane Copolymer Crosslinker type",
#     ["HMS-151", "HMS-301"]
# )
# f2 = st.slider("Siloxane copolymer Crosslinker fraction (grams)", 0.5, 1.5, 1.0)

# # ---- Component 3: Silicone Oil ----
# silicone_type = st.selectbox(
#     "Select Silicone Oil type",
#     ["PMM-0025", "PMM-1015", "PMM-1021", "PDM-0421"]
# )
# f3 = st.slider("Silicone oil fraction (grams)",0.0, 1.0, 3.0, 2.0)


# # --- Get descriptors ---
# pw5 = descriptors_df.loc[descriptors_df['NAME'] == pdms_type, 'PW5'].values[0]
# tic1 = descriptors_df.loc[descriptors_df['NAME'] == siloxane_type, 'TIC1'].values[0]
# mwc08 = descriptors_df.loc[descriptors_df['NAME'] == silicone_type, 'MWC08'].values[0]

# # --- Compute combinatorial descriptor ---
# combinatorial_descriptor = np.array([[f1*pw5 + f2*tic1 + f3*mwc08]])


# # --- Display sample info ---
# st.subheader("Selected Sample Info")

# sample_info = pd.DataFrame({
#     "Component": [pdms_type, siloxane_type , silicone_type],
#     "Fraction (g)": [f1, f2, f3]
# })

# st.dataframe(sample_info)

# # st.write("Combinatorial descriptor:")
# # st.write(combinatorial_descriptor )

# # --- Predict button ---
# if st.button("Predict Property"):
#     prediction = model.predict(combinatorial_descriptor)[0]
#     st.success(f"Predicted property value (Ulva. linza Removal): {prediction:.3f}")

#     # -----------------------------
#     # Allow CSV download
#     # -----------------------------
#     result_df = pd.DataFrame({
#         "PDMS_type":[pdms_type],
#         "PDMS_fraction":[f1],
#         "Siloxane_type":[siloxane_type],
#         "Siloxane_fraction":[f2],
#         "Silicone_type":[silicone_type],
#         "Silicone_fraction":[f3],
#         "Combinatorial_descriptor":[combinatorial_descriptor],
#         "Ulva_linza_Removal":[prediction]
#     })

#     csv = result_df.to_csv(index=False)

#     st.download_button(
#         label="Download prediction as CSV",
#         data=csv,
#         file_name="prediction.csv",
#         mime="text/csv"
#     )
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- AD Helper Functions ---
def applicability_domain(x_test_val, x_train_val):
    """Calculates leverage-based AD."""
    X_train = x_train_val
    X_test = x_test_val
    
    # Calculate Hat Matrix inverse
    xtx_inv = np.linalg.inv(X_train.T @ X_train)
    
    # Leverage for test point: h = x_test * (X_train.T * X_train)^-1 * x_test.T
    leverage_test = np.sum((X_test @ xtx_inv) * X_test, axis=1)
    
    # threshold h* = 3 * (p + 1) / n
    h3 = 3 * ((X_train.shape[1] + 1) / X_train.shape[0])
    
    return leverage_test[0], leverage_test[0] < h3

def get_color(confidence):
    colors = {"HIGH": "#90EE90", "MEDIUM": "#ADD8E6", "LOW": "#FFCCCB"}
    return colors.get(confidence, "#FFFFFF")

# --- Load ML model ---
with open("Model/combinatorial_descision_tree.pkl", "rb") as f:
    model = pickle.load(f)

# --- Load and Process training data (needed for AD) ---
descriptors_df = pd.read_csv("Dataset/descriptors.csv")
# Note: For AD, we need the actual combinatorial features used during training.
# Assuming df_train_normalized is your training descriptor matrix:
# Here we simulate the training matrix based on your selected features.
# Ideally, you should load the pre-calculated X_train_scaled used during model fitting.
X_train_raw = descriptors_df[['PW5', 'TIC1', 'MWC08']].values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
mean_val_training = 2.56 # Replace with the actual mean of your training target variable

# --- App UI ---
st.title("Fouling Release Property Predictor")
st.image("header.png", use_container_width=True)

st.header("Select Components and Fractions")

pdms_type = st.selectbox("Select PDMS type", ["DMS-V22", "DMS-V31"])
f1 = st.slider("PDMS fraction (grams)", 15.0, 25.0, 20.0)

siloxane_type = st.selectbox("Select Siloxane Copolymer Crosslinker type", ["HMS-151", "HMS-301"])
f2 = st.slider("Siloxane copolymer Crosslinker fraction (grams)", 0.5, 1.5, 1.0)

silicone_type = st.selectbox("Select Silicone Oil type", ["PMM-0025", "PMM-1015", "PMM-1021", "PDM-0421"])
f3 = st.slider("Silicone oil fraction (grams)", 0.0, 3.0, 2.0)

# --- Get descriptors ---
pw5 = descriptors_df.loc[descriptors_df['NAME'] == pdms_type, 'PW5'].values[0]
tic1 = descriptors_df.loc[descriptors_df['NAME'] == siloxane_type, 'TIC1'].values[0]
mwc08 = descriptors_df.loc[descriptors_df['NAME'] == silicone_type, 'MWC08'].values[0]

# --- Compute combinatorial descriptor ---
# Ensure this matches exactly how the model was trained
comb_raw = np.array([[f1*pw5, f2*tic1, f3*mwc08]]) 
comb_scaled = scaler.transform(comb_raw) # Scale the input

# --- Predict button ---
if st.button("Predict Property"):
    # 1. Prediction
    prediction = model.predict(comb_scaled)[0]
    
    # 2. AD Calculation (Leverage)
    leverage_val, is_inside_h = applicability_domain(comb_scaled, X_train_scaled)
    
    # 3. Residual AD Calculation
    # Using a placeholder for "true" value as mean_value to calculate standardized residual
    residual = mean_val_training - prediction
    # Simplified Standardized Residual calculation for app context:
    std_dev_train = 0.5 # Replace with the RMSE of your training set
    std_resid = residual / std_dev_train
    is_inside_std = -3 <= std_resid <= 3
    
    # 4. Determine Confidence Level
    if is_inside_h and is_inside_std:
        confidence = "HIGH"
    elif not is_inside_h and not is_inside_std:
        confidence = "LOW"
    elif not is_inside_h and is_inside_std:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Display Results
    st.success(f"Predicted property value: {prediction:.3f}")
    
    # Result Dataframe
    result_df = pd.DataFrame({
        "PDMS": [pdms_type],
        "Siloxane": [siloxane_type],
        "Silicone": [silicone_type],
        "Prediction": [round(prediction, 3)],
        "Confidence": [confidence]
    })

    # Apply styling to the dataframe
    def style_confidence(val):
        color = get_color(val)
        return f'background-color: {color}'

    st.subheader("Prediction Results")
    st.dataframe(result_df.style.applymap(style_confidence, subset=['Confidence']))

    # Download Button
    csv = result_df.to_csv(index=False)
    st.download_button("Download as CSV", csv, "prediction.csv", "text/csv")