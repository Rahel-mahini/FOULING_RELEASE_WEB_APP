# -*- coding: utf-8 -*-
"""
Algae Removal Predictor - Simplified Version
Focuses on formulation input and prediction
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

def create_model_directory():
    """Create Model directory if it doesn't exist"""
    if not os.path.exists("Model"):
        os.makedirs("Model")
        print("✓ Created Model directory")

def load_descriptors():
    """Load molecular descriptors database"""
    print("\n" + "="*70)
    print("STEP 1: Loading Descriptors Database")
    print("="*70)
    
    df_desc = pd.read_csv("Dataset/descriptors.csv")
    df_desc.columns = df_desc.columns.str.strip().str.replace('﻿', '')
    df_desc['NAME'] = df_desc['NAME'].astype(str).str.strip()
    
    print(f"✓ Loaded descriptors for {len(df_desc)} components")
    print(f"✓ Columns: {list(df_desc.columns)}")
    
    return df_desc

def load_training_data():
    """Load algae removal training data"""
    print("\n" + "="*70)
    print("STEP 2: Loading Training Data")
    print("="*70)
    
    df = pd.read_csv("Dataset/AlgeaRemoval110pka.csv")
    df.columns = df.columns.str.strip().str.replace('﻿', '')
    df['NAME'] = df['NAME'].astype(str).str.strip()
    
    # Convert to numeric
    df['AlgaeRemoval110kpa'] = pd.to_numeric(df['AlgaeRemoval110kpa'], errors='coerce')
    
    # Split train/test
    train_data = df[df['Train/Test'] == 'Train'].copy()
    test_data = df[df['Train/Test'] == 'Test'].copy()
    
    print(f"✓ Loaded {len(df)} total samples")
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Test samples: {len(test_data)}")
    
    return train_data, test_data

def parse_formulation_name(name):
    """
    Parse formulation name to extract components
    Example: 'v31-151-1021(10%)' -> PDMS: v31, Siloxane: 151, Oil: 1021, Fraction: 10%
    """
    name = name.strip()
    parts = name.split('-')
    
    # Extract PDMS type
    if parts[0].startswith('v'):
        pdms = f"DMS-V{parts[0][1:]}"  # v31 -> DMS-V31, v22 -> DMS-V22
    else:
        pdms = None
    
    # Extract Siloxane type
    if len(parts) > 1:
        siloxane = f"HMS-{parts[1]}"  # 151 -> HMS-151, 301 -> HMS-301
    else:
        siloxane = None
    
    # Extract Oil and percentage
    oil = None
    oil_fraction = 0.0
    
    if len(parts) > 2:
        oil_part = parts[2]
        # Check if it has percentage
        if '(' in oil_part:
            oil_name = oil_part.split('(')[0]
            pct_str = oil_part.split('(')[1].replace(')', '').replace('%', '')
            oil_fraction = float(pct_str) / 100.0
            
            if oil_name and oil_name != '':
                oil = f"PMM-{oil_name}"  # 1021 -> PMM-1021
        else:
            # No oil specified
            oil = None
            oil_fraction = 0.0
    
    return pdms, siloxane, oil, oil_fraction

def calculate_combinatorial_descriptor(pdms_name, siloxane_name, oil_name, oil_fraction, df_desc):
    """
    Calculate combinatorial descriptor from component names
    """
    # Get descriptors
    try:
        mwc08 = df_desc.loc[df_desc['NAME'] == pdms_name, 'MWC08'].values[0]
    except:
        print(f"  ⚠️  Warning: {pdms_name} not found in descriptors")
        mwc08 = 0.0
    
    try:
        tic1 = df_desc.loc[df_desc['NAME'] == siloxane_name, 'TIC1'].values[0]
    except:
        print(f"  ⚠️  Warning: {siloxane_name} not found in descriptors")
        tic1 = 0.0
    
    if oil_name and oil_name != "None":
        try:
            pw5 = df_desc.loc[df_desc['NAME'] == oil_name, 'PW5'].values[0]
        except:
            print(f"  ⚠️  Warning: {oil_name} not found in descriptors")
            pw5 = 0.0
    else:
        pw5 = 0.0
    
    # Calculate fractions (assuming fixed ratios)
    # PDMS is typically ~90-95%, Siloxane ~4-5%, Oil varies based on percentage
    if oil_fraction > 0:
        frac_oil = oil_fraction
        frac_sil = 0.04  # Approximate
        frac_pdms = 1.0 - frac_sil - frac_oil
    else:
        frac_oil = 0.0
        frac_sil = 0.043  # Approximate siloxane fraction when no oil
        frac_pdms = 1.0 - frac_sil
    
    # Combinatorial descriptor
    descriptor = (frac_pdms * mwc08) + (frac_sil * tic1) + (frac_oil * pw5)
    
    return descriptor, frac_pdms, frac_sil, frac_oil

def calculate_all_descriptors(data, df_desc):
    """Calculate combinatorial descriptors for all samples"""
    print("\n" + "="*70)
    print("STEP 3: Calculating Combinatorial Descriptors")
    print("="*70)
    
    descriptors = []
    
    for idx, row in data.iterrows():
        name = row['NAME']
        pdms, siloxane, oil, oil_frac = parse_formulation_name(name)
        
        desc, f_pdms, f_sil, f_oil = calculate_combinatorial_descriptor(
            pdms, siloxane, oil, oil_frac, df_desc
        )
        
        descriptors.append(desc)
        
        print(f"  {name:25s} -> MWC08_TIC1_PW5 = {desc:.6f}")
    
    print(f"\n✓ Calculated {len(descriptors)} descriptors")
    
    return np.array(descriptors)

def train_model(X_train_scaled, y_train):
    """Train decision tree model"""
    print("\n" + "="*70)
    print("STEP 5: Training Decision Tree Model")
    print("="*70)
    
    model = DecisionTreeRegressor(
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    print(f"\nModel Structure:")
    print(f"  Number of leaves: {model.get_n_leaves()}")
    print(f"  Tree depth: {model.get_depth()}")
    print(f"  Total nodes: {model.tree_.node_count}")
    
    if model.get_n_leaves() == 8:
        print(f"  ✅ Perfect! Got exactly 8 leaf nodes!")
    elif model.get_n_leaves() >= 6:
        print(f"  ✅ Good! Got {model.get_n_leaves()} leaf nodes")
    else:
        print(f"  ⚠️  Only {model.get_n_leaves()} leaf nodes")
    
    return model

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """Evaluate model performance"""
    print("\n" + "="*70)
    print("STEP 6: Model Evaluation")
    print("="*70)
    
    # Training predictions
    y_train_pred = model.predict(X_train_scaled)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    print(f"\nTraining Performance:")
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.4f}%")
    print(f"  MAE: {train_mae:.4f}%")
    
    # Test predictions
    if len(y_test) > 0:
        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"\nTest Performance:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  RMSE: {test_rmse:.4f}%")
        print(f"  MAE: {test_mae:.4f}%")
    
    return y_train_pred

def analyze_predictions(model, X_train_scaled, y_train, train_data):
    """Analyze predictions and show details"""
    print("\n" + "="*70)
    print("STEP 7: Prediction Analysis")
    print("="*70)
    
    y_train_pred = model.predict(X_train_scaled)
    unique_preds = sorted(list(set(y_train_pred)))
    
    print(f"\nUnique Prediction Values: {len(unique_preds)}")
    print("\nAll Leaf Predictions:")
    for i, pred in enumerate(unique_preds, 1):
        count = sum(1 for p in y_train_pred if abs(p - pred) < 0.01)
        print(f"  {i}. {pred:.2f}% ({count} samples)")
    
    print(f"\nDetailed Training Predictions:")
    print(f"{'Sample':<5} {'Name':<25} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 70)
    
    for i, (idx, row) in enumerate(train_data.iterrows()):
        name = row['NAME']
        actual = y_train[i]
        pred = y_train_pred[i]
        error = pred - actual
        print(f"{i+1:<5} {name:<25} {actual:>7.2f}% {pred:>9.2f}% {error:>+7.2f}%")

def save_model_and_scaler(model, scaler):
    """Save trained model and scaler"""
    print("\n" + "="*70)
    print("STEP 8: Saving Model and Scaler")
    print("="*70)
    
    # Save model
    with open("Model/Decision_Tree_Trained.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✓ Saved model to Model/Decision_Tree_Trained.pkl")
    
    # Save scaler
    with open("Model/scaler_trained.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Saved scaler to Model/scaler_trained.pkl")
    
    # Save scaler parameters separately for easy reference
    scaler_params = {
        'mean': float(scaler.mean_[0]),
        'std': float(scaler.scale_[0])
    }
    
    with open("Model/scaler_params.pkl", "wb") as f:
        pickle.dump(scaler_params, f)
    print("✓ Saved scaler parameters to Model/scaler_params.pkl")
    
    print(f"\nScaler Parameters:")
    print(f"  Mean: {scaler.mean_[0]:.10f}")
    print(f"  Std: {scaler.scale_[0]:.10f}")

def print_tree_structure(model):
    """Print tree structure"""
    print("\n" + "="*70)
    print("STEP 9: Tree Structure")
    print("="*70)
    
    try:
        from sklearn.tree import export_text
        tree_rules = export_text(model, feature_names=['MWC08_TIC1_PW5_scaled'])
        print("\n" + tree_rules)
    except Exception as e:
        print(f"Could not export tree: {e}")

def main():
    """Main training pipeline"""
    print("\n" + "🌊"*35)
    print("TRAIN DECISION TREE MODEL FROM CSV DATA")
    print("🌊"*35)
    
    # Create directory
    create_model_directory()
    
    # Load descriptors database
    df_desc = load_descriptors()
    
    # Load training data
    train_data, test_data = load_training_data()
    
    # Calculate combinatorial descriptors
    X_train_raw = calculate_all_descriptors(train_data, df_desc)
    X_test_raw = calculate_all_descriptors(test_data, df_desc) if len(test_data) > 0 else np.array([])
    
    # Get target values
    y_train = train_data['AlgaeRemoval110kpa'].values
    y_test = test_data['AlgaeRemoval110kpa'].values if len(test_data) > 0 else np.array([])
    
    print("\n" + "="*70)
    print("STEP 4: Scaling Descriptors")
    print("="*70)
    
    # Scale descriptors
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test_raw.reshape(-1, 1)) if len(X_test_raw) > 0 else np.array([])
    
    print(f"\nRaw Descriptors:")
    print(f"  Min: {X_train_raw.min():.6f}")
    print(f"  Max: {X_train_raw.max():.6f}")
    print(f"  Mean: {X_train_raw.mean():.6f}")
    print(f"  Std: {X_train_raw.std():.6f}")
    
    print(f"\nScaled Descriptors:")
    print(f"  Min: {X_train_scaled.min():.6f}")
    print(f"  Max: {X_train_scaled.max():.6f}")
    print(f"  Mean: {X_train_scaled.mean():.10f}")
    print(f"  Std: {X_train_scaled.std():.10f}")
    
    print(f"\nScaler Parameters:")
    print(f"  Mean: {scaler.mean_[0]:.10f}")
    print(f"  Std: {scaler.scale_[0]:.10f}")
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate
    evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Analyze
    analyze_predictions(model, X_train_scaled, y_train, train_data)
    
    # Save
    save_model_and_scaler(model, scaler)
    
    # Print tree
    print_tree_structure(model)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  ✓ Trained with {len(X_train_raw)} samples")
    print(f"  ✓ Model has {model.get_n_leaves()} leaf nodes")
    print(f"  ✓ Tree depth: {model.get_depth()}")
    print(f"  ✓ Files saved:")
    print(f"     - Model/Decision_Tree_Trained.pkl")
    print(f"     - Model/scaler_trained.pkl")
    print(f"     - Model/scaler_params.pkl")
    print(f"\nNext: Run the Streamlit app!")
    print(f"Command: streamlit run algae_app_complete.py")

if __name__ == "__main__":
    main()


# -*- coding: utf-8 -*-
"""
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Algae Removal Predictor",
    page_icon="🌊",
    layout="wide"
)

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        with open("Model/Decision_Tree_Trained.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("Model/scaler_trained.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # st.sidebar.success(f"✅ Model loaded ({model.get_n_leaves()} leaves)")
        return model, scaler
        
    except FileNotFoundError:
        st.error("❌ Model not found. Please run train_complete.py first!")
        st.stop()

@st.cache_data
def load_descriptors():
    """Load component descriptors"""
    try:
        df_desc = pd.read_csv("Dataset/descriptors.csv")
        df_desc.columns = df_desc.columns.str.strip().str.replace('﻿', '')
        df_desc['NAME'] = df_desc['NAME'].astype(str).str.strip()
        # st.sidebar.success("✅ Descriptors loaded")
        return df_desc
    except FileNotFoundError:
        st.error("❌ descriptors.csv not found!")
        st.stop()

# Load everything
model, scaler = load_model_and_scaler()
descriptors_df = load_descriptors()

# --- Main UI ---
st.title("🌊 Algae Removal Predictor")
#st.markdown("### Simple Formulation-Based Prediction")

try:
    st.image("header.png", use_container_width=True)
except:
    pass

st.markdown("---")

# --- Component Selection ---
st.header("🧪 Component Selection")

col1, col2, col3 = st.columns(3)

with col1:
    pdms_type = st.selectbox(
        "PDMS Type",
        ["DMS-V22", "DMS-V31"],
        help="Polydimethylsiloxane base polymer"
    )

with col2:
    siloxane_type = st.selectbox(
        "Siloxane Crosslinker",
        ["HMS-151", "HMS-301"],
        help="Crosslinking agent"
    )

with col3:
    oil_type = st.selectbox(
        "Silicone Oil",
        ["None", "PMM-0025", "PMM-1015", "PMM-1021", "PDM-0421"],
        help="Optional silicone oil additive"
    )

# --- Component Amounts ---
st.header("⚖️ Component Amounts (grams)")

c1, c2, c3 = st.columns(3)

with c1:
    g_pdms = st.number_input(
        f"{pdms_type} (g)",
        min_value=0.0,
        value=20.0,
        step=0.1
    )

with c2:
    default_sil = 0.91 if "151" in siloxane_type else 1.251
    g_sil = st.number_input(
        f"{siloxane_type} (g)",
        min_value=0.0,
        value=default_sil,
        step=0.01
    )

with c3:
    g_oil = st.number_input(
        "Oil (g)",
        min_value=0.0,
        value=0.0 if oil_type == "None" else 1.0,
        step=0.1,
        disabled=(oil_type == "None")
    )

st.markdown("---")

# --- Calculate and Predict ---
try:
    # Get molecular descriptors
    mwc08 = descriptors_df.loc[descriptors_df['NAME'] == pdms_type, 'MWC08'].values[0]
    tic1 = descriptors_df.loc[descriptors_df['NAME'] == siloxane_type, 'TIC1'].values[0]
    pw5 = 0.0 if oil_type == "None" else descriptors_df.loc[descriptors_df['NAME'] == oil_type, 'PW5'].values[0]
    
    # Calculate total mass
    total_mass = g_pdms + g_sil + g_oil
    
    if total_mass == 0:
        st.warning("⚠️ Please enter component amounts")
        st.stop()
    
    # Calculate fractions
    frac_pdms = g_pdms / total_mass
    frac_sil = g_sil / total_mass
    frac_oil = g_oil / total_mass
    
    # Calculate combinatorial descriptor
    raw_descriptor = (frac_pdms * mwc08) + (frac_sil * tic1) + (frac_oil * pw5)
    
    # Scale descriptor
    scaled_descriptor = scaler.transform([[raw_descriptor]])[0][0]
    
    # --- Prediction Button ---
    if st.button("🔮 Predict Algae Removal", type="primary", use_container_width=True):
        
        # Make prediction
        prediction = model.predict([[scaled_descriptor]])[0]
        
        # --- Display Result ---
        st.markdown("---")
        st.success(f"### 🎯 Predicted Algae Removal: **{prediction:.2f}%**")
        
        # --- Formulation Summary (Components and Amounts Only) ---
        st.markdown("---")
        st.subheader("📋 Formulation Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label=pdms_type,
                value=f"{g_pdms:.2f} g",
                delta=f"{frac_pdms*100:.1f}%"
            )
        
        with col2:
            st.metric(
                label=siloxane_type,
                value=f"{g_sil:.2f} g",
                delta=f"{frac_sil*100:.1f}%"
            )
        
        with col3:
            st.metric(
                label=oil_type if oil_type != "None" else "No Oil",
                value=f"{g_oil:.2f} g",
                delta=f"{frac_oil*100:.1f}%" if g_oil > 0 else "0%"
            )
        

except KeyError as e:
    st.error(f"❌ Component not found in descriptors database: {e}")
    st.info("Available components in descriptors.csv:")
    st.write(descriptors_df['NAME'].tolist())
    
except Exception as e:
    st.error(f"❌ An error occurred: {e}")
    import traceback
    with st.expander("Show error details"):
        st.code(traceback.format_exc())

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Algae Removal Predictor | Decision Tree Model"
    "</div>",
    unsafe_allow_html=True
)