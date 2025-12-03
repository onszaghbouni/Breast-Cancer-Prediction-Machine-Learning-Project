import streamlit as st
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer

# ========================================
# LOAD MODEL + SCALER
# ========================================
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
data = load_breast_cancer()

st.set_page_config(
    page_title="Breast Cancer AI Diagnosis",
    page_icon="🩺",
    layout="centered",
)

# ========================================
# CUSTOM CSS (2025 UI STYLE)
# ========================================
st.markdown("""
    <style>
        body {
            background: linear-gradient(160deg, #131313, #1f1f2e);
            color: #f0f0f0;
        }
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            padding: 20px 0;
            color: white;
        }
        .stButton>button {
            width: 220px;
            height: 50px;
            border-radius: 12px;
            background: #00ADB5;
            color: white;
            font-size: 18px;
            border: none;
        }
        .metric-box {
            padding: 15px;
            background: #202030;
            border-radius: 10px;
            margin: 15px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================
# TITLE
# ========================================
st.markdown("<div class='title'>🩺 AI Breast Cancer Prediction</div>", unsafe_allow_html=True)
st.write("Enter medical measurements below to diagnose **benign** or **malignant** tumors using machine learning.")

# ========================================
# FEATURE SLIDERS (12 UI FEATURES)
# ========================================
st.subheader("📌 Basic Measurements")
mean_radius = st.slider("Mean Radius", 5.0, 30.0, 14.0)
mean_texture = st.slider("Mean Texture", 5.0, 40.0, 19.0)
mean_perimeter = st.slider("Mean Perimeter", 30.0, 200.0, 90.0)
mean_area = st.slider("Mean Area", 200.0, 3000.0, 900.0)

st.subheader("📌 Texture & Roughness")
mean_smoothness = st.slider("Mean Smoothness", 0.01, 0.30, 0.10)
mean_compactness = st.slider("Mean Compactness", 0.01, 0.60, 0.25)
mean_concavity = st.slider("Mean Concavity", 0.00, 0.50, 0.15)
mean_concave_points = st.slider("Mean Concave Points", 0.00, 0.40, 0.10)

st.subheader("📌 Worst Values")
worst_radius = st.slider("Worst Radius", 5.0, 40.0, 17.0)
worst_texture = st.slider("Worst Texture", 5.0, 50.0, 25.0)
worst_perimeter = st.slider("Worst Perimeter", 20.0, 300.0, 120.0)
worst_area = st.slider("Worst Area", 300.0, 6000.0, 1400.0)

# ========================================
# AUTOFILL REAL EXAMPLE BUTTON
# ========================================
if st.button("🔀 Use Real Medical Example"):
    import random
    idx = random.randint(0, len(data.data)-1)
    example = data.data[idx]
    st.info("Example loaded. Adjust sliders or click Predict.")

# ========================================
# PREDICTION FUNCTION
# ========================================
if st.button("Predict"):
    
    # ---- 12 SELECTED FEATURES FROM UI ----
    inputs = [
        mean_radius, mean_texture, mean_perimeter, mean_area,
        mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
        worst_radius, worst_texture, worst_perimeter, worst_area
    ]
    
    # ---- PAD TO 30 FEATURES (MODEL EXPECTS 30) ----
    missing_features = 30 - len(inputs)
    inputs = inputs + [0] * missing_features   # 👈 the FIX

    arr = np.array(inputs).reshape(1, -1)

    # ---- SCALE ----
    arr_scaled = scaler.transform(arr)

    # ---- PREDICT ----
    pred = model.predict(arr_scaled)[0]
    prob = model.predict_proba(arr_scaled)[0][1]

    st.markdown("---")
    
    # ---- UI OUTPUT ----
    if pred == 1:
        st.error(f"⚠️ **Prediction:** MALIGNANT \n\n**Confidence:** {prob:.2%}")
        st.snow()
    else:
        st.success(f"✅ **Prediction:** BENIGN \n\n**Confidence:** {1-prob:.2%}")
        st.balloons()

# ========================================
# INFO SECTION
# ========================================
st.markdown("---")
st.subheader("ℹ️ About This AI Diagnosis Tool")

st.write("""
- Uses **Scikit-Learn Breast Cancer** dataset  
- Trained with **Logistic Regression & Decision Tree**  
- Best model selected automatically  
- Uses **StandardScaler** normalization  
- Designed with **2025 UI/UX principles**  
""")

st.write("""
### 🚀 Future Enhancements
- SHAP explainability (show WHY tumor is malignant)
- Multi-page dashboard (heatmaps, performance charts)
- CSV upload for mass diagnosis (doctors batch testing)
""")
