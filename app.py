import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered",
)

# ---------- Custom styling (sweet pastel theme) ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(160deg, #fff5f7 0%, #ffe3ea 45%, #ffd4de 100%);
    }

    h1 {
        color: #c9425e !important;
        font-weight: 800 !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #a15769 !important;
    }

    [data-testid="stForm"] {
        background: #ffffffcc;
        padding: 1.8rem 1.8rem 1rem 1.8rem;
        border-radius: 18px;
        border: 1px solid #ffc4d3;
        box-shadow: 0 4px 18px rgba(201, 66, 94, 0.12);
    }

    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #f2748f, #e8607a);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        transition: transform 0.15s ease;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #e8607a, #d94c68);
        color: white;
    }

    [data-testid="stExpander"] {
        background: #ffffffcc;
        border-radius: 12px;
        border: 1px solid #ffc4d3;
    }

    div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 10px !important;
    }

    hr {
        border-color: #ffc4d3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load model artifacts (cached so it only runs once) ----------
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        expected_columns = joblib.load("columns.pkl")
        return model, scaler, expected_columns
    except FileNotFoundError as e:
        st.error(
            "Model files not found. Make sure KNN_heart.pkl, scaler.pkl, "
            "and columns.pkl are in the same folder as app.py."
        )
        st.stop()

model, scaler, expected_columns = load_artifacts()

# ---------- Header ----------
st.title("❤️ Heart Disease Risk Predictor made by Tanmay Sharma")
st.caption("Built by Tanmay_Sharma")
st.markdown(
    "This tool estimates your risk of heart disease using a K-Nearest "
    "Neighbors model trained on clinical data. Fill in your details below "
    "and click **Predict**."
)
st.caption(
    "⚠️ This is an educational tool, not a medical diagnosis. "
    "Always consult a doctor for medical advice."
)
st.divider()

# ---------- Input form ----------
with st.form("prediction_form"):
    st.subheader("Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["ATA", "NAP", "TA", "ASY"],
            help=(
                "ATA: Atypical Angina · NAP: Non-Anginal Pain · "
                "TA: Typical Angina · ASY: Asymptomatic"
            ),
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)", 80, 200, 120
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)", 100, 600, 200
        )
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate Achieved", 60, 220, 150)
        exercise_angina = st.selectbox(
            "Exercise-Induced Angina",
            ["Y", "N"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )
        oldpeak = st.slider(
            "Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1
        )
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ---------- Prediction ----------
if submitted:
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing dummy columns (e.g. dropped baseline categories) with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder to match training column order exactly
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    proba = model.predict_proba(scaled_input)[0]
    risk_proba = proba[1] * 100

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Heart Disease** — estimated {risk_proba:.1f}% risk")
    else:
        st.success(f"✅ **Low Risk of Heart Disease** — estimated {risk_proba:.1f}% risk")

    st.progress(min(int(risk_proba), 100))

    with st.expander("See the details sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))

st.divider()
st.caption("Built with Streamlit · KNN model · by Tanmay_Sharma")
