import streamlit as st
import joblib
import numpy as np
import pandas as pd
import subprocess
import sys
import base64
import os

# -------------------------
# Configure page settings
# -------------------------
st.set_page_config(page_title="AI Health Assistant", layout="centered")

# -------------------------
# Set background image
# -------------------------
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        * {{
            color: white !important;
            font-weight: bold !important;
        }}
        </style>
        """, unsafe_allow_html=True)

set_background("C:/Users/arnav/DiseaseRecognitionAndDiagnosis/backgroundimage.jpeg")

# -------------------------
# Title box
# -------------------------
st.markdown("""
<div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 28px; text-align: center;'>
🧪 AI Health Assistant
</div>
""", unsafe_allow_html=True)

# -------------------------
# Input method selection
# -------------------------
st.markdown("""
<div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 20px; margin-bottom: 10px;'>
<strong>Choose input method:</strong>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    text_clicked = st.button("Text (Symptoms)", key="text_method")
with col2:
    image_clicked = st.button("Image", key="image_method")

if "input_method" not in st.session_state:
    st.session_state.input_method = None
if text_clicked:
    st.session_state.input_method = "Text (Symptoms)"
if image_clicked:
    st.session_state.input_method = "Image"

option = st.session_state.input_method

# -------------------------
# TEXT MODE
# -------------------------
if option == "Text (Symptoms)":
    model_disease = joblib.load("model_disease.pkl")
    model_treatment = joblib.load("model_treatment.pkl")
    model_medicine = joblib.load("model_medicine.pkl")
    model_treatdur = joblib.load("model_treatmentduration.pkl")
    model_medidur = joblib.load("model_medicineduration.pkl")
    symptoms = joblib.load("symptom_list.pkl")

    df = pd.read_csv(r"C:\Users\arnav\FINALDISEASERECOGNITIONPROJECT\disease_symptom_matrix (9).csv", encoding='cp1252')
    df.columns = df.columns.str.strip()
    df.rename(columns={"1_Disease": "Disease"}, inplace=True)

    if "filtered_diseases" not in st.session_state:
        st.session_state.filtered_diseases = df.index.tolist()
    if "selected_symptoms" not in st.session_state:
        st.session_state.selected_symptoms = []
    if "symptom_times" not in st.session_state:
        st.session_state.symptom_times = {}
    if "input_finished" not in st.session_state:
        st.session_state.input_finished = False
    if "symptom_index" not in st.session_state:
        st.session_state.symptom_index = 0

    # -------------------------
    # Symptom input form
    # -------------------------
    st.markdown("""
    <div style='border: 2px solid white; padding: 10px; border-radius: 10px; background-color: black; font-size: 20px;'>
    📅 Add Symptoms One by One
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.input_finished:
        remaining_symptoms = [s for s in symptoms if s not in st.session_state.selected_symptoms]
        with st.form(f"symptom_form_{st.session_state.symptom_index}"):
            st.markdown("<div style='background-color: black; padding: 10px;'>Select a symptom:</div>", unsafe_allow_html=True)
            selected_symptom = st.selectbox("", [""] + remaining_symptoms)
            st.markdown("<div style='background-color: black; padding: 10px;'>How long have you had this symptom?</div>", unsafe_allow_html=True)
            duration = st.text_input("", value="")
            submitted = st.form_submit_button("✅ Confirm Symptom")

        if submitted:
            if selected_symptom.strip() == "" or duration.strip() == "":
                st.warning("⚠️ Please select a symptom and enter duration.")
            else:
                st.session_state.selected_symptoms.append(selected_symptom)
                st.session_state.symptom_times[selected_symptom] = duration
                filtered_df = df.loc[st.session_state.filtered_diseases]
                filtered_df = filtered_df[filtered_df[selected_symptom] == 1]
                st.session_state.filtered_diseases = filtered_df.index.tolist()
                st.session_state.symptom_index += 1

        if st.button("🚩 No More Symptoms"):
            st.session_state.input_finished = True

    # -------------------------
    # Display selected symptoms
    # -------------------------
    if st.session_state.selected_symptoms:
        st.markdown("<div style='border: 2px solid white; padding: 10px; border-radius: 10px; background-color: black; font-size: 20px;'>✅ Selected Symptoms & Duration:</div>", unsafe_allow_html=True)
        for sym in st.session_state.selected_symptoms:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.markdown(f"<div style='border: 2px solid green; padding: 10px; border-radius: 10px; background-color: black; font-size: 16px; margin-bottom: 5px;'>🦠 <strong>{sym}</strong> — ⏱️ {st.session_state.symptom_times[sym]}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{sym}"):
                    st.session_state.selected_symptoms.remove(sym)
                    st.session_state.symptom_times.pop(sym, None)
                    st.session_state.filtered_diseases = df.index.tolist()
                    for s in st.session_state.selected_symptoms:
                        filtered_df = df.loc[st.session_state.filtered_diseases]
                        filtered_df = filtered_df[filtered_df[s] == 1]
                        st.session_state.filtered_diseases = filtered_df.index.tolist()
                    st.rerun()

    # -------------------------
    # Final Prediction
    # -------------------------
    if st.session_state.input_finished and st.button("🔍 Predict Disease"):
        input_vector = [1 if s in st.session_state.selected_symptoms else 0 for s in symptoms]
        input_array = np.array(input_vector).reshape(1, -1)

        pred_disease = model_disease.predict(input_array)[0]
        pred_treatment = model_treatment.predict(input_array)[0]
        pred_medicine = model_medicine.predict(input_array)[0]
        pred_treatdur = model_treatdur.predict(input_array)[0]
        pred_medidur = model_medidur.predict(input_array)[0]

        st.markdown(f"""
        <div style='border: 2px solid green; padding: 15px; border-radius: 10px; background-color: black; font-size: 20px;'>
        🧬 <strong>Disease:</strong> {pred_disease}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 18px; margin-top: 10px;'>
        💉 <strong>Treatment:</strong> {pred_treatment}<br>
        ⏱️ <strong>Estimated Duration:</strong> {pred_treatdur}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 18px; margin-top: 10px;'>
        💊 <strong>Medicine:</strong> {pred_medicine}<br>
        ⏱️ <strong>Estimated Duration:</strong> {pred_medidur}
        </div>
        """, unsafe_allow_html=True)

    # ✅ Restart logic
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------------
# IMAGE MODE
# -------------------------
elif option == "Image":
    st.markdown("""
    <div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 20px; text-align: center;'>
    🔄 Redirecting to image-based model...
    </div>
    """, unsafe_allow_html=True)

    if st.button("Launch Image Classifier"):
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "ma2.py"])
        st.stop()





