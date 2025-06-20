import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import base64

# ----------------------------
# ✅ Paths
# ----------------------------
model_path = r"C:\Users\arnav\DiseaseRecognitionAndDiagnosis\diseaserecognition.h5"
class_indices_path = r"C:\Users\arnav\DiseaseRecognitionAndDiagnosis\class_indices1.json"
csv_path = r"C:\Users\arnav\FINALDISEASERECOGNITIONPROJECT\image_dataset.csv"
background_path = r"C:\Users\arnav\DiseaseRecognitionAndDiagnosis\backgroundimage.jpeg"

# ----------------------------
# ✅ Load Model
# ----------------------------
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ----------------------------
# ✅ Load Class Indices
# ----------------------------
try:
    with open(class_indices_path, 'r') as f:
        class_indices = json.load(f)
except Exception as e:
    st.error(f"❌ Error loading class indices: {e}")
    st.stop()

# ----------------------------
# ✅ Preprocess Image
# ----------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    return img_array

# ----------------------------
# ✅ Predict Class
# ----------------------------
def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_indices.get(str(predicted_index), "Unknown")
    return predicted_class

# ----------------------------
# ✅ Background Image
# ----------------------------
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    * {{
        color: white !important;
        font-weight: bold !important;
    }}
    [data-testid="stFileUploader"] section > div:nth-of-type(2) {{
        display: none;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply background
set_background(background_path)

# ----------------------------
# ✅ App Title
# ----------------------------
st.set_page_config(page_title="Disease Classifier", layout="centered")
st.markdown("""
<div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 28px; text-align: center;'>
🧠 Disease Classifier & Treatment Finder
</div>
""", unsafe_allow_html=True)

# ----------------------------
# ✅ Image Upload
# ----------------------------
st.markdown("""
<div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; margin-top: 20px;'>
📸 <strong>Upload an image:</strong>
</div>
""", unsafe_allow_html=True)
uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ----------------------------
# ✅ Prediction
# ----------------------------
if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image.resize((150, 150)), use_container_width=False)
    with col2:
        if st.button("🔍 Classify"):
            prediction = predict_image_class(model, image, class_indices)
            st.session_state.predicted_disease = prediction

            st.markdown(f"""
            <div style='border: 2px solid green; padding: 15px; border-radius: 10px; background-color: black; font-size: 20px; text-align: center;'>
            ✅ Prediction: <strong>{prediction}</strong>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# ✅ Treatment Info Display
# ----------------------------
if 'predicted_disease' in st.session_state:
    st.markdown("""
    <div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; margin-top: 20px;'>
    📝 <strong>Selected Disease (auto-filled after prediction):</strong>
    </div>
    """, unsafe_allow_html=True)

    selected_disease = st.text_input("", value=st.session_state.predicted_disease, key="auto_disease")

    try:
        # Use utf-8-sig to remove BOM (ï»¿)
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()

        required_cols = ['Disease', 'Treatment', 'Medicine', 'Treatment Duration', 'Medicine Duration']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ CSV must contain columns: {', '.join(required_cols)}")
        else:
            row = df[df['Disease'].str.lower() == selected_disease.lower()]
            if not row.empty:
                treatment = row['Treatment'].values[0]
                medicine = row['Medicine'].values[0]
                treat_duration = row['Treatment Duration'].values[0]
                medicine_duration = row['Medicine Duration'].values[0]

                st.markdown(f"""
                <div style='border: 2px solid white; padding: 15px; border-radius: 10px; background-color: black; font-size: 20px; margin-top: 20px;'>
                💊 <strong>Recommended Treatment Plan:</strong><br><br>
                🧪 <strong>Treatment:</strong> {treatment}<br>
                ⏱️ <strong>Treatment Duration:</strong> {treat_duration}<br><br>
                💊 <strong>Medicine:</strong> {medicine}<br>
                ⏱️ <strong>Medicine Duration:</strong> {medicine_duration}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ Disease '{selected_disease}' not found in CSV.")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")


