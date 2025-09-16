import streamlit as st
import os
import json
from train_and_save_model import train_and_save_model

st.set_page_config(page_title="AutoML Model Trainer", layout="wide")

st.title("🤖 AutoML Model Trainer")
st.write("Upload a CSV file and let the app detect problem type, train multiple models, and save the best one.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary local path
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model_name = st.text_input("Enter Model Name", value="MyCustomModel")

    if st.button("🚀 Train Models"):
        # When starting a new training, clear any results from a previous run
        if 'training_complete' in st.session_state:
            del st.session_state['training_complete']
            # ... and other related keys

        with st.spinner("Training models... Please wait."):
            # This function will now save its results to st.session_state
            train_and_save_model(file_path, model_name)

        # After training, the script will automatically rerun, and the section below will activate.

# --- ✨ NEW: Display download buttons if results are in memory ✨ ---
# This code runs on EVERY page refresh.
if st.session_state.get('training_complete', False):
    st.success("🏆 Training Complete! Your files are ready for download.")

    st.subheader("📑 Model Metadata")
    # Load metadata from session state to display it
    st.json(json.loads(st.session_state['json_string']))

    st.subheader("⬇️ Download Your Files")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Trained Model (.joblib)",
            data=st.session_state['model_bytes'],
            file_name=st.session_state['model_filename'],
            mime="application/octet-stream"
        )
    with col2:
        st.download_button(
            label="Download Metadata (.json)",
            data=st.session_state['json_string'],
            file_name=st.session_state['json_filename'],
            mime="application/json"
        )