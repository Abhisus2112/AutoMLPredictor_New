import streamlit as st
import os
from train_and_save_model import train_and_save_model  # your function in a separate file

st.set_page_config(page_title="AutoML Model Trainer", layout="wide")

st.title("🤖 AutoML Model Trainer")
st.write("Upload a CSV file and let the app detect problem type, train multiple models, and save the best one.")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")

    model_name = st.text_input("Enter Model Name", value="MyCustomModel")

    if st.button("🚀 Train Models"):
        with st.spinner("Training models... Please wait."):
            best_model, model_file, metadata = train_and_save_model(file_path, model_name)

        if best_model:
            st.success("🏆 Training Complete! Best model found.")

            # --- Metadata ---
            st.subheader("📑 Model Metadata")
            st.json(metadata)

            # --- Download Buttons ---
            if os.path.exists(model_file):
                with open(model_file, "rb") as f:
                    st.download_button("⬇️ Download Trained Model", f, file_name=model_file)

            meta_file = f"{model_name}_metadata.json"
            if os.path.exists(meta_file):
                with open(meta_file, "rb") as f:
                    st.download_button("⬇️ Download Metadata JSON", f, file_name=meta_file)
