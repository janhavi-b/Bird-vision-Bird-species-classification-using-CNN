import streamlit as st
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import predict_bird

st.set_page_config(
    page_title="BirdVision - Bird Species Classification",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 BirdVision")
st.subheader("Bird Species Classification Using CNN")

st.write(
    "Upload a bird photograph and our MobileNetV2-based "
    "deep learning model will predict its species."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Bird Image", width=350)

    if st.button("Identify Bird Species", type="primary"):
        with st.spinner("Analyzing the bird image..."):
            try:
                species, confidence, top_predictions = predict_bird(image)

                st.success("Prediction completed!")

                st.subheader("Predicted Bird Species")
                st.header(species.replace("_", " ").title())

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader("Top 5 Predictions")

                for item in top_predictions:
                    st.write(
                        f"**{item['species'].replace('_', ' ').title()}** "
                        f"— {item['confidence']:.2f}%"
                    )
                    st.progress(
                        min(1.0, max(0.0, item["confidence"] / 100))
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")

st.divider()

st.caption(
    "BirdVision | CNN-based Bird Species Classification | "
    "MobileNetV2 Transfer Learning"
)
