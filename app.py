import streamlit as st
from PIL import Image, UnidentifiedImageError
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import predict_bird, is_bird_image

st.set_page_config(
    page_title="BirdVision - Bird Species Classification",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 BirdVision")
st.subheader("Bird Species Classification Using CNN")

st.write(
    "Upload a bird photograph. The system first checks whether "
    "the uploaded image contains a bird, then predicts its species."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose a bird image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=350)

        if st.button("Identify Bird Species", type="primary"):
            with st.spinner("Checking the uploaded image..."):
                try:
                    if not is_bird_image(image):
                        st.error(
                            "❌ This does not appear to be a bird image. "
                            "Please upload a photograph of a bird."
                        )
                    else:
                        with st.spinner("Identifying bird species..."):
                            species, confidence, top_predictions = predict_bird(image)

                        st.success("Bird image verified! Prediction completed.")

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
                                min(1.0, max(0.0, item['confidence'] / 100))
                            )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    except (UnidentifiedImageError, OSError):
        st.error("Invalid image. Please upload a valid JPG or PNG file.")

st.divider()

st.caption(
    "BirdVision | CNN-based Bird Species Classification | "
    "MobileNetV2 Transfer Learning"
)
