"""Local browser interface for uploading an image and predicting its plant species."""

from io import BytesIO

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.predict import load_class_mapping, load_inference_model, predict_image

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@st.cache_resource
def load_model_bundle():
    model = load_inference_model()
    class_names = load_class_mapping()
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"The model has {model.output_shape[-1]} outputs but "
            f"{len(class_names)} class names are configured."
        )
    return model, class_names


st.set_page_config(page_title="Plant Species Classifier", layout="centered")
st.title("Plant Species Classifier")
st.write("Upload a clear plant image to see the predicted species.")

uploaded_image = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, PNG, BMP, and WebP. Maximum size: 20 MB.",
)

if uploaded_image is not None:
    if uploaded_image.size > MAX_UPLOAD_BYTES:
        st.error("This image is larger than 20 MB. Please choose a smaller image.")
        st.stop()

    try:
        with Image.open(BytesIO(uploaded_image.getvalue())) as opened_image:
            opened_image.load()
            image = opened_image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        st.error(f"Could not read this image: {error}")
        st.stop()

    preview = image.copy()
    preview.thumbnail((700, 700))
    st.image(preview, caption=uploaded_image.name)
    if st.button("Predict species", type="primary"):
        try:
            with st.spinner("Analyzing image..."):
                model, class_names = load_model_bundle()
                result = predict_image(image, model, class_names)
        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
            st.stop()

        st.success(
            f"Prediction: {result['predicted_species']} "
            f"({result['confidence_percentage']:.1f}%)"
        )
        st.subheader("Top predictions")
        st.table(result["top_predictions"])
