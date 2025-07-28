import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import zipfile

# Set up the page
st.set_page_config(
    page_title="Smart Photo Organizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📸 Smart Photo Organizer")
st.write("Upload multiple images to have them automatically sorted into folders based on their content.")


# --- Load the pre-trained model ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads a pre-trained MobileNetV2 model for image classification."""
    model = MobileNetV2(weights='imagenet')
    return model


model = load_model()


# --- Image Processing Function ---
def classify_image(img, model):
    """
    Classifies a single image using the pre-trained model.
    Args:
        img (PIL.Image): The input image.
        model (keras.Model): The pre-trained classification model.
    Returns:
        str: The predicted class label.
    """
    # Resize and preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    # Decode the predictions to get a readable label
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    label = decoded_predictions[0][1]
    return label


# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose images to upload",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    # --- Classification and display ---
    st.subheader("Classification Results")
    st.write("Processing... This may take a moment.")

    classification_results = {}
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file)
            label = classify_image(img, model)

            if label not in classification_results:
                classification_results[label] = []

            classification_results[label].append({
                "name": uploaded_file.name,
                "image_data": uploaded_file.getvalue()
            })

            st.write(f"✓ **{uploaded_file.name}** classified as: **{label.replace('_', ' ').title()}**")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # --- ZIP file creation and download button ---
    if classification_results:
        st.subheader("Download Sorted Images")

        # Use a BytesIO buffer to create the ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for label, files in classification_results.items():
                # Create a folder for each category
                for file_info in files:
                    zip_file.writestr(
                        f"{label}/{file_info['name']}",
                        file_info['image_data']
                    )

        st.download_button(
            label="Download Organized Images",
            data=zip_buffer.getvalue(),
            file_name="organized_images.zip",
            mime="application/zip"
        )

# --- Instructions and Credits ---
st.markdown("---")
st.markdown(
    """
    #### How it works:
    1.  Upload one or more images.
    2.  A lightweight AI model (`MobileNetV2`) is used to classify the images.
    3.  The app shows the classification results for each image.
    4.  A download button appears to get a ZIP file with your images sorted into folders.
    """
)
st.markdown(
    """
    _Powered by TensorFlow and Streamlit._
    """
)