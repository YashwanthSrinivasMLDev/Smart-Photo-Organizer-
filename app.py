import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms.v2 as T
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


# --- Load the pre-trained PyTorch model ---
@st.cache_resource
def load_model():
    """Loads a pre-trained PyTorch model for image classification."""
    weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
    model = torchvision.models.mobilenet_v3_large(weights=weights)
    model.eval()  # Set the model to evaluation mode
    return model, weights.meta["categories"]


model, categories = load_model()


# --- Image Processing Function ---
def classify_image(img, model, categories):
    """
    Classifies a single image using the pre-trained PyTorch model.
    Args:
        img (PIL.Image): The input image.
        model (torch.nn.Module): The pre-trained classification model.
        categories (list): The list of class labels.
    Returns:
        str: The predicted class label.
    """
    # Define preprocessing transforms
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image and add a batch dimension
    img_tensor = preprocess(img).unsqueeze(0)

    # Make predictions
    with torch.no_grad():
        predictions = model(img_tensor)

    # Get the top prediction
    predicted_class_idx = predictions.argmax(1).item()
    predicted_label = categories[predicted_class_idx]

    return predicted_label.replace('_', ' ').title()


# --- File Uploader ---
uploaded_files = st.file_uploader(
    "Choose images to upload",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("Classification Results")
    st.write("Processing... This may take a moment.")

    classification_results = {}
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert('RGB')
            label = classify_image(img, model, categories)

            if label not in classification_results:
                classification_results[label] = []

            classification_results[label].append({
                "name": uploaded_file.name,
                "image_data": uploaded_file.getvalue()
            })

            st.write(f"✓ **{uploaded_file.name}** classified as: **{label}**")

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

    # --- ZIP file creation and download button ---
    if classification_results:
        st.subheader("Download Sorted Images")

        # Use a BytesIO buffer to create the ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for label, files in classification_results.items():
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

st.markdown("---")
st.markdown(
    """
    #### How it works:
    1.  Upload one or more images.
    2.  A pre-trained AI model (`MobileNetV3`) from PyTorch's `torchvision` library is used to classify them.
    3.  The app shows the classification results.
    4.  A download button appears to get a ZIP file with your images sorted into folders.
    """
)
st.markdown(
    """
    _Powered by PyTorch and Streamlit._
    """
)