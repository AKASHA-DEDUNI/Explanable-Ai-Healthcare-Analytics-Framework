import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
import os, io
import torch
from torchvision import transforms
import numpy as np

# ----------------------------
# Device & Model Setup
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import your model
from model import MultiModalPneumoniaModel  # replace with actual file


@st.cache_resource
def load_model():
    model = MultiModalPneumoniaModel().to(DEVICE)
    model.load_state_dict(torch.load(r"C:\Users\ASUS\Desktop\resnet50\best_multimodal_model_resnet50.pth", map_location=DEVICE))
    model.eval()
    return model


model = load_model()


# ----------------------------
# Load Bounding Box Data
# ----------------------------
@st.cache_data
def load_bbox_data():
    df = pd.read_csv(r"C:\Users\ASUS\Desktop\resnet50\BBox_List_2017.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Bbox [x': 'Bbox x', 'h]': 'h'})
    df['Image Index'] = df['Image Index'].str.strip().str.lower()
    return df


df = load_bbox_data()


# ----------------------------
# Draw Bounding Boxes
# ----------------------------
def draw_bounding_boxes(image, image_name, df):
    matching_rows = df[df["Image Index"] == image_name]
    if matching_rows.empty:
        return image, 0

    orig_w, orig_h = image.size
    max_width = 700
    resize_ratio = min(max_width / orig_w, 1.0)
    new_w, new_h = int(orig_w * resize_ratio), int(orig_h * resize_ratio)
    resized_image = image.resize((new_w, new_h))

    draw = ImageDraw.Draw(resized_image)
    for _, row in matching_rows.iterrows():
        x = float(row["x"]) * resize_ratio
        y = float(row["y"]) * resize_ratio
        w = float(row["w"]) * resize_ratio
        h = float(row["h"]) * resize_ratio
        label = row["Finding Label"]

        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.rectangle([x, y - 15, x + len(label) * 6, y], fill="black")
        draw.text((x, y - 12), label, fill="white")

    return resized_image, len(matching_rows)


# ----------------------------
# Grad-CAM Placeholder
# ----------------------------
def generate_gradcam(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(img_tensor)
        pred_class = output.argmax(dim=1).item()

    # TODO: Replace with actual Grad-CAM logic
    gradcam_img = image.point(lambda p: min(255, p * 1.2))  # simple brightening as placeholder
    return gradcam_img


# ----------------------------
# Counterfactual Placeholder
# ----------------------------
def generate_counterfactual(image):
    # TODO: Replace with DiCE / Alibi for real counterfactuals
    cf_image = image.point(lambda p: min(255, p * 1.1))  # slightly brighter as placeholder

    # Example tabular feature changes
    feature_changes = pd.DataFrame({
        "Feature": ["Lesion area (%)", "Opacity", "Age"],
        "Original": [8.5, 0.7, 67],
        "Counterfactual": [3.2, 0.2, 64],
        "Impact": ["↓ Risk", "↓ Risk", "↓ Risk"]
    })

    summary_text = "Reducing lesion area and opacity changed prediction from 'Pneumonia' → 'No Pneumonia'."

    return cf_image, feature_changes, summary_text


# ----------------------------
# Streamlit App
# ----------------------------
st.title("🩻 Explainable AI: Multimodal Pneumonia Counterfactual Viewer")

uploaded_file = st.file_uploader("Upload NIH Chest X-ray", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_name = os.path.basename(uploaded_file.name).lower().strip()

    # 1️⃣ Bounding boxes
    img_with_box, box_count = draw_bounding_boxes(image, image_name, df)

    # 2️⃣ Grad-CAM
    gradcam_img = generate_gradcam(image, model)

    # 3️⃣ Counterfactual
    cf_img, feature_changes, summary_text = generate_counterfactual(image)
    cf_with_box, _ = draw_bounding_boxes(cf_img, image_name, df)

    # 4️⃣ Side-by-side comparison
    st.subheader("Original vs Counterfactual vs Grad-CAM")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_with_box, caption=f"Original ({box_count} boxes)", use_container_width=True)
    with col2:
        st.image(gradcam_img, caption="Grad-CAM Overlay", use_container_width=True)
    with col3:
        st.image(cf_with_box, caption="Counterfactual", use_container_width=True)

    # 5️⃣ Feature Changes
    st.subheader("🔹 Feature Changes Leading to Counterfactual Prediction")
    st.dataframe(feature_changes)

    # 6️⃣ Doctor-Friendly Summary
    st.subheader("📄 Summary for Doctor")
    st.write(summary_text)
