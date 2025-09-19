# streamlit_pneumonia_app.py
import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import io
import os

from model import MultiModalPneumoniaModel
from utils import GradCAM, apply_heatmap

# --- Load CSV ---
@st.cache_data
def load_bbox_data():
    df = pd.read_csv(r"C:\Users\ASUS\Desktop\resnet50\BBox_List_2017.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        'Bbox [x': 'x',
        'y]': 'y',
        'w': 'w',   # width
        'h]': 'h'
    })
    df['Image Index'] = df['Image Index'].str.strip().str.lower()
    return df

bbox_df = load_bbox_data()

def get_bbox_for_image(image_name):
    return bbox_df[bbox_df['Image Index'] == image_name]

# --- Load model ---
@st.cache_resource
def load_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalPneumoniaModel().to(DEVICE)
    model.load_state_dict(torch.load(r"C:\Users\ASUS\Desktop\resnet50\best_multimodal_model_resnet50.pth", map_location=DEVICE))
    model.eval()
    gradcam = GradCAM(model, model.cnn.layer4[-1].conv2)
    return model, gradcam, DEVICE

model, gradcam, DEVICE = load_model()

# --- Image transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Draw bounding boxes ---
def draw_bbox(image, bbox_rows, resize_ratio=1.0):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for _, row in bbox_rows.iterrows():
        try:
            x = float(row["x"]) * resize_ratio
            y = float(row["y"]) * resize_ratio
            w = float(row["w"]) * resize_ratio
            h = float(row["h"]) * resize_ratio
            label = row["Finding Label"]

            draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
            draw.text((x, max(0, y - 12)), label, fill="red")
        except Exception as e:
            st.error(f"Error drawing bounding box: {e}")
    return img_copy

# --- Streamlit App ---
st.title("🩺 Pneumonia Grad-CAM + Bounding Box Viewer")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
age = st.number_input("Patient Age", min_value=0, max_value=120, value=40)
gender = st.selectbox("Gender", ["M", "F"])
text_report = st.text_area("Radiology Report (optional)", "No report")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_name = uploaded_file.name.lower().strip()
    bbox_rows = get_bbox_for_image(image_name)

    # Resize for display
    orig_w, orig_h = image.size
    max_width = 700
    resize_ratio = min(max_width / orig_w, 1.0)
    display_w = int(orig_w * resize_ratio)
    display_h = int(orig_h * resize_ratio)
    resized_image = image.resize((display_w, display_h))

    st.subheader("Original Image")
    st.image(resized_image, use_container_width=True)

    # Bounding box overlay
    if not bbox_rows.empty:
        st.success(f"Found {len(bbox_rows)} bounding box(es)")
        bbox_image = draw_bbox(resized_image, bbox_rows, resize_ratio=resize_ratio)
        st.subheader("Bounding Box Overlay")
        st.image(bbox_image, use_container_width=True)
    else:
        st.warning("No bounding box found for this image.")

    # --- Grad-CAM ---
    st.subheader("Grad-CAM Heatmap")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    gender_val = 0 if gender == "M" else 1
    meta = torch.tensor([[age, gender_val]], dtype=torch.float32).to(DEVICE)
    text_emb = torch.zeros((1, 384)).to(DEVICE)

    with torch.no_grad():
        pred = model(image_tensor, meta, text_emb)
        pred_class = pred.argmax(dim=1).item()

    heatmap = gradcam.generate(image_tensor, meta, text_emb, class_idx=pred_class)
    heatmap_img = apply_heatmap(image, heatmap).resize((display_w, display_h))

    st.image(heatmap_img, use_container_width=True)
