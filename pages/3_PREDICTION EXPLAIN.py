import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import requests

from model import MultiModalPneumoniaModel
from sentence_transformers import SentenceTransformer
from utils import GradCAM

# --------------------- Config ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Pneumonia"]

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("API_KEY")

# --------------------- Load Models ---------------------
@st.cache_resource
def load_model():
    model = MultiModalPneumoniaModel().to(DEVICE)
    model.load_state_dict(torch.load(
        r"C:\Users\ASUS\Desktop\resnet50\best_multimodal_model_resnet50.pth",
        map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_text_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --------------------- Load Bounding Boxes ---------------------
@st.cache_data
def load_bbox_data():
    df = pd.read_csv(r"C:\Users\ASUS\Desktop\resnet50\BBox_List_2017.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Bbox [x': 'Bbox x', 'h]': 'h'})
    df['Image Index'] = df['Image Index'].str.strip().str.lower()
    return df

# --------------------- IoU Functions ---------------------
def compute_iou_boxes(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    intersection = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - intersection
    return intersection / union if union != 0 else 0

# --------------------- Grad-CAM Bounding Box Extraction ---------------------
def get_gradcam_bbox(heatmap, threshold=0.5):
    """Extract bounding boxes from Grad-CAM heatmap using a threshold."""
    bboxes = []
    heatmap_uint8 = np.uint8(heatmap * 255)

    # Apply threshold
    _, heatmap_bin = cv2.threshold(heatmap_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Morphological opening to remove weak connections
    kernel = np.ones((5, 5), np.uint8)
    heatmap_clean = cv2.morphologyEx(heatmap_bin, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(heatmap_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, x + w, y + h))

    return bboxes

# --------------------- Overlay Function ---------------------
def overlay_gradcam_with_bbox(image_pil, gradcam_map, gt_bboxes, alpha=0.5, threshold=0.5):
    image_np = np.array(image_pil.resize((224, 224)))
    gradcam_resized = cv2.resize(gradcam_map, (image_np.shape[1], image_np.shape[0]))
    gradcam_color = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
    gradcam_color = cv2.cvtColor(gradcam_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 1 - alpha, gradcam_color, alpha, 0)

    # Draw GT boxes in green
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw Grad-CAM boxes in red
    cam_bboxes = get_gradcam_bbox(gradcam_resized, threshold)
    for bbox in cam_bboxes:
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    return overlay, cam_bboxes

# --------------------- Visualization Function ---------------------
def visualize_gt_boxes(image_pil, gt_bboxes):
    """Draw only the scaled ground truth bounding boxes on the resized image."""
    image_resized = image_pil.resize((224, 224))
    image_np = np.array(image_resized)
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_np

# --------------------- LLM Explanation Function ---------------------
def explain_gradcam(pred_class, confidence, iou_scores, num_cam_regions):
    """Ask an LLM to generate explanation for Grad-CAM results."""
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY not set. Please configure your .env file."

    prompt = f"""
    The model predicted **{pred_class}** with confidence {confidence:.2f}.
    The Grad-CAM highlighted {num_cam_regions} suspicious regions in the lung X-ray.
    IoU scores with ground truth bounding boxes are: {iou_scores if iou_scores else 'No GT available'}.

    Please provide a radiology-style explanation of what the heatmap highlights mean 
    for this case, including reasoning about pneumonia vs. normal findings.
    """

    response = requests.post(
        GROQ_API_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400
        },
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return f"❌ LLM Error: {response.text}"

# --------------------- Streamlit App ---------------------
st.title("📊 Pneumonia Grad-CAM Bounding Box & IoU + LLM Explanation")

model = load_model()
text_model = load_text_model()
gradcam = GradCAM(model, model.cnn.layer4[-1].conv2)
df_bbox = load_bbox_data()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Patient Gender", options=["M", "F"])
report = st.text_area("Radiology Report Text")

# Grad-CAM threshold slider
threshold_val = st.slider("Grad-CAM Threshold", 0.1, 0.9, 0.5, 0.05)

if st.button("Run Prediction & Compute IoU"):
    if uploaded_file is None or report.strip() == "":
        st.error("Please upload image and enter report text.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = os.path.basename(uploaded_file.name).lower().strip()

        # Preprocess inputs
        image_t = transform(image).unsqueeze(0).to(DEVICE)
        gender_val = 0 if gender == "M" else 1
        meta = torch.tensor([[age, gender_val]], dtype=torch.float32).to(DEVICE)
        text_emb = torch.tensor(text_model.encode([report]), dtype=torch.float32).to(DEVICE)

        # Forward pass
        with torch.no_grad():
            logits = model(image_t, meta, text_emb)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # Grad-CAM
        heatmap = gradcam.generate(image_t, meta, text_emb, class_idx=pred_class)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # Load & scale ground truth bounding boxes
        matching_rows = df_bbox[df_bbox["Image Index"] == image_name]
        gt_bboxes = []
        if not matching_rows.empty:
            orig_w, orig_h = image.size
            target_w, target_h = 224, 224
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            for _, row in matching_rows.iterrows():
                x = float(row["Bbox x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
                x_min = int(x * scale_x)
                y_min = int(y * scale_y)
                x_max = int((x + w) * scale_x)
                y_max = int((y + h) * scale_y)
                gt_bboxes.append((x_min, y_min, x_max, y_max))

        # Overlay Grad-CAM and get Grad-CAM bounding boxes
        overlay_img, cam_bboxes = overlay_gradcam_with_bbox(image, heatmap_norm, gt_bboxes, threshold=threshold_val)

        # Compute IoU
        iou_scores = []
        if gt_bboxes:
            for gt_box in gt_bboxes:
                best_iou = 0
                for cam_box in cam_bboxes:
                    iou = compute_iou_boxes(gt_box, cam_box)
                    best_iou = max(best_iou, iou)
                iou_scores.append(best_iou)

        # Save results
        st.session_state["results"] = {
            "original": image,
            "gt_bboxes": gt_bboxes,
            "gt_only": visualize_gt_boxes(image, gt_bboxes) if gt_bboxes else None,
            "overlay": overlay_img,
            "pred_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "iou_scores": iou_scores
        }

# ---------------- Display results ----------------
if "results" in st.session_state:
    res = st.session_state["results"]

    st.subheader("Results")

    # --- Side by side comparison ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(res["original"], caption="Original X-ray", use_container_width=True)
    with col2:
        if res["gt_only"] is not None:
            st.image(res["gt_only"], caption="GT Bounding Boxes (Green)", use_container_width=True)
        else:
            st.warning("No ground truth bounding boxes found for this image.")
    with col3:
        st.image(res["overlay"], caption="Grad-CAM + BBoxes (Red=GradCAM, Green=GT)", use_container_width=True)

    # --- Prediction info ---
    st.markdown(f"**Prediction:** {res['pred_class']}  \n"
                f"**Confidence:** {res['confidence']:.3f}")

    # --- IoU Visualization ---
    if res["iou_scores"]:
        st.subheader("IoU Scores")
        cols = st.columns(len(res["iou_scores"]))
        for i, score in enumerate(res["iou_scores"]):
            color = "🟩" if score > 0.5 else ("🟨" if score > 0.2 else "🟥")
            with cols[i]:
                st.markdown(f"**GT Box {i+1}**")
                st.metric(label="IoU", value=f"{score:.2f}")
                st.markdown(color + (" High" if score > 0.5 else " Medium" if score > 0.2 else " Low"))
    else:
        st.warning("No ground truth bounding boxes found for this image.")

    # --- LLM Explanation of Grad-CAM ---
    st.subheader("🧠 AI Explanation of Grad-CAM")
    if st.button("Generate Explanation"):
        with st.spinner("Generating explanation..."):
            explanation = explain_gradcam(
                res["pred_class"],
                res["confidence"],
                res["iou_scores"],
                num_cam_regions=len(res["gt_bboxes"])
            )
        st.markdown(explanation)
