import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from model import MultiModalPneumoniaModel
from sentence_transformers import SentenceTransformer
from utils import GradCAM

# --------------------- Config ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Pneumonia"]

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

# --------------------- Grad-CAM Bounding Box Extraction (Improved) ---------------------
def get_gradcam_bbox(heatmap, threshold=0.5):
    bboxes = []
    heatmap_uint8 = np.uint8(heatmap * 255)

    # Binary mask
    _, heatmap_bin = cv2.threshold(heatmap_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # 1. Opening → remove small noise
    heatmap_clean = cv2.morphologyEx(heatmap_bin, cv2.MORPH_OPEN, kernel)

    # 2. Closing → fill small holes inside the regions
    heatmap_clean = cv2.morphologyEx(heatmap_clean, cv2.MORPH_CLOSE, kernel)

    # 3. Dilation → expand regions slightly to better cover lesions
    heatmap_clean = cv2.dilate(heatmap_clean, kernel, iterations=1)

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

    # draw GT boxes
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # draw Grad-CAM boxes
    cam_bboxes = get_gradcam_bbox(gradcam_resized, threshold)
    for bbox in cam_bboxes:
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    return overlay, cam_bboxes

# --------------------- Visualization Function ---------------------
def visualize_gt_boxes(image_pil, gt_bboxes):
    image_resized = image_pil.resize((224, 224))
    image_np = np.array(image_resized)
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_np

# --------------------- Threshold Sweep (Single Image) ---------------------
def sweep_thresholds(heatmap, gt_bboxes, image):
    top_percentile = np.percentile(heatmap, 95)  # stricter
    heatmap_norm = np.clip(heatmap / (top_percentile + 1e-8), 0, 1)

    results = {}
    thresholds = np.arange(0.05, 0.96, 0.01)

    for t in thresholds:
        overlay_img, cam_bboxes = overlay_gradcam_with_bbox(image, heatmap_norm, gt_bboxes, threshold=t)
        iou_scores = []
        if gt_bboxes:
            for gt_box in gt_bboxes:
                best_iou = max([compute_iou_boxes(gt_box, cam_box) for cam_box in cam_bboxes] or [0])
                iou_scores.append(best_iou)
        avg_iou = np.mean(iou_scores) if iou_scores else 0
        results[round(t, 2)] = {"overlay": overlay_img, "iou_scores": iou_scores, "avg_iou": avg_iou}

    best_threshold = max(results, key=lambda x: results[x]["avg_iou"])
    return results, best_threshold

# --------------------- Dataset-wide Threshold Evaluation (Improved) ---------------------
def evaluate_dataset_thresholds(dataset_folder, df_bbox, model, text_model, gradcam):
    thresholds = np.arange(0.05, 0.96, 0.01)
    all_best_ious = []

    image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for img_file in image_files:
        image_path = os.path.join(dataset_folder, img_file)
        image = Image.open(image_path).convert("RGB")
        image_name = img_file.lower().strip()

        # preprocess
        image_t = transform(image).unsqueeze(0).to(DEVICE)
        meta = torch.tensor([[30, 0]], dtype=torch.float32).to(DEVICE)   # dummy values if meta not available
        text_emb = torch.tensor(text_model.encode(["dummy report"]), dtype=torch.float32).to(DEVICE)

        # predict
        with torch.no_grad():
            logits = model(image_t, meta, text_emb)
            pred_class = torch.softmax(logits, dim=1).argmax(dim=1).item()

        # gradcam
        heatmap = gradcam.generate(image_t, meta, text_emb, class_idx=pred_class)
        top_percentile = np.percentile(heatmap, 95)   # stricter normalization
        heatmap_norm = np.clip(heatmap / (top_percentile + 1e-8), 0, 1)

        # ground truth bboxes
        matching_rows = df_bbox[df_bbox["Image Index"] == image_name]
        gt_bboxes = []
        if not matching_rows.empty:
            orig_w, orig_h = image.size
            scale_x, scale_y = 224 / orig_w, 224 / orig_h
            for _, row in matching_rows.iterrows():
                x_min = int(float(row["Bbox x"]) * scale_x)
                y_min = int(float(row["y"]) * scale_y)
                x_max = int((float(row["Bbox x"]) + float(row["w"])) * scale_x)
                y_max = int((float(row["y"]) + float(row["h"])) * scale_y)
                gt_bboxes.append((x_min, y_min, x_max, y_max))

        if not gt_bboxes:
            continue   # skip images with no GT

        # best IoU search per image
        best_iou = 0
        for t in thresholds:
            _, cam_bboxes = overlay_gradcam_with_bbox(image, heatmap_norm, gt_bboxes, threshold=t)
            iou_scores = []
            for gt_box in gt_bboxes:
                best_match = max([compute_iou_boxes(gt_box, cam_box) for cam_box in cam_bboxes] or [0])
                iou_scores.append(best_match)
            avg_iou = np.mean(iou_scores) if iou_scores else 0
            best_iou = max(best_iou, avg_iou)

        all_best_ious.append(best_iou)

    mean_iou = np.mean(all_best_ious) if all_best_ious else 0
    return mean_iou

# --------------------- Streamlit App ---------------------
st.title("📊 Pneumonia Grad-CAM Bounding Box & IoU Optimizer")

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

# --------- Single Image Evaluation ---------
st.header("🔍 Single Image Evaluation")
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Patient Gender", options=["M", "F"])
report = st.text_area("Radiology Report Text")

if st.button("Run Prediction & Compute IoU (Single Image)"):
    if uploaded_file is None or report.strip() == "":
        st.error("Please upload image and enter report text.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_name = os.path.basename(uploaded_file.name).lower().strip()

        image_t = transform(image).unsqueeze(0).to(DEVICE)
        gender_val = 0 if gender == "M" else 1
        meta = torch.tensor([[age, gender_val]], dtype=torch.float32).to(DEVICE)
        text_emb = torch.tensor(text_model.encode([report]), dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = model(image_t, meta, text_emb)
            probs = torch.softmax(logits, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        heatmap = gradcam.generate(image_t, meta, text_emb, class_idx=pred_class)

        matching_rows = df_bbox[df_bbox["Image Index"] == image_name]
        gt_bboxes = []
        if not matching_rows.empty:
            orig_w, orig_h = image.size
            scale_x, scale_y = 224 / orig_w, 224 / orig_h
            for _, row in matching_rows.iterrows():
                x_min = int(float(row["Bbox x"]) * scale_x)
                y_min = int(float(row["y"]) * scale_y)
                x_max = int((float(row["Bbox x"]) + float(row["w"])) * scale_x)
                y_max = int((float(row["y"]) + float(row["h"])) * scale_y)
                gt_bboxes.append((x_min, y_min, x_max, y_max))

        sweep_results, best_t = sweep_thresholds(heatmap, gt_bboxes, image)
        best_result = sweep_results[best_t]

        st.subheader("Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original X-ray", use_container_width=True)
        with col2:
            if gt_bboxes:
                st.image(visualize_gt_boxes(image, gt_bboxes), caption="GT Boxes", use_container_width=True)
            else:
                st.warning("No GT boxes found.")
        with col3:
            st.image(best_result["overlay"], caption=f"Best Threshold {best_t}", use_container_width=True)

        st.markdown(f"**Prediction:** {CLASS_NAMES[pred_class]}  \n"
                    f"**Confidence:** {confidence:.3f}")

        if best_result["iou_scores"]:
            st.subheader("IoU Scores")
            for i, score in enumerate(best_result["iou_scores"]):
                st.metric(label=f"GT Box {i+1}", value=f"{score:.2f}")
            st.success(f"✅ Best Threshold: {best_t} | Avg IoU: {best_result['avg_iou']:.3f}")

# --------- Dataset Evaluation ---------
st.header("📂 Dataset-wide Evaluation")
dataset_folder = st.text_input("Dataset Folder Path", r"C:\Users\ASUS\Desktop\resnet50\images")

if st.button("Evaluate Dataset Thresholds"):
    if not os.path.exists(dataset_folder):
        st.error("Dataset folder not found.")
    else:
        mean_iou = evaluate_dataset_thresholds(dataset_folder, df_bbox, model, text_model, gradcam)
        st.success(f"✅ Dataset Mean IoU (best threshold per image): {mean_iou:.3f}")

        if mean_iou >= 0.85:
            st.balloons()
            st.info("🎉 Congratulations! You reached the 0.85 IoU target.")

