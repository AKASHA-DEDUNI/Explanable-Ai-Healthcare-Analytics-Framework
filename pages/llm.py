import streamlit as st
import openai
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os

from model import MultiModalPneumoniaModel
from sentence_transformers import SentenceTransformer
from utils import GradCAM

# --------------------- Config ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Pneumonia"]

# --------------------- OpenAI API Key ---------------------
# Load API key securely from Streamlit secrets
openai.api_key = st.secrets["openai"]["api_key"]

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
    bboxes = []
    heatmap_uint8 = np.uint8(heatmap * 255)
    _, heatmap_bin = cv2.threshold(heatmap_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    heatmap_clean = cv2.morphologyEx(heatmap_bin, cv2.MORPH_OPEN, kernel)
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
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
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

# --------------------- OpenAI LLM Chat ---------------------
def ask_llm(question, context=""):
    prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].text.strip()

# --------------------- Streamlit App ---------------------
st.title("📊 Pneumonia Grad-CAM + IoU + LLM Chat")

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
threshold_val = st.slider("Grad-CAM Threshold", 0.1, 0.9, 0.5, 0.05)

# ----------------- Run Prediction & Grad-CAM -----------------
if st.button("Run Prediction & Compute IoU"):
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
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        matching_rows = df_bbox[df_bbox["Image Index"] == image_name]
        gt_bboxes = []
        if not matching_rows.empty:
            orig_w, orig_h = image.size
            scale_x = 224 / orig_w
            scale_y = 224 / orig_h
            for _, row in matching_rows.iterrows():
                x_min = int(float(row["Bbox x"]) * scale_x)
                y_min = int(float(row["y"]) * scale_y)
                x_max = int((float(row["Bbox x"]) + float(row["w"])) * scale_x)
                y_max = int((float(row["y"]) + float(row["h"])) * scale_y)
                gt_bboxes.append((x_min, y_min, x_max, y_max))

        overlay_img, cam_bboxes = overlay_gradcam_with_bbox(image, heatmap_norm, gt_bboxes, threshold=threshold_val)

        iou_scores = []
        if gt_bboxes:
            for gt_box in gt_bboxes:
                best_iou = 0
                for cam_box in cam_bboxes:
                    best_iou = max(best_iou, compute_iou_boxes(gt_box, cam_box))
                iou_scores.append(best_iou)

        st.session_state["results"] = {
            "original": image,
            "gt_bboxes": gt_bboxes,
            "gt_only": visualize_gt_boxes(image, gt_bboxes) if gt_bboxes else None,
            "overlay": overlay_img,
            "pred_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "iou_scores": iou_scores,
            "report": report
        }

# ----------------- Display Results -----------------
if "results" in st.session_state:
    res = st.session_state["results"]

    st.subheader("Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(res["original"], caption="Original X-ray", use_container_width=True)
    with col2:
        if res["gt_only"] is not None:
            st.image(res["gt_only"], caption="GT Bounding Boxes (Green)", use_container_width=True)
    with col3:
        st.image(res["overlay"], caption="Grad-CAM + BBoxes (Red=GradCAM, Green=GT)", use_container_width=True)

    st.markdown(f"**Prediction:** {res['pred_class']}  \n**Confidence:** {res['confidence']:.3f}")

    if res["iou_scores"]:
        st.subheader("IoU Scores")
        for i, score in enumerate(res["iou_scores"]):
            color = "🟩" if score > 0.5 else ("🟨" if score > 0.2 else "🟥")
            st.metric(label=f"GT Box {i+1} IoU", value=f"{score:.2f}")
            st.markdown(color)

# ----------------- LLM Chat -----------------
st.subheader("💬 Ask about the X-ray")
question = st.text_input("Enter your question here:")

if st.button("Get LLM Explanation"):
    if "results" not in st.session_state:
        st.error("Please run prediction first!")
    elif question.strip() == "":
        st.error("Please enter a question!")
    else:
        context = f"Patient age: {age}, gender: {gender}\nReport: {st.session_state['results']['report']}\nPrediction: {st.session_state['results']['pred_class']}"
        answer = ask_llm(question, context)
        st.markdown(f"**Answer:** {answer}")
