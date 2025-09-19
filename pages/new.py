import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import io

from model import MultiModalPneumoniaModel
from sentence_transformers import SentenceTransformer
from utils import GradCAM

# --------------------- Config ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Normal", "Pneumonia"]
IMG_SIZE = (224, 224)

# --------------------- Load Models ---------------------
@st.cache_resource
def load_model():
    model = MultiModalPneumoniaModel().to(DEVICE)
    model.load_state_dict(torch.load(
        r"C:\\Users\\ASUS\\Desktop\\resnet50\\best_multimodal_model_resnet50.pth",
        map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_text_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --------------------- Load Bounding Boxes ---------------------
@st.cache_data
def load_bbox_data():
    df = pd.read_csv(r"C:\\Users\\ASUS\\Desktop\\resnet50\\BBox_List_2017.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Bbox [x': 'Bbox x', 'h]': 'h'})
    df['Image Index'] = df['Image Index'].str.strip().str.lower()
    return df

# --------------------- Helper: IoU ---------------------
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

# --------------------- Grad-CAM BBox Extraction ---------------------
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

# --------------------- Visual Helpers ---------------------
def overlay_gradcam_with_bbox(image_pil, gradcam_map, gt_bboxes, alpha=0.5, threshold=0.5):
    image_np = np.array(image_pil.resize(IMG_SIZE))
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


def visualize_gt_boxes(image_pil, gt_bboxes):
    image_resized = image_pil.resize(IMG_SIZE)
    image_np = np.array(image_resized)
    for bbox in gt_bboxes:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return image_np

# --------------------- Counterfactuals ---------------------

def mask_top_k_regions(image_pil, gradcam_map, k=1, threshold=0.5, fill_mode="blur"):
    """
    Create a counterfactual image by masking the top-k Grad-CAM regions.
    fill_mode: 'blur' | 'gray'
    """
    img = np.array(image_pil.resize(IMG_SIZE))
    cam = cv2.resize(gradcam_map, IMG_SIZE)

    # threshold + connected components to get regions
    cam_uint8 = np.uint8(cam * 255)
    _, binmap = cv2.threshold(cam_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(binmap)

    # score each component by average CAM value
    regions = []
    for lbl in range(1, num_labels):
        mask = (labels_im == lbl).astype(np.uint8)
        score = (cam * mask).sum() / (mask.sum() + 1e-6)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        regions.append(((x0, y0, x1, y1), score))

    # pick top-k by score
    regions = sorted(regions, key=lambda z: z[1], reverse=True)[:max(1, k)]

    img_cf = img.copy()
    for (x0, y0, x1, y1), _ in regions:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        roi = img_cf[y0:y1+1, x0:x1+1]
        if roi.size == 0:
            continue
        if fill_mode == "blur":
            roi_blur = cv2.GaussianBlur(roi, (21, 21), 0)
            img_cf[y0:y1+1, x0:x1+1] = roi_blur
        else:
            gray_val = int(img.mean())
            img_cf[y0:y1+1, x0:x1+1] = gray_val
    return Image.fromarray(img_cf)


def what_if_meta(age, gender_val, step=5):
    """Generate counterfactual meta variants: age ± step, gender flipped."""
    variants = []
    variants.append((max(0, age - step), gender_val))
    variants.append((age, 1 - gender_val))
    variants.append((age + step, gender_val))
    return variants


def simple_token_edits(report_text, important_terms, max_edits=3):
    """Return simple counterfactual text variants by removing or replacing important terms."""
    variants = []
    toks = report_text.split()
    # remove up to max_edits important tokens
    removed = [w for w in toks if w.lower() not in important_terms][:]
    variants.append(" ".join(removed))
    # replace top terms with neutral synonyms (very naive)
    repl = []
    mapping = {
        'consolidation': 'opacity',
        'infiltrate': 'finding',
        'pneumonia': 'infection',
        'pleural': 'chest',
        'effusion': 'fluid'
    }
    for w in toks:
        key = w.lower().strip(",.()")
        repl.append(mapping.get(key, w))
    variants.append(" ".join(repl))
    return variants[:max_edits]

# --------------------- NL Explanation (Rule-based) ---------------------

def make_natural_explanation(pred_label, confidence, iou_scores, meta, deltas):
    parts = []
    parts.append(f"The model predicts **{pred_label}** with confidence {confidence:.2f}.")
    if iou_scores is not None and len(iou_scores) > 0:
        avg_iou = float(np.mean(iou_scores))
        parts.append(f"Grad-CAM overlaps with ground-truth regions (avg IoU ≈ {avg_iou:.2f}).")
    age, gender_val = meta
    gtxt = 'Male' if gender_val == 0 else 'Female'
    parts.append(f"Patient metadata considered: age {age}, gender {gtxt}.")
    # deltas is a dict like {"cf_img":delta, "cf_meta":[(desc,delta),...], "cf_text":[(desc,delta),...]}
    if deltas.get("cf_img") is not None:
        d = deltas["cf_img"]
        parts.append(f"Masking the top highlighted image regions changed the probability by {d:+.2f}.")
    if deltas.get("cf_meta"):
        for desc, d in deltas["cf_meta"]:
            parts.append(f"If {desc}, probability would change by {d:+.2f}.")
    if deltas.get("cf_text"):
        for desc, d in deltas["cf_text"]:
            parts.append(f"With text edit ({desc}), probability changes by {d:+.2f}.")
    return " \n".join(parts)

# --------------------- Streamlit App ---------------------
st.set_page_config(page_title="Explainable Multimodal Pneumonia - Counterfactuals", layout="wide")
st.title("🩺 Explainable Multimodal Pneumonia: Counterfactuals & What‑Ifs")

model = load_model()
text_model = load_text_model()
gradcam = GradCAM(model, model.cnn.layer4[-1].conv2)
df_bbox = load_bbox_data()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------- Sidebar Controls ----------
with st.sidebar:
    st.header("Controls")
    threshold_val = st.slider("Grad-CAM Threshold", 0.1, 0.9, 0.5, 0.05)
    alpha_val = st.slider("Heatmap Opacity", 0.1, 0.9, 0.5, 0.05)
    k_regions = st.slider("Top-K regions to mask (CF)", 1, 5, 1)
    fill_mode = st.selectbox("CF fill mode", ["blur", "gray"])

# ---------- Inputs ----------
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["png", "jpg", "jpeg"])
age = st.number_input("Patient Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Patient Gender", options=["M", "F"])  # 0=M,1=F
default_report = "Chest X-ray shows mild consolidation in the right lower lobe consistent with pneumonia. No pleural effusion."
report = st.text_area("Radiology Report Text", value=default_report, height=120)

col_run1, col_run2 = st.columns([1,1])
run_clicked = col_run1.button("Run Prediction & Explanations", type="primary")
reset_clicked = col_run2.button("Reset Session")
if reset_clicked:
    st.session_state.clear()

if run_clicked:
    if uploaded_file is None or report.strip() == "":
        st.error("Please upload an image and enter report text.")
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

        # Grad-CAM
        heatmap = gradcam.generate(image_t, meta, text_emb, class_idx=pred_class)
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        # GT boxes
        matching_rows = df_bbox[df_bbox["Image Index"] == image_name]
        gt_bboxes = []
        if not matching_rows.empty:
            orig_w, orig_h = image.size
            scale_x = IMG_SIZE[0] / orig_w
            scale_y = IMG_SIZE[1] / orig_h
            for _, row in matching_rows.iterrows():
                x = float(row["Bbox x"]); y = float(row["y"]) ; w = float(row["w"]); h = float(row["h"])
                x_min = int(x * scale_x); y_min = int(y * scale_y)
                x_max = int((x + w) * scale_x); y_max = int((y + h) * scale_y)
                gt_bboxes.append((x_min, y_min, x_max, y_max))

        overlay_img, cam_bboxes = overlay_gradcam_with_bbox(image, heatmap_norm, gt_bboxes, alpha=alpha_val, threshold=threshold_val)

        # IoU per GT box
        iou_scores = []
        if gt_bboxes:
            for gt_box in gt_bboxes:
                best_iou = 0
                for cam_box in cam_bboxes:
                    iou = compute_iou_boxes(gt_box, cam_box)
                    best_iou = max(best_iou, iou)
                iou_scores.append(best_iou)

        # ---------- Counterfactuals ----------
        deltas = {"cf_img": None, "cf_meta": [], "cf_text": []}

        # (A) Image CF: mask top-k CAM regions and recompute
        img_cf = mask_top_k_regions(image, heatmap_norm, k=k_regions, threshold=threshold_val, fill_mode=fill_mode)
        img_cf_t = transform(img_cf).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits_cf = model(img_cf_t, meta, text_emb)
            p_cf = torch.softmax(logits_cf, dim=1)[0, pred_class].item()
        delta_img = p_cf - confidence
        deltas["cf_img"] = delta_img

        # (B) Metadata CFs: age ± step, gender flip
        for a2, g2 in what_if_meta(age, gender_val, step=5):
            meta2 = torch.tensor([[a2, g2]], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p2 = torch.softmax(model(image_t, meta2, text_emb), dim=1)[0, pred_class].item()
            desc = f"age {a2} & gender {'M' if g2==0 else 'F'}"
            deltas["cf_meta"].append((desc, p2 - confidence))

        # (C) Text CFs: simple token removals/replacements on important terms
        important_terms = {"consolidation", "infiltrate", "pneumonia", "effusion", "opacity", "lobar"}
        for i, rep_text in enumerate(simple_token_edits(report, important_terms, max_edits=2)):
            text_emb2 = torch.tensor(text_model.encode([rep_text]), dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p2 = torch.softmax(model(image_t, meta, text_emb2), dim=1)[0, pred_class].item()
            desc = f"edit {i+1}"
            deltas["cf_text"].append((desc, p2 - confidence))

        # Save results in session
        st.session_state["results"] = {
            "original": image,
            "gt_bboxes": gt_bboxes,
            "gt_only": visualize_gt_boxes(image, gt_bboxes) if gt_bboxes else None,
            "overlay": overlay_img,
            "pred_class": CLASS_NAMES[pred_class],
            "confidence": confidence,
            "iou_scores": iou_scores,
            "img_cf": img_cf,
            "deltas": deltas,
        }

# ---------------- Display ----------------
if "results" in st.session_state:
    res = st.session_state["results"]

    tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explanations", "Counterfactuals", "Report"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(res["original"], caption="Original X-ray", use_container_width=True)
        with col2:
            if res["gt_only"] is not None:
                st.image(res["gt_only"], caption="GT BBoxes (Green)", use_container_width=True)
            else:
                st.warning("No ground truth bounding boxes found for this image.")
        with col3:
            st.image(res["overlay"], caption="Grad-CAM + Boxes (Red=CAM, Green=GT)", use_container_width=True)

        st.markdown(f"**Prediction:** {res['pred_class']}  \\n**Confidence:** {res['confidence']:.3f}")
        if res["iou_scores"]:
            st.subheader("IoU Scores")
            cols = st.columns(len(res["iou_scores"]))
            for i, score in enumerate(res["iou_scores"]):
                color = "🟩" if score > 0.5 else ("🟨" if score > 0.2 else "🟥")
                with cols[i]:
                    st.markdown(f"**GT Box {i+1}**")
                    st.metric(label="IoU", value=f"{score:.2f}")
                    st.markdown(color + (" High" if score > 0.5 else " Medium" if score > 0.2 else " Low"))

    with tab2:
        st.markdown("### Visual Explanation")
        st.caption("Adjust threshold/opacity from the sidebar to explore Grad-CAM")
        st.image(res["overlay"], use_container_width=True)

    with tab3:
        st.markdown("### Counterfactual Analysis")
        cf_cols = st.columns(2)
        with cf_cols[0]:
            st.image(res["img_cf"], caption=f"Counterfactual (masked top-{k_regions})", use_container_width=True)
        with cf_cols[1]:
            st.metric("Δ Prob (mask regions)", f"{res['deltas']['cf_img']:+.3f}")

        st.divider()
        st.markdown("#### Metadata What‑Ifs")
        for desc, d in res['deltas']['cf_meta']:
            st.write(f"• If **{desc}**, Δ prob = {d:+.3f}")

        st.markdown("#### Text What‑Ifs")
        for desc, d in res['deltas']['cf_text']:
            st.write(f"• With **{desc}**, Δ prob = {d:+.3f}")

    with tab4:
        st.markdown("### Natural-Language Summary")
        # Use rule-based generator (safe offline). You can replace with an LLM call if available.
        gender_val = 0 if gender == "M" else 1
        nl = make_natural_explanation(
            res['pred_class'], res['confidence'], res['iou_scores'], (age, gender_val), res['deltas']
        )
        st.markdown(nl)

        # Export simple text report
        if st.button("Download Summary (.txt)"):
            buf = io.StringIO()
            buf.write("Explainable Multimodal Pneumonia Report\n\n")
            buf.write(nl.replace("**", ""))
            st.download_button(
                label="Save report",
                data=buf.getvalue().encode('utf-8'),
                file_name="explainable_report.txt",
                mime="text/plain",
            )
