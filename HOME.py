import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from model import MultiModalPneumoniaModel
from sentence_transformers import SentenceTransformer
from utils import GradCAM, apply_heatmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = MultiModalPneumoniaModel().to(DEVICE)
    model.load_state_dict(torch.load(
        r'C:\Users\ASUS\Desktop\resnet50\best_multimodal_model_resnet50.pth', map_location=DEVICE
    ))
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_text_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
text_model = load_text_model()
gradcam = GradCAM(model, model.cnn.layer4[-1].conv3)

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="🩺 Pneumonia Diagnosis Dashboard", layout="wide")
st.title("🩺 Multimodal Pneumonia Diagnosis Dashboard")

with st.expander("👆 Upload Patient Data"):
    uploaded_file = st.file_uploader("📷 Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
    age = st.number_input("🎂 Patient Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("⚥ Patient Gender", options=["M", "F"])
    report = st.text_area("📝 Radiology Report Text", placeholder="Enter radiology report here...")

# -------------------------------
# Initialize Session State
# -------------------------------
for key in ['pred_class', 'confidence', 'heatmap_img_bytes', 'shap_values', 'meta_features', 'attention_tokens', 'attention_weights']:
    if key not in st.session_state:
        st.session_state[key] = None

# -------------------------------
# Inference & Explainability
# -------------------------------
def run_inference():
    # Image preprocessing
    image = Image.open(uploaded_file).convert('RGB')
    image_t = transform(image).unsqueeze(0).to(DEVICE)

    # Metadata preprocessing
    gender_val = 0 if gender == "M" else 1
    meta = torch.tensor([[age, gender_val]], dtype=torch.float32).to(DEVICE)

    # Text embeddings
    text_emb_np = text_model.encode([report])
    text_emb = torch.tensor(text_emb_np, dtype=torch.float32).to(DEVICE)

    # Model prediction
    with torch.no_grad():
        logits = model(image_t, meta, text_emb)
        probs = torch.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

    # Grad-CAM
    heatmap = gradcam.generate(image_t, meta, text_emb, class_idx=pred_class)
    heatmap_img = apply_heatmap(image, heatmap)
    buf = io.BytesIO()
    heatmap_img.save(buf, format="PNG")
    buf.seek(0)

    # SHAP for metadata
    def model_predict(meta_np):
        meta_tensor = torch.tensor(meta_np, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(image_t, meta_tensor, text_emb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    explainer = shap.KernelExplainer(model_predict, np.array([[age, gender_val]]))
    shap_values = explainer.shap_values(np.array([[age, gender_val]]))

    # Attention for text
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    att_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', output_attentions=True)
    inputs = tokenizer(report, return_tensors='pt')
    outputs = att_model(**inputs)
    attentions = outputs.attentions
    attn_last = attentions[-1].squeeze(0).mean(dim=0)
    cls_attention = attn_last[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))

    # Save results to session_state
    st.session_state['pred_class'] = pred_class
    st.session_state['confidence'] = confidence
    st.session_state['heatmap_img_bytes'] = buf.getvalue()
    st.session_state['shap_values'] = shap_values
    st.session_state['meta_features'] = np.array([[age, gender_val]])
    st.session_state['attention_tokens'] = tokens
    st.session_state['attention_weights'] = cls_attention.detach().numpy()

# -------------------------------
# Run Button
# -------------------------------
if st.button("🔍 Predict and Explain"):
    if uploaded_file is None:
        st.error("⚠️ Please upload an image file.")
    elif report.strip() == "":
        st.error("⚠️ Please enter the radiology report text.")
    else:
        run_inference()

# -------------------------------
# Display Results
# -------------------------------
if st.session_state['pred_class'] is not None:
    st.markdown(f"### 🏷 Prediction: {'🦠 Pneumonia' if st.session_state['pred_class'] == 1 else '✅ Normal'}")
    st.markdown(f"### 💯 Confidence: {st.session_state['confidence']:.3f}")

    # Optional visualizations
    with st.expander("🔥 Grad-CAM Heatmap"):
        if st.session_state['heatmap_img_bytes'] is not None:
            heatmap_img = Image.open(io.BytesIO(st.session_state['heatmap_img_bytes']))
            st.image(heatmap_img, use_column_width=True)

    with st.expander("📊 Metadata Feature Importance (SHAP)"):
        shap_vals = st.session_state['shap_values']
        meta_feats = st.session_state['meta_features']
        for i, feature in enumerate(["Age", "Gender"]):
            st.bar_chart([shap_vals[0][0][i]])

    with st.expander("📝 Text Report Attention Visualization"):
        if st.session_state['attention_tokens'] is not None:
            plt.figure(figsize=(10,4))
            sns.barplot(
                x=st.session_state['attention_weights'],
                y=st.session_state['attention_tokens'],
                palette="viridis"
            )
            plt.xlabel("Attention Weight")
            plt.ylabel("Token")
            st.pyplot(plt.gcf())
            plt.clf()
