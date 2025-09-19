import streamlit as st
import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

# --------------------- Config ---------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is not set in .env file")
    st.stop()

# --------------------- Helper ---------------------
def query_xray(image: Image.Image, query: str, model="llama-3.2-90b-vision-preview"):
    try:
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        encoded_image = base64.b64encode(buf.getvalue()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
                    }
                ]
            }
        ]

        response = requests.post(
            GROQ_API_URL,
            json={"model": model, "messages": messages, "max_tokens": 800},
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            # Extract text chunks only
            answer = "".join(
                c["text"] for c in content if c["type"] == "text"
            ) if isinstance(content, list) else str(content)
            return answer
        else:
            return f"❌ Error {response.status_code}: {response.text}"

    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# --------------------- Streamlit UI ---------------------
st.set_page_config(page_title="X-ray Chatbot", page_icon="🩻", layout="wide")
st.title("🩻 X-ray Chat Assistant")
st.write("Upload a chest X-ray and ask questions. The model will explain findings.")

# Sidebar
st.sidebar.header("Options")
model_choice = st.sidebar.selectbox(
    "Choose Groq Vision Model:",
    ["meta-llama/llama-4-scout-17b-16e-instruct"]
)

# Session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload X-ray
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-ray", use_column_width=True)

        # Chat input
        query = st.chat_input("Ask something about this X-ray...")

        if query:
            with st.spinner("Analyzing X-ray..."):
                answer = query_xray(image, query, model=model_choice)

            # Save to chat history
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", answer))

    except Exception as e:
        st.error(f"Error opening image: {e}")

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
