import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import tempfile
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime

# ---------------------------
# 🔹 Firebase Configuration
# ---------------------------
FIREBASE_BUCKET = "medical-classifier.firebasestorage.app"
cred = credentials.Certificate("firebase_key.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET})

bucket = storage.bucket()

# ---------------------------
# 🎨 Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="CareLens - X-ray Classifier", page_icon="🩺", layout="centered")

# Subtle styling
st.markdown("""
    <style>
        h1 {
            color: #004d99;
            text-align: center;
            font-family: 'Helvetica Neue', sans-serif;
        }
        .subtitle {
            text-align: center;
            color: #555;
            margin-bottom: 2rem;
            font-size: 1rem;
        }
        .result {
            text-align: center;
            background-color: #f0f7ff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            border: 1px solid #cce0ff;
        }
        .footer {
            text-align: center;
            margin-top: 3rem;
            color: #777;
            font-size: 0.9em;
        }
        .stButton>button {
            background-color: #007acc;
            color: white;
            border-radius: 6px;
            padding: 0.5em 1.5em;
        }
        .stButton>button:hover {
            background-color: #005fa3;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# 🩺 Header
# ---------------------------
st.markdown("<h1>CARELENS</h1>", unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-assisted chest X-ray analysis — simple, accurate, and human</div>', unsafe_allow_html=True)

# ---------------------------
# 🔹 Model Loading
# ---------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 2)

    blob = bucket.blob("model/model1.pth")
    with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
        blob.download_to_filename(temp_model_file.name)
        model_path = temp_model_file.name

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# ---------------------------
# 🩻 Upload + Predict
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Preview", use_container_width=True)

    if st.button("Analyze Image 🧠"):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, pred = torch.max(output, 1)

        label = "Normal" if pred.item() == 0 else "Pneumonia"
        color = "#007acc" if label == "Normal" else "#e53935"

        st.markdown(f"""
            <div class="result">
                <h3>Result</h3>
                <h2 style="color:{color};">{label}</h2>
                <p>Analyzed on: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</p>
            </div>
        """, unsafe_allow_html=True)

        try:
            blob_path = f"user_uploads/{uploaded_file.name}"
            blob = bucket.blob(blob_path)
            uploaded_file.seek(0)
            blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)
            st.success("✅ Image securely saved to cloud storage.")
        except Exception as e:
            st.error(f"⚠️ Upload failed: {e}")

# ---------------------------
# ⚕️ Footer
# ---------------------------
st.markdown('<div class="footer">© 2025 CareLens • Empowering medical clarity through AI</div>', unsafe_allow_html=True)
