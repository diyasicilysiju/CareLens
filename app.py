import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import tempfile
import firebase_admin
from firebase_admin import credentials, storage, firestore
from datetime import datetime
import json
import pyrebase

# ---------------------------
# 🌩 Firebase Config
# ---------------------------

firebase_config = json.loads(st.secrets["FIREBASE_KEY"])
firebase_admin_config = dict(st.secrets["FIREBASE_ADMIN"])

# Pyrebase (for Auth)
firebase = pyrebase.initialize_app({
    "apiKey": firebase_config["apiKey"],
    "authDomain": firebase_config["authDomain"],
    "projectId": firebase_config["project_id"],
    "storageBucket": firebase_config["storageBucket"],
    "messagingSenderId": firebase_config["messagingSenderId"],
    "appId": firebase_config["appId"],
    "databaseURL": firebase_config["databaseURL"],
})

auth = firebase.auth()

# Firebase Admin (for Firestore + Storage)
cred = credentials.Certificate(firebase_admin_config)
firebase_admin.initialize_app(cred, {
    "storageBucket": "medical-classifier.firebasestorage.app"
})


bucket = storage.bucket()
db = firestore.client()

# ---------------------------
# 🎨 Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="CareLens", page_icon="🩺", layout="centered")

# ---------------------------
# 🔐 Login / Signup
# ---------------------------
def login_page():
    st.title("🔐 CareLens Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user_email"] = email
            st.success("✅ Login successful!")
            st.experimental_rerun()
        except Exception as e:
            st.error("❌ Invalid credentials or user not found.")


def signup_page():
    st.title("🩺 Create an Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        try:
            auth.create_user_with_email_and_password(email, password)
            st.success("🎉 Account created! You can log in now.")
        except Exception as e:
            st.error("⚠ Error creating account.")

# ---------------------------
# 🧠 Load Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)

        blob = bucket.blob("model/model1.pth")
        with tempfile.NamedTemporaryFile(delete=False) as temp_model_file:
            blob.download_to_filename(temp_model_file.name)
        model.load_state_dict(torch.load(temp_model_file.name, map_location=torch.device("cpu")))
        model.eval()
        return model
    except Exception as e:
        st.error(f"⚠ Failed to load model: {e}")
        return None

# ---------------------------
# 🩻 Main App
# ---------------------------
def main_app():
    st.markdown("<h1 style='text-align:center;color:#004d99;'>CARELENS</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>AI-assisted chest X-ray analysis — simple, accurate, and human.</p>", unsafe_allow_html=True)

    model = load_model()
    uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview", use_container_width=True)

        if st.button("Analyze Image 🧠"):
            if model:
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
                    <div style='text-align:center;background:#f0f7ff;padding:20px;border-radius:10px;margin-top:20px;'>
                        <h3>Result</h3>
                        <h2 style='color:{color};'>{label}</h2>
                        <p>Analyzed on: {datetime.now().strftime("%d %b %Y, %I:%M %p")}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Save prediction to Firestore
                db.collection("predictions").add({
                    "email": st.session_state["user_email"],
                    "filename": uploaded_file.name,
                    "prediction": label,
                    "timestamp": datetime.now()
                })

                # Upload image to Storage
                blob = bucket.blob(f"user_uploads/{uploaded_file.name}")
                uploaded_file.seek(0)
                blob.upload_from_file(uploaded_file, content_type=uploaded_file.type)

                st.success("✅ Image and prediction logged successfully!")

# ---------------------------
# 🚪 Logout
# ---------------------------
def logout():
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------------
# 🧭 Navigation
# ---------------------------
if "user_email" not in st.session_state:
    page = st.sidebar.radio("Welcome to CareLens", ["Login", "Sign Up"])
    if page == "Login":
        login_page()
    else:
        signup_page()
else:
    logout()
    main_app()
