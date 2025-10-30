import streamlit as st
import pyrebase

# Directly use FIREBASE_KEY from secrets
firebase_config = st.secrets["FIREBASE_KEY"]

# Initialize Firebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

st.title("Firebase Login")

email = st.text_input("Email")
password = st.text_input("Password", type="password")

if st.button("Login"):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.success("Login successful!")
    except Exception as e:
        st.error(f"Error: {e}")
