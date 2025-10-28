import firebase_admin
from firebase_admin import credentials, storage

# Step 1: Use your service account key (JSON file)
cred = credentials.Certificate("firebase_key.json")

# ✅ Step 2: Initialize app only if not already done
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'medical-classifier.firebasestorage.app'
    })

# Step 3: Get a reference to your bucket
bucket = storage.bucket()
