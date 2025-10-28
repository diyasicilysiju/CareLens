from firebase_config import bucket
import tempfile

# create a small text file
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
    f.write(b"Hello Firebase!")
    local_path = f.name

# upload to Firebase Storage
blob = bucket.blob("test/test_upload.txt")
blob.upload_from_filename(local_path)
blob.make_public()

print("✅ File uploaded successfully!")

print("URL:", blob.public_url)
