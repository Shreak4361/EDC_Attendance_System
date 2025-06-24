import cv2
import dlib
import numpy as np
import requests
from pymongo import MongoClient
from facenet_pytorch import MTCNN
import torch

# ✅ MongoDB Setup
client = MongoClient("mongodb://localhost:27017/")
db = client["attendance_system"]
users = db["users"]

# ✅ Face Detection with MTCNN (fast + reliable)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)

# ✅ Face Encoding Models from dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# ✅ Download image from URL
def download_image(url):
    try:
        response = requests.get(url)
        img_array = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

# ✅ Get face encoding from image using cropped face
def get_face_encoding(image):
    if image is None:
        return None
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is None or len(boxes) == 0:
        print("⚠️ No face detected.")
        return None

    # Take the first detected face
    x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
    face_crop = img_rgb[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]

    dets = face_detector(face_crop)
    if len(dets) == 0:
        print("⚠️ No Dlib face detected in cropped region.")
        return None

    shape = shape_predictor(face_crop, dets[0])
    encoding = face_encoder.compute_face_descriptor(face_crop, shape)
    return list(encoding)

# ✅ Compute average of encodings
def compute_average_encoding(encodings):
    if not encodings:
        return None
    return np.mean(np.array(encodings), axis=0).tolist()

# ✅ Update user encodings if not already stored or new photo is added
def update_user_encodings(email):
    user = users.find_one({"email": email})
    if not user:
        print("❌ User not found")
        return

    photo_urls = user.get("photo_links", [])
    existing_data = user.get("photo_data", [])
    
    if len(existing_data) == len(photo_urls):
        print("✅ No new photos found. Skipping update.")
        return

    photo_data = []
    encodings = []

    for url in photo_urls:
        img = download_image(url)
        enc = get_face_encoding(img)
        if enc:
            photo_data.append({"url": url, "encoding": enc})
            encodings.append(enc)

    avg_enc = compute_average_encoding(encodings)

    users.update_one(
        {"email": email},
        {"$set": {
            "photo_data": photo_data,
            "average_encoding": avg_enc
        }}
    )
    print(f"✅ Updated {email} with {len(encodings)} encodings.")

# ✅ Call function
try:
    update_user_encodings("2023csb107.yasharth@students.iiests.ac.in")
except Exception as e:
    print(f"❌ Error occurred: {e}")


