from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import os

load_dotenv()

# Cloudinary config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# MongoDB config
client = MongoClient(os.getenv("MONGO_URI"))
db = client["attendance_system"]
users = db["users"]

app = Flask(__name__)
CORS(app)

@app.route("/register", methods=["POST"])
def register_user():
    name = request.form.get("name")
    email = request.form.get("email")
    phone = request.form.get("phone")
    files = request.files.getlist("photos")

    photo_urls = []
    for file in files:
        result = cloudinary.uploader.upload(file)
        photo_urls.append(result["secure_url"])

    existing = users.find_one({"email": email})
    if existing:
        users.update_one(
            {"email": email},
            {"$push": {"photo_links": {"$each": photo_urls}}}
        )
        return jsonify({"message": "Photos added to existing user!"})

    doc = {
        "name": name,
        "email": email,
        "phone": phone,
        "photo_links": photo_urls,
        "face_encodings": [],
        "attendance": {
            "days_present": 0,
            "present_dates": []
        }
    }
    users.insert_one(doc)
    return jsonify({"message": "User registered successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
