# main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import update_user_encodings, update_all_users_encodings, recognize_faces_from_image
from io import BytesIO

app = FastAPI()

@app.get("/")
def welcome():
    return {"message": "✅ Face Recognition Attendance API is live!"}

@app.get("/update-user/")
def update_user(email: str):
    result = update_user_encodings(email)
    return {"result": result}

@app.get("/update-all/")
def update_all():
    results = update_all_users_encodings()
    return {"updated": results}

@app.post("/mark-attendance/")
async def mark_attendance(file: UploadFile = File(...)):
    image_bytes = BytesIO(await file.read())
    result = recognize_faces_from_image(image_bytes)

    return JSONResponse({
        "total_faces_detected": result["total_faces"],
        "matched_faces": result["matched_names"],
        "unmatched_faces_count": result["total_faces"] - result["matched_count"],
        "attendance_marked_for": result["matched_names"]
    })
