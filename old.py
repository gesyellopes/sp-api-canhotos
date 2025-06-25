
import os
import cv2
import numpy as np
import tempfile
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image
from cv2 import QRCodeDetector

load_dotenv()
app = FastAPI()

TEMPLATE_PATH = "template.jpg"
STORAGE_URL = os.getenv("STORAGE_URL")
API_KEY = os.getenv("X_API_KEY")

def align_and_crop_image(uploaded_image_path, template_image_path):
    try:
        # Read images
        img = cv2.imread(uploaded_image_path)
        template = cv2.imread(template_image_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Feature detection
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img_gray, None)
        kp2, des2 = orb.detectAndCompute(template_gray, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 4:
            return None, "Pontos insuficientes para alinhamento"

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:10]]).reshape(-1, 1, 2)

        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        aligned = cv2.warpPerspective(img, M, (template.shape[1], template.shape[0]))

        return aligned, None
    except Exception as e:
        return None, str(e)

def read_qr_code(image_np):
    qr = QRCodeDetector()
    val, _, _ = qr.detectAndDecode(image_np)
    return val if val else None

def send_to_storage(image_path, qr_code):
    try:
        with open(image_path, "rb") as f:
            response = requests.post(
                os.getenv("STORAGE_URL"),
                headers={"x-api-key": os.getenv("X_API_KEY")},
                files={"file": (f"{qr_code[-6:]}.jpg", f)}
            )
            if response.ok:
                return response.json().get("file_id")
            return None
    except Exception:
        return None


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    try:
        # Save temp image
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Align and crop
        aligned_img, err = align_and_crop_image(tmp_path, TEMPLATE_PATH)
        if aligned_img is None:
            os.remove(tmp_path)
            return JSONResponse(status_code=200, content={"success": False, "message": err})

        # Read QR
        qr_code = read_qr_code(aligned_img)
        if not qr_code:
            os.remove(tmp_path)
            return JSONResponse(status_code=200, content={"success": False, "message": "QR Code não identificado"})

        # Save aligned image temporarily
        filename = f"{qr_code[-6:]}.jpg"
        final_path = os.path.join(tempfile.gettempdir(), filename)

        cv2.imwrite(final_path, aligned_img)

        # Upload to cloud
        file_id = send_to_storage(final_path, qr_code)
        if not file_id:
            os.remove(tmp_path)
            os.remove(final_path)
            return JSONResponse(status_code=200, content={"success": False, "message": "Erro ao enviar para storage"})

        # Cleanup
        os.remove(tmp_path)
        os.remove(final_path)

        return {
            "success": True,
            "message": "Canhoto salvo com sucesso",
            "filename": qr_code[-6:],
            "file_id": file_id
        }


    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": f"Erro interno: {str(e)}"})
