from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from io import BytesIO
import base64
from PIL import Image
from ultralytics import YOLO
import uuid  # เพิ่มการใช้งาน uuid
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# เสิร์ฟไฟล์ในโฟลเดอร์ static
app.mount("/static", StaticFiles(directory="static"), name="static")

# โหลดโมเดล YOLO
model = YOLO("models/heart.pt")

# class names ของยา
class_names = {
    0:  'Amitriptyline Hydrochloride Tablets 10 mg', 
    1:  'Amiodarone Tablets 200 mg', 
    2:  'Amlodipine Tablets 10 mg', 
    3:  'Amlodipine Tablets 5 mg', 
    4:  'Atenolol 100 mg', 
    5:  'Atenolol 25 mg', 
    6:  'Atenolol 50 mg', 
    7:  'Bisoprolol Fumarate 5 mg', 
    8:  'Carvedilol 25 mg',
    9:  'Carvedilol 6.25 mg',
    10: 'Clopidogrel 75 mg', 
    11: 'Digoxin 0.25 mg', 
    12: 'Enalapril Maleate 5 mg',
    13: 'Furosemide 40 mg',
    14: 'Hydralazine HCl 25 mg',
    15: 'sosorbide Dinitrate BP 10 mg', 
    16: 'Isosorbide Dinitrate BP 5 mg', 
    17: 'Isosorbide Dinitrate Tablets 10 mg',
    18: 'Lercanidipine HCl 20 mg', 
    19: 'Losartan Potassium 50 mg',
    20: 'Manidipine Hydrochloride 20 mg', 
    21: 'Spironolactone 25 mg',
    22: 'Propranolol 10 mg', 
    23: 'Warfarin Sodium 2 mg', 
    24: 'Warfarin Sodium 3 mg',
    25: 'Warfarin Sodium 5 mg'
}

# คำแนะนำการใช้ยาและข้อควรระวัง
instructions = {
    "Amitriptyline Hydrochloride Tablets 10 mg": ".",
    "Amiodarone Tablets 200 mg": ".",
    "Amlodipine Tablets 10 mg": ".",
    "Amlodipine Tablets 5 mg": ".",
    "Atenolol 100 mg": ".",
    "Atenolol 25 mg": ".",
    "Atenolol 50 mg": ".",
    "Bisoprolol Fumarate 5 mg": ".",
    "Carvedilol 25 mg": ".",
    "Carvedilol 6.25 mg": ".",
    "Clopidogrel 75 mg": ".",
    "Digoxin 0.25 mg": ".",
    "Enalapril Maleate 5 mg": ".",
    "Furosemide 40 mg": ".",
    "Hydralazine HCl 25 mg": ".",
    "Isosorbide Dinitrate BP 10 mg": ".",
    "Isosorbide Dinitrate BP 5 mg": ".",
    "Isosorbide Dinitrate Tablets 10 mg": ".",
    "Lercanidipine HCl 20 mg": ".",
    "Losartan Potassium 50 mg": ".",
    "Manidipine Hydrochloride 20 mg": ".",
    "Spironolactone 25 mg": ".",
    "Propranolol 10 mg": ".",
    "Warfarin Sodium 2 mg": ".",
    "Warfarin Sodium 3 mg": ".",
    "Warfarin Sodium 5 mg": "."
}

# สร้างตัวจัดการเทมเพลต
templates = Jinja2Templates(directory="templates")

def convert_image_to_base64(image: Image) -> str:
    """แปลงภาพเป็น Base64"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
async def main(request: Request):
    return templates.TemplateResponse("heart.html", {"request": request})

@app.post("/upload/") 
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        # เปิดไฟล์ภาพที่อัปโหลด
        img = Image.open(file.file)

        # ประมวลผลด้วยโมเดล YOLO
        results = model.predict(source=img)

        # ผลลัพธ์จะถูกเก็บใน numpy.ndarray
        result_image_array = results[0].plot()  # วาดผลลัพธ์ลงบนภาพ

        # แปลง numpy.ndarray เป็น PIL.Image
        result_image = Image.fromarray(result_image_array)

        # แปลงภาพผลลัพธ์และภาพที่อัปโหลดเป็น Base64
        uploaded_image_base64 = convert_image_to_base64(img)
        result_image_base64 = convert_image_to_base64(result_image)

        # ดึงข้อมูลการตรวจจับ
        detection = []
        instructions_list = []

        for box in results[0].boxes:
            label_index = int(box.cls[0].item())  # ดึง index ของ class
            detection_name = class_names[label_index]  # แปลง index เป็นชื่อยา
            detection.append(detection_name)
            # เพิ่มข้อมูลคำแนะนำ
            instructions_list.append(instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ"))

        # ส่งข้อมูลไปยังเทมเพลต
        return templates.TemplateResponse(
            "result_h.html",
            {
                "request": request,
                "uploaded_image_base64": uploaded_image_base64,
                "result_image_base64": result_image_base64,
                "detections": zip(detection, instructions_list),
            }
        )

    except Exception as e:
        # ถ้ามีข้อผิดพลาดเกิดขึ้น
        return HTMLResponse(content=f"Error: {str(e)}", status_code=500)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # เปิดไฟล์ภาพที่อัปโหลด
        img = Image.open(file.file)

        # ประมวลผลด้วยโมเดล YOLO
        results = model.predict(source=img)

        # ดึงข้อมูลการตรวจจับ
        detections = []
        for box in results[0].boxes:
            label_index = int(box.cls[0].item())  # ดึง index ของ class
            detection_name = class_names[label_index]  # แปลง index เป็นชื่อยา
            instructions_text = instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ")
            detections.append({
                "name": detection_name,
                "instructions": instructions_text
            })

        # ส่งผลลัพธ์เป็น JSON
        return JSONResponse(content={"detections": detections})

    except Exception as e:
        # ถ้ามีข้อผิดพลาดเกิดขึ้น
        return JSONResponse(content={"error": str(e)}, status_code=500)

# เพิ่ม endpoint สำหรับรับภาพจากกล้องเป็น Base64
@app.post("/capture/")
async def capture_image(request: Request, image_base64: str = Form(...)):
    try:
        # แปลง Base64 เป็นภาพ
        image_data = base64.b64decode(image_base64.split(",")[1])
        img = Image.open(BytesIO(image_data))

        # ประมวลผลภาพ
        results = model.predict(source=img)

        # ดึงข้อมูลการตรวจจับ
        detections = []
        for box in results[0].boxes:
            label_index = int(box.cls[0].item())
            detection_name = class_names[label_index]
            instructions_text = instructions.get(detection_name, "ไม่พบข้อมูลคำแนะนำ")
            detections.append({
                "name": detection_name,
                "instructions": instructions_text
            })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
