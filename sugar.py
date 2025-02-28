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
model = YOLO("models/sugar.pt")

# class names ของยา
class_names = {
   0: 'Metformin_HCl500', 
   1: 'Metformin_HCl850'
}

# คำแนะนำการใช้ยาและข้อควรระวัง
instructions = {
    "Metformin_HCl500" : "",
    "Metformin_HCl850" : ""
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
    return templates.TemplateResponse("sugar.html", {"request": request})

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
            "result_s.html",
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

