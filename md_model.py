import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr

# Load model YOLO
model = YOLO('models/md_model.pt')  # Pastikan path ini benar

# Inisialisasi EasyOCR reader sekali saja
reader = easyocr.Reader(['en'], gpu=True)  # Ubah gpu=False jika tidak pakai GPU

def detect_image(file):
    # Baca gambar dari file upload
    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    # Deteksi objek dengan YOLO
    results = model.predict(img_np)[0]

    # Ambil hasil deteksi
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_np[y1:y2, x1:x2]

        # OCR dengan EasyOCR
        ocr_result = reader.readtext(crop)
        text = ' '.join([res[1] for res in ocr_result])

        detections.append({
            'class': model.names[cls],
            'confidence': f'{conf:.2f}',
            'text': text.strip(),
            'bbox': [x1, y1, x2, y2]
        })

    return {'detections': detections}

def detect_webcam_frame(frame):
    # Deteksi objek dengan YOLO
    results = model.predict(frame)[0]

    # Gambar bounding box di frame
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f'{model.names[cls]} {conf:.2f}'

        # OCR dengan EasyOCR
        crop = frame[y1:y2, x1:x2]
        ocr_result = reader.readtext(crop)
        text = ' '.join([res[1] for res in ocr_result])

        # Gambar bounding box dan text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label + f' | {text.strip()}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame
