import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import easyocr
import os
from rapidfuzz import process, fuzz
import re

# Load model YOLO
model = YOLO('models/md_model.pt')

# Inisialisasi EasyOCR reader
reader = easyocr.Reader(['en', 'id'], gpu=True)

# Daftar nominal uang (angka + teks)
nominal_list = [
    "1000", "2000", "5000", "10000", "20000", "50000", "100000",
    "Seribu rupiah", 
    "Dua ribu rupiah", 
    "Lima ribu rupiah",
    "Sepuluh ribu rupiah", 
    "Dua puluh ribu rupiah",
    "Lima puluh ribu rupiah", 
    "Seratus ribu rupiah"
]

def extract_number(text):
    numbers = re.findall(r'\\d{3,6}', text)
    for n in numbers:
        if n in nominal_list:
            return n
    return None


def extract_number(text):
    """
    Ekstrak angka dari teks OCR dan cocokkan dengan nominal yang valid.
    """
    matches = re.findall(r'\d{3,6}', text)
    for m in matches:
        if m in nominal_list:
            return m
    return None

def fuzzy_match(text, choices=nominal_list):
    """
    Gunakan fuzzy matching untuk mencari nominal teks (misal: 'seratus ribu rupiah'),
    tapi prioritaskan angka valid jika terdeteksi di OCR.
    """
    number_result = extract_number(text)
    if number_result:
        return number_result

    clean_text = text.lower().replace(' ', '')
    best_match, score, _ = process.extractOne(
        clean_text, 
        [c.lower().replace(' ', '') for c in choices], 
        scorer=fuzz.partial_ratio
    )
    if score >= 80:
        for choice in choices:
            if best_match in choice.lower().replace(' ', ''):
                return choice
    return None

def get_final_result(yolo_label, yolo_conf, ocr_text):
    ocr_match = fuzzy_match(ocr_text)
    if ocr_match:
        if ocr_match == yolo_label:
            return yolo_label
        else:
            return ocr_match
    return yolo_label

def detect_image(file):
    img = Image.open(file).convert('RGB')
    img_np = np.array(img)
    results = model.predict(img_np)[0]

    detections = []
    yolo_labels = []
    yolo_confidences = []
    ocr_labels = []
    final_results = []
    boxes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_np[y1:y2, x1:x2]

        ocr_result = reader.readtext(crop)
        ocr_text = ' '.join([res[1] for res in ocr_result]).strip()

        yolo_label = model.names[cls]
        final_result = get_final_result(yolo_label, conf, ocr_text)

        detections.append({
            'yolo_class': yolo_label,
            'confidence': f'{conf:.2f}',
            'ocr_text': ocr_text,
            'final_result': final_result,
            'bbox': [x1, y1, x2, y2]
        })

        yolo_labels.append(yolo_label)
        yolo_confidences.append(f"{conf:.2f}")
        ocr_labels.append(ocr_text)
        final_results.append(final_result)
        boxes.append({
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "label": final_result
        })

    return {
        'detections': detections,
        'yolo_result': {
            'labels': yolo_labels,
            'confidences': yolo_confidences
        },
        'ocr_result': {
            'label': ', '.join(ocr_labels)
        },
        'final_result': ', '.join(final_results),
        'boxes': boxes
    }

def detect_webcam_frame(frame):
    try:
        results = model.predict(frame)[0]
        detections = []

        yolo_labels = []
        yolo_confidences = []
        ocr_labels = []
        final_results = []
        boxes = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else frame

            ocr_text = ""
            if crop is not None and crop.size > 0:
                ocr_result = reader.readtext(crop)
                ocr_text = ' '.join([res[1] for res in ocr_result]).strip()

            yolo_label = model.names[cls]
            final_result = get_final_result(yolo_label, conf, ocr_text)

            detections.append({
                'yolo_class': yolo_label,
                'confidence': f'{conf:.2f}',
                'ocr_text': ocr_text,
                'final_result': final_result,
                'bbox': [x1, y1, x2, y2]
            })

            label = f'{final_result} ({conf:.2f})'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            yolo_labels.append(yolo_label)
            yolo_confidences.append(f"{conf:.2f}")
            ocr_labels.append(ocr_text)
            final_results.append(final_result)
            boxes.append({
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "label": final_result
            })

        result_json = {
            'detections': detections,
            'yolo_result': {
                'labels': yolo_labels,
                'confidences': yolo_confidences
            },
            'ocr_result': {
                'labels': ocr_labels
            },
            'final_result': final_results,
            'boxes': boxes
        }

        return frame, result_json

    except Exception as e:
        print("❌ Error:", str(e))
        return frame, {
            'detections': [],
            'yolo_result': {'labels': [], 'confidences': []},
            'ocr_result': {'labels': []},
            'final_result': [],
            'boxes': []
        }