from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
from md_model import detect_image, detect_webcam_frame
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
camera = None  # Global variabel webcam

# =======================
# Route halaman utama
# =======================
@app.route('/')
def index():
    return render_template('rupiah_detection_app.html')

# =======================
# Upload dan Deteksi Gambar
# =======================
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Simpan file upload ke folder upload
    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"

    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, unique_filename)
    file.save(upload_path)

    try:
        # Deteksi dengan YOLO
        model = YOLO("md_model.pt")
        results = model.predict(
            source=upload_path,
            save=True,
            save_txt=False,
            project="static/results",
            name="",
            exist_ok=True
        )

        # Ambil path hasil deteksi dari YOLO
        result_path = getattr(results[0], 'save_path', None)

        # Fallback jika path tidak ditemukan
        if not result_path or not os.path.exists(result_path):
            print("⚠️ save_path tidak ditemukan. Mencari file terbaru di static/results/predict")
            result_dir = os.path.join("static", "results", "predict")
            if not os.path.exists(result_dir):
                return jsonify({'error': 'Folder hasil tidak ditemukan'}), 500

            all_files = sorted(
                [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png'))],
                key=lambda x: os.path.getmtime(os.path.join(result_dir, x)),
                reverse=True
            )
            if all_files:
                result_path = os.path.join(result_dir, all_files[0])
            else:
                return jsonify({'error': 'Tidak ada hasil deteksi ditemukan'}), 500

        return jsonify({'result': result_path})

    except Exception as e:
        print("❌ Deteksi gagal:", str(e))
        return jsonify({'error': f'Deteksi gagal: {str(e)}'}), 500

# =======================
# Streaming Webcam
# =======================
def gen_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while True:
        if camera is None or not camera.isOpened():
            break
        success, frame = camera.read()
        if not success:
            break
        frame = detect_webcam_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# =======================
# Hentikan Webcam
# =======================
@app.route('/stop-detection')
def stop_detection():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("✅ Kamera berhasil dimatikan.")
    return jsonify({"status": "stopped"})

# =======================
# Jalankan Flask App
# =======================
if __name__ == '__main__':
    app.run(debug=True)
