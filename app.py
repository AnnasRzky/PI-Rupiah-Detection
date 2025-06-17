from flask import Flask, render_template, request, jsonify, Response
import os
import uuid
import cv2
import shutil
from werkzeug.utils import secure_filename
from md_model import detect_image, detect_webcam_frame

app = Flask(__name__)

camera = None
is_running = False  # Flag untuk loop real-time


# =======================
# Route halaman utama
# =======================
@app.route('/')
def index():
    # Hapus hasil lama setiap reload
    results_dir = 'static/results'
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    return render_template('rupiah_detection_app.html')


# =======================
# Upload dan Deteksi Gambar
# =======================
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Simpan file dengan nama unik
    original_name = secure_filename(file.filename)
    ext = os.path.splitext(original_name)[1]
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    upload_folder = 'static/uploads'
    os.makedirs(upload_folder, exist_ok=True)
    upload_path = os.path.join(upload_folder, unique_filename)
    file.save(upload_path)

    try:
        result_data = detect_image(upload_path)

        return jsonify({
            "success": True,
            "image_path": upload_path.replace('static/', ''),
            **result_data
        })

    except Exception as e:
        print("❌ Deteksi gagal:", str(e))
        return jsonify({"error": f"Deteksi gagal: {str(e)}"}), 500


# =======================
# Video Feed (Webcam)
# =======================
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen_frames():
    global camera, is_running
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        is_running = True

    while is_running:
        success, frame = camera.read()
        if not success:
            break

        frame, _ = detect_webcam_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if camera:
        camera.release()
        camera = None


# =======================
# JSON Data dari Webcam (polling tiap detik)
# =======================
@app.route('/realtime-json')
def realtime_json():
    global camera
    if camera is None or not camera.isOpened():
        return jsonify({"error": "Kamera belum aktif"}), 400

    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Gagal membaca frame dari kamera"}), 500

    _, result_json = detect_webcam_frame(frame)
    return jsonify(result_json)


# =======================
# Stop Kamera
# =======================
@app.route('/stop-detection')
def stop_detection():
    global camera, is_running
    is_running = False

    if camera is not None:
        camera.release()
        camera = None
        print("✅ Kamera berhasil dimatikan.")

    return jsonify({"status": "stopped"})


# =======================
# Run Flask App
# =======================
if __name__ == '__main__':
    app.run(debug=True)
