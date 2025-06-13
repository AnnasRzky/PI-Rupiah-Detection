// ========== DOM Elements ==========
const realTimeBtn = document.getElementById("realTimeBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const uploadArea = document.getElementById("uploadArea");
const previewImage = document.getElementById("previewImage");
const previewContainer = document.getElementById("previewContainer");
const detectionResults = document.getElementById("detectionResults");
const loadingSpinner = document.getElementById("loadingSpinner");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const webcamContainer = document.getElementById("webcamContainer");
const webcamVideo = document.getElementById("webcamVideo");

// ========== Inisialisasi Default Result Box ==========
document.getElementById("yoloResultBox").innerHTML = "<h3>📦 YOLO Result</h3><p>Belum ada hasil.</p>";
document.getElementById("ocrResultBox").innerHTML = "<h3>🔤 OCR Result</h3><p>Belum ada hasil.</p>";
document.getElementById("finalResultBox").innerHTML = "<h3>✅ Final Result</h3><p>Belum ada hasil.</p>";

// ========== Upload Image Button / Drag-Drop ==========
uploadBtn.addEventListener("click", () => fileInput.click());
uploadArea.addEventListener("click", () => fileInput.click());

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
        handleFileUpload(fileInput.files[0]);
    }
});

// ========== Handle File Upload ==========
function handleFileUpload(file) {
    previewContainer.style.display = "block";
    loadingSpinner.style.display = "block";
    detectionResults.innerHTML = "";

    // Preview gambar sebelum dikirim
    const reader = new FileReader();
    reader.onload = () => previewImage.src = reader.result;
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload-image", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        loadingSpinner.style.display = "none";

        if (data.error) {
            detectionResults.innerHTML = `<p style='color:red'>${data.error}</p>`;
            return;
        }

        // Tampilkan gambar hasil deteksi
        if (data.result) {
            previewImage.src = "/" + data.result;
        }

        if (data.yolo_result || data.ocr_result || data.best_result) {
            showDetectionResults(data);
        }
    })
    .catch((err) => {
        loadingSpinner.style.display = "none";
        console.error("Upload error:", err);
        detectionResults.innerHTML = "<p style='color:red'>Error saat mengupload gambar.</p>";
    });
}

// ========== Menampilkan Hasil Deteksi ==========
function showDetectionResults(data) {
    const { yolo_result, ocr_result, best_result } = data;

    document.getElementById("yoloResultBox").innerHTML = `
        <h3>📦 YOLO Result</h3>
        <p><strong>Label:</strong> ${yolo_result?.label || '-'}</p>
        <p><strong>Confidence:</strong> ${yolo_result?.confidence || '-'}%</p>
    `;

    document.getElementById("ocrResultBox").innerHTML = `
        <h3>🔤 OCR Result</h3>
        <p><strong>Text:</strong> ${ocr_result?.label || '-'}</p>
        <p><strong>Confidence:</strong> ${ocr_result?.confidence || '-'}%</p>
    `;

    document.getElementById("finalResultBox").innerHTML = `
        <h3>✅ Final Result</h3>
        <p><strong>Output:</strong> ${best_result?.label || '-'}</p>
    `;
}

// ========== Real-Time Webcam Mode ==========
realTimeBtn.addEventListener("click", () => {
    webcamContainer.style.display = "block";
    previewContainer.style.display = "none";
    detectionResults.innerHTML = "";

    webcamVideo.src = "/video_feed";
    stopCameraBtn.style.display = "inline-block";
    realTimeBtn.style.display = "none";
});

stopCameraBtn.addEventListener("click", () => {
    webcamVideo.src = ""; // Hentikan streaming
    webcamContainer.style.display = "none";
    stopCameraBtn.style.display = "none";
    realTimeBtn.style.display = "inline-block";

    fetch("/stop-detection")
        .then(res => res.json())
        .then(() => console.log("✅ Webcam stopped"))
        .catch(err => console.error("❌ Gagal stop webcam", err));
});