document.addEventListener("DOMContentLoaded", function () {
    const realTimeBtn = document.getElementById("realTimeBtn");
    const uploadBtn = document.getElementById("uploadBtn");
    const fileInput = document.getElementById("fileInput");
    const uploadArea = document.getElementById("uploadArea");
    const previewImage = document.getElementById("previewImage");
    const previewContainer = document.getElementById("previewContainer");
    const webcamContainer = document.getElementById("webcamContainer");
    const webcamVideo = document.getElementById("webcamVideo");
    const detectionCanvas = document.getElementById("detectionCanvasStatic");

    const yoloResultBox = document.getElementById("yoloResultBox");
    const ocrResultBox = document.getElementById("ocrResultBox");
    const finalResultBox = document.getElementById("finalResultBox");

    const loadingSpinner = document.getElementById("loadingSpinner");
    const detectionResults = document.getElementById("detectionResults");
    const stopCameraBtn = document.getElementById("stopCameraBtn");

    resetResults();
    let realtimeInterval = null;

    if (uploadBtn) uploadBtn.addEventListener("click", () => fileInput.click());
    if (uploadArea) {
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
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) {
                handleFileUpload(fileInput.files[0]);
            }
        });
    }

    function resetResults() {
        if (yoloResultBox) yoloResultBox.innerHTML = "<h3>📦 YOLO Result</h3><p>Belum ada hasil.</p>";
        if (ocrResultBox) ocrResultBox.innerHTML = "<h3>🔤 OCR Result</h3><p>Belum ada hasil.</p>";
        if (finalResultBox) finalResultBox.innerHTML = "<h3>✅ Final Result</h3><p>Belum ada hasil.</p>";

        if (detectionCanvas) {
            const ctx = detectionCanvas.getContext("2d");
            ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        }
    }

    function handleFileUpload(file) {
        const validTypes = ["image/jpeg", "image/png", "image/webp"];
        if (!validTypes.includes(file.type)) {
            alert("Format file tidak didukung. Gunakan JPEG, PNG, atau WEBP.");
            return;
        }

        resetResults();

        if (previewContainer) previewContainer.style.display = "block";
        if (webcamContainer) webcamContainer.style.display = "none";
        if (loadingSpinner) loadingSpinner.style.display = "block";
        if (detectionResults) detectionResults.innerHTML = "";

        const reader = new FileReader();
        reader.onload = () => {
            if (previewImage) {
                previewImage.onload = () => {
                    const formData = new FormData();
                    formData.append("file", file);

                    fetch("/upload-image", {
                        method: "POST",
                        body: formData
                    })
                        .then(async (res) => {
                            if (loadingSpinner) loadingSpinner.style.display = "none";
                            if (!res.ok) {
                                const errText = await res.text();
                                if (detectionResults) detectionResults.innerHTML = `<p style='color:red'>${errText}</p>`;
                                return;
                            }
                            const data = await res.json();
                            if (data.error) {
                                if (detectionResults) detectionResults.innerHTML = `<p style='color:red'>${data.error}</p>`;
                                return;
                            }

                            updateDetectionResults(data);
                            drawBoxes(data.boxes || []);
                        })
                        .catch((err) => {
                            if (loadingSpinner) loadingSpinner.style.display = "none";
                            console.error("Upload error:", err);
                            if (detectionResults) detectionResults.innerHTML = "<p style='color:red'>Gagal mengunggah atau memproses gambar.</p>";
                        });
                };
                previewImage.src = reader.result;
            }
        };
        reader.readAsDataURL(file);

        fileInput.value = "";
    }

    function updateDetectionResults(data) {
        const yolo = data.yolo_result;
        const ocr = data.ocr_result;
        const final = data.final_result;

        if (yoloResultBox) {
            yoloResultBox.innerHTML = `
                <h3>📦 YOLO Result</h3>
                <p>Label: ${yolo.labels?.join(', ') || "-"}</p>
                <p>Confidence: ${yolo.confidences?.join(', ') || "-"}</p>
            `;
        }

        if (ocrResultBox) {
            const ocrLabels = ocr.labels || (ocr.label ? [ocr.label] : []);
            ocrResultBox.innerHTML = `
                <h3>🔤 OCR Result</h3>
                <p>${ocrLabels.length ? ocrLabels.join(', ') : "-"}</p>
            `;
        }

        if (finalResultBox) {
            finalResultBox.innerHTML = `
                <h3>✅ Final Result</h3>
                <p>${final || "-"}</p>
            `;
        }
    }

    function drawBoxes(boxes) {
        if (!detectionCanvas || !previewImage) return;

        const ctx = detectionCanvas.getContext("2d");
        const img = previewImage;

        detectionCanvas.width = img.width;
        detectionCanvas.height = img.height;

        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);

        boxes.forEach((box) => {
            ctx.strokeStyle = "lime";
            ctx.lineWidth = 2;
            ctx.strokeRect(box.x, box.y, box.width, box.height);

            ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
            ctx.fillRect(box.x, box.y - 20, box.width, 20);

            ctx.fillStyle = "#fff";
            ctx.font = "14px Arial";
            ctx.fillText(box.label, box.x + 4, box.y - 5);
        });
    }

    if (realTimeBtn) {
        realTimeBtn.addEventListener("click", () => {
            resetResults();
            if (webcamContainer) webcamContainer.style.display = "block";
            if (previewContainer) previewContainer.style.display = "none";
            if (webcamVideo) webcamVideo.src = "/video_feed";

            realtimeInterval = setInterval(() => {
                fetch("/realtime-json")
                    .then((res) => res.json())
                    .then((data) => {
                        updateDetectionResults(data);
                        drawBoxes(data.boxes || []);
                    })
                    .catch((err) => {
                        console.error("❌ Gagal ambil data realtime:", err);
                    });
            }, 1000);

            if (stopCameraBtn) stopCameraBtn.style.display = "inline-block";
            realTimeBtn.style.display = "none";
        });
    }

    if (stopCameraBtn) {
        stopCameraBtn.addEventListener("click", () => {
            if (webcamVideo) webcamVideo.src = "";
            if (webcamContainer) webcamContainer.style.display = "none";
            stopCameraBtn.style.display = "none";
            if (realTimeBtn) realTimeBtn.style.display = "inline-block";

            if (realtimeInterval) {
                clearInterval(realtimeInterval);
                realtimeInterval = null; 
            }

            fetch("/stop-detection")
                .then(res => res.json())
                .then(() => console.log("✅ Webcam stopped"))
                .catch(err => console.error("❌ Gagal stop webcam", err));
        });
    }

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && stopCameraBtn && stopCameraBtn.style.display !== "none") {
            stopCameraBtn.click();
        }
    });
});