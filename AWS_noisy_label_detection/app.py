from fastapi import FastAPI, UploadFile, Form
from inference import load_model, predict_from_bytes

app = FastAPI(title="Noise Detection API")

# 先用你訓練好的 checkpoint 路徑
MODEL_PATH = "checkpoints/model.pt"
model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect-noise")
async def detect_noise(file: UploadFile, y_tilde: int = Form(...)):
    image_bytes = await file.read()
    result = predict_from_bytes(model, image_bytes, y_tilde)
    return result
from fastapi import FastAPI, UploadFile, Form
from typing import List

@app.post("/batch-detect")
async def batch_detect(files: List[UploadFile], y_tilde: int = Form(...)):
    results = []

    for file in files:
        image_bytes = await file.read()
        result = predict_from_bytes(model, image_bytes, y_tilde)
        result["filename"] = file.filename
        results.append(result)

    results = sorted(results, key=lambda x: x["noise_score"], reverse=True)
    return {"results": results}