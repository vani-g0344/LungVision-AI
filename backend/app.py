from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="LungVision-AI Backend")

@app.get("/")
def home():
    return {"status": "LungVision-AI backend running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {
        "prediction": "Not connected yet",
        "confidence": 0.0
    }
