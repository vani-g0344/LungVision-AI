import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model'))

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq
from inference import load_model, predict # pyright: ignore[reportMissingImports]
from gradcam import GradCAM # type: ignore
import tempfile
import logging

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(

    title="LungVision AI",
    description="Lung cancer detection with Grad-CAM + GenAI explanation",
    version="1.0.0"

# Serve frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/app")
def serve_frontend():
    return FileResponse("frontend/index.html")
)

# CORS — allow all in dev, restrict in production
if os.getenv("ENV") == "production":
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
else:
    ALLOWED_ORIGINS = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load model — fail early if not found
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'lungvision_model.pth')
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Download it and place in /model folder.")

model = load_model(MODEL_PATH)
gradcam = GradCAM(model)

# Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")
groq_client = Groq(api_key=api_key)

CLASS_NAMES = ['Benign', 'Malignant', 'Normal']
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
MAX_FILE_SIZE_MB = 10


def get_explanation(prediction: str, confidence: float, all_scores: dict) -> str:
    prompt = f"""A lung CT scan was analyzed by an AI model:
- Prediction: {prediction}
- Confidence: {confidence}%
- Scores: Benign {all_scores['Benign']}%, Malignant {all_scores['Malignant']}%, Normal {all_scores['Normal']}%

Write a clear, compassionate explanation in simple human language (3-4 sentences). Recommend consulting a doctor."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content # pyright: ignore[reportReturnType]
    except Exception as e:
        logger.error(f"GROQ API error: {e}")
        return (
            f"The scan shows {prediction} patterns with {confidence:.1f}% confidence. "
            "Please consult a qualified doctor for proper medical evaluation."
        )


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    # Validate file extension
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    contents = await file.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size is {MAX_FILE_SIZE_MB}MB."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, model)

        if result["prediction"] not in CLASS_NAMES:
            raise HTTPException(status_code=500, detail="Unexpected prediction class from model.")

        explanation = get_explanation(
            result["prediction"],
            result["confidence"],
            result["all_scores"]
        )

        class_idx = CLASS_NAMES.index(result["prediction"])
        heatmap = gradcam.generate(tmp_path, class_idx)

        logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%)")

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "all_scores": result["all_scores"],
            "explanation": explanation,
            "heatmap": heatmap
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction pipeline error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")
    finally:
        os.unlink(tmp_path)


@app.get("/")
def root():
    return {"status": "LungVision-AI backend running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "groq_configured": api_key is not None
    }
