from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from plant_app.ml.predictor import PlantPredictor
import os

app = FastAPI()

# CORS (change "*" to frontend URL in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Load ML Model (SAFE PATH)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "final_model.keras")

try:
    predictor = PlantPredictor(model_path)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    predictor = None


@app.get("/")
def root():
    return {"message": "Medicinal Plant API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()
    plant_name, confidence = predictor.predict(image_bytes)

    return {
        "plant_name": plant_name,
        "confidence": round(confidence, 2)
    }