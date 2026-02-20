from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from plant_app.ml.predictor import PlantPredictor
from medicinal_data import MedicinalData   # ✅ Import the class
import os

app = FastAPI()

# -----------------------------
# CORS Configuration
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
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


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "Medicinal Plant API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        return {"error": "Model not loaded"}

    image_bytes = await file.read()

    # Get prediction from model
    plant_name, confidence = predictor.predict(image_bytes)

    # Get medicinal uses from class method
    uses = MedicinalData.get_uses(plant_name)

    return {
        "plant_name": plant_name,
        "confidence": round(confidence, 2),
        "medicinal_uses": uses
    }