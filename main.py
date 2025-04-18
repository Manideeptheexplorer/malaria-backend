import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
from datetime import datetime
import shutil
from pathlib import Path

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/malaria_model.h5")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")

# Create necessary directories
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Malaria Detection API",
    description="API for detecting malaria in blood cell images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image for model prediction"""
    # Resize image
    image = image.resize((224, 224))
    # Convert to numpy array
    image_array = np.array(image)
    # Normalize pixel values
    image_array = image_array / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def save_uploaded_file(file: UploadFile, directory: str) -> str:
    """Save uploaded file to specified directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(directory, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return file_path

@app.get("/")
async def root():
    return {"message": "Malaria Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Make prediction
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Model not loaded"}
            )
        
        prediction = model.predict(processed_image)
        confidence = float(prediction[0][0])
        is_infected = confidence > 0.5
        
        # Save the uploaded file
        file_path = save_uploaded_file(file, UPLOAD_DIR)
        
        return {
            "infected": bool(is_infected),
            "confidence": float(confidence),
            "image_path": file_path
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=ENVIRONMENT == "development") 