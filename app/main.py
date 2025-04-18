from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import base64
from PIL import Image
import io
import cv2
import numpy as np
from typing import List, Dict, Any
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.upsampling import Upsample
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
MODEL_PATH = os.getenv("MODEL_PATH", "./models/best.pt")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
TEMP_DIR = os.getenv("TEMP_DIR", "./temp")
PORT = int(os.getenv("PORT", 10000))

# Create necessary directories
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)

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

# Global model variable
model = None

def load_model():
    """Load the model with error handling"""
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
        
        logger.info(f"Loading model from {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if not load_model():
        logger.error("Failed to load model on startup")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Malaria Detection API is running",
        "model_loaded": model is not None,
        "environment": ENVIRONMENT,
        "port": PORT
    }

# Create temporary directory for uploaded images
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load YOLOv8 model
model_path = Path("./models/best.pt")
if not model_path.exists():
    raise FileNotFoundError("YOLOv8 model not found. Please ensure best.pt exists in the models directory.")

# Load model with explicit device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Infection status mapping
infected_map = {
    "red blood cell": False,
    "leukocyte": False,
    "ring": True,
    "trophozoite": True,
    "schizont": True,
    "gametocyte": True,
}

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess the image for YOLOv8 inference."""
    img = cv2.imread(image_path)
    
    # # Convert grayscale to RGB if needed
    # if len(img.shape) == 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize to 640x640 for YOLOv8
    img = cv2.resize(img, (640, 640))
    
    # Enhance contrast
    # lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # l, a, b = cv2.split(lab)
    # l = cv2.equalizeHist(l)
    # lab = cv2.merge((l, a, b))
    # img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    return img

def process_detections(results: Any) -> List[Dict[str, Any]]:
    """Process YOLOv8 detection results into a structured format."""
    detections = []
    result = results[0]  # Get first result (single image)
    
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        class_name = result.names[int(cls)]
        x1, y1, x2, y2 = box.tolist()
        width = x2 - x1
        height = y2 - y1
        
        detections.append({
            "class": class_name,
            "confidence": float(conf),
            "infected": infected_map.get(class_name, False),
            "bbox": [float(x1), float(y1), float(width), float(height)]
        })
    
    return detections

def get_statistics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate detection statistics."""
    total_cells = len(detections)
    infected_count = sum(1 for d in detections if d["infected"])
    non_infected_count = total_cells - infected_count
    
    class_counts = {}
    for d in detections:
        class_name = d["class"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "total_cells": total_cells,
        "infected_count": infected_count,
        "non_infected_count": non_infected_count,
        "class_counts": class_counts
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Preprocess image
        # preprocessed_img = preprocess_image(str(file_path))
        # cv2.imwrite(str(file_path), preprocessed_img)
        
        # Run prediction
        results = model(file_path)
        
        # Process detections
        detections = process_detections(results)
        
        # Get statistics
        statistics = get_statistics(detections)
        
        # Get annotated image
        annotated_img = Image.fromarray(results[0].plot())
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Clean up
        file_path.unlink()
        
        # Prepare response
        response = {
            "detections": detections,
            "statistics": statistics,
            "annotated_image": img_str
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def cleanup():
    # Clean up temporary directory
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR)

if __name__ == "__main__":
    logger.info(f"Starting server on port {PORT}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=ENVIRONMENT == "development"
    ) 