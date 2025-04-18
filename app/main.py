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

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for uploaded images
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load YOLOv8 model
model_path = Path("../models/best.pt")
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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 