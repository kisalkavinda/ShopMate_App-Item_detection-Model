import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "my_model.pt")
model = YOLO(MODEL_PATH)
app = FastAPI(title="YOLO Object Detection")

@app.get("/")
def root():
    return {
        "message": "YOLO API is running", 
        "model": os.path.basename(MODEL_PATH)
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    try:
        # Read image
        img_bytes = await file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Inference (force CPU for Spaces)
        results = model.predict(image, device="cpu", verbose=False)
        
        # Format results
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]  # [x1,y1,x2,y2]
                })
        
        return {"detections": detections}
    
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")