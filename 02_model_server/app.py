from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import os

app = FastAPI(
    title="Construction Worker Safety Model API",
    version="1.0.0",
    description="Experimental movement analysis API"
)

class IMURequest(BaseModel):
    windows: List[List[List[float]]] = Field(
        description="Batch of IMU windows [batch, time, channels]"
    )

class KeypointRequest(BaseModel):
    frames: List[List[List[float]]] = Field(
        description="Pose sequence [frames, joints, features]"
    )

@app.get("/")
def home():
    return {
        "service": "Construction Worker Safety API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "running",
        "model_repo_configured": bool(os.getenv("HF_MODEL_REPO"))
    }

@app.post("/predict/imu")
def predict_imu(request: IMURequest):
    raise HTTPException(
        status_code=503,
        detail="IMU model inference is not yet connected."
    )

@app.post("/predict/keypoints")
def predict_keypoints(request: KeypointRequest):
    raise HTTPException(
        status_code=503,
        detail="Keypoint model inference is not yet connected."
    )