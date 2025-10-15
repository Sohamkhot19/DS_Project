from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from typing import List
import os

# Initialize FastAPI app
app = FastAPI(title="Movie Rating Prediction API", version="1.0")

# Load the trained model
try:
    # Get the path to the models directory (one level up from src/)
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "..", "models", "Random_Forest_Tuned_model.pkl")
    model_path = os.path.abspath(model_path)
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define input data structure
class MovieFeatures(BaseModel):
    runtimeMinutes: float
    averageRating: float
    numVotes: int
    budget: float
    gross: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "runtimeMinutes": 120,
                "averageRating": 7.5,
                "numVotes": 50000,
                "budget": 100000000,
                "gross": 250000000
            }
        }

# Define output structure
class PredictionOutput(BaseModel):
    rating_status: str
    confidence: float

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Movie Rating Prediction API",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    model_status = "loaded" if model is not None else "not loaded"
    return {"status": "healthy", "model": model_status}

# Prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(features: MovieFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input data in the correct order
        input_data = pd.DataFrame([{
            "runtimeMinutes": features.runtimeMinutes,
            "averageRating": features.averageRating,
            "numVotes": features.numVotes,
            "budget": features.budget,
            "gross": features.gross
        }])
        
        # Handle missing values (fill with median values used during training)
        input_data = input_data.fillna({
            "runtimeMinutes": 107.0,
            "averageRating": 6.1,
            "numVotes": 23000,
            "budget": 25000000,
            "gross": 50000000
        })
          # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Get confidence (max probability)
        confidence = float(np.max(prediction_proba))
        
        return PredictionOutput(
            rating_status=str(prediction[0]),
            confidence=round(confidence, 4)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

# Run with: uvicorn app:app --reload
