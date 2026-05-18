"""
ADMET-Net FastAPI Backend
Run: uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import yaml
import traceback

from src.model import build_model
from src.featurizer import MolecularFeaturizer

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="ADMET-Net API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ───────────────────────────────────────────────
print("Loading ADMET-Net model...")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cpu")
model, _ = build_model(cfg)

# ✅ Key is 'model_state' (confirmed from your checkpoint)
checkpoint = torch.load("models/admet_net_best.pt", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()
featurizer = MolecularFeaturizer()
print("✅ Featurizer ready")

print(f"✅ ADMET-Net loaded! Best epoch: {checkpoint['epoch']} | Val loss: {checkpoint['val_loss']:.4f}")

# ── Task definitions ─────────────────────────────────────────────────────────
CLASSIFICATION_TASKS = ["bioavailability", "bbb", "cyp3a4", "cyp2c9", "cyp2d6", "herg", "ames", "dili"]
REGRESSION_TASKS     = ["caco2", "logP", "half_life", "clearance"]

# Risk thresholds for toxicity tasks
RISK_THRESHOLDS = {
    "herg":  {"high": 0.7, "moderate": 0.4},
    "ames":  {"high": 0.6, "moderate": 0.3},
    "dili":  {"high": 0.6, "moderate": 0.3},
    "default": {"high": 0.7, "moderate": 0.4},
}

def get_risk(task, prob):
    t = RISK_THRESHOLDS.get(task, RISK_THRESHOLDS["default"])
    if prob > t["high"]:   return "high"
    if prob > t["moderate"]: return "moderate"
    return "low"

# ── Request / Response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str
    drug_name: str = "Unknown"

class PredictResponse(BaseModel):
    drug_name: str
    smiles: str
    predictions: dict
    status: str = "success"

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "ADMET-Net API is running!",
        "model": "admet_net_best.pt",
        "tasks": CLASSIFICATION_TASKS + REGRESSION_TASKS
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "epoch": int(checkpoint["epoch"]),
        "val_loss": float(checkpoint["val_loss"])
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        # Step 1: SMILES → molecular graph
        graph = featurizer.smiles_to_graph(req.smiles)
        if graph is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse SMILES string: {req.smiles}"
            )

        graph = graph.to(device)

        # Step 2: Run your trained model
        with torch.no_grad():
            outputs = model(graph)

        # Step 3: Parse predictions into clean JSON
        predictions = {}

        for task in CLASSIFICATION_TASKS:
            if task in outputs and outputs[task] is not None:
                prob = torch.sigmoid(outputs[task]).item()
                predictions[task] = {
                    "type":        "classification",
                    "probability": round(prob, 4),
                    "label":       "Positive" if prob > 0.5 else "Negative",
                    "risk":        get_risk(task, prob),
                }

        for task in REGRESSION_TASKS:
            if task in outputs and outputs[task] is not None:
                val = outputs[task].item()
                predictions[task] = {
                    "type":  "regression",
                    "value": round(val, 4),
                }

        return PredictResponse(
            drug_name=req.drug_name,
            smiles=req.smiles,
            predictions=predictions,
            status="success"
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
