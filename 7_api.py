"""
Drug Repurposing with Graph Neural Networks
Step 7: FastAPI REST Backend
=============================
Serves the trained model as a REST API.
Use this to integrate with a React frontend or call from other services.

Endpoints:
    GET  /diseases?q=alzheimer       → search diseases by name
    POST /predict                    → top-K drug predictions for a disease
    POST /explain                    → biological explanation for a pair
    GET  /health                     → health check

Run with:
    uvicorn 7_api:app --reload --port 8000
"""

import os
import torch
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from explain import (load_model_and_data, predict_drugs_for_disease,
                     explain_prediction, search_diseases, _idx_to_name)

# ── Global state ──────────────────────────────────────────────────────────
MODEL = DATA = KIND_DFS = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, DATA, KIND_DFS
    print("Loading model...")
    MODEL, DATA, KIND_DFS = load_model_and_data()
    print("Model loaded.")
    yield


app = FastAPI(
    title="Drug Repurposing GNN API",
    description="GNN-powered drug repurposing predictions over Hetionet",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    disease_idx: int
    top_k: int = 10


class ExplainRequest(BaseModel):
    drug_idx: int
    disease_idx: int


# ── Endpoints ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/diseases")
def diseases(q: str = Query("", description="Search query for disease name")):
    """Search diseases by name."""
    results = search_diseases(KIND_DFS, q, top_n=20)
    if results.empty:
        return {"results": []}
    return {
        "results": results.rename(
            columns={"node_idx": "id", "name": "name"}
        ).to_dict(orient="records")
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Return top-K drug candidates for a given disease index."""
    n_diseases = DATA["Disease"].num_nodes
    if req.disease_idx < 0 or req.disease_idx >= n_diseases:
        raise HTTPException(400, f"disease_idx out of range [0, {n_diseases})")

    disease_name = _idx_to_name(KIND_DFS, "Disease", req.disease_idx)
    top_drugs    = predict_drugs_for_disease(
        MODEL, DATA, req.disease_idx, top_k=req.top_k)

    return {
        "disease": {"idx": req.disease_idx, "name": disease_name},
        "predictions": [
            {
                "rank":     i + 1,
                "drug_idx": drug_idx,
                "drug":     _idx_to_name(KIND_DFS, "Compound", drug_idx),
                "score":    round(score, 4),
            }
            for i, (drug_idx, score) in enumerate(top_drugs)
        ],
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    """Return biological explanation paths for a (drug, disease) pair."""
    exp = explain_prediction(
        MODEL, DATA, req.drug_idx, req.disease_idx, KIND_DFS)
    return exp


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("7_api:app", host="0.0.0.0", port=8000, reload=True)
