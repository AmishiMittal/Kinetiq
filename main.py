from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictRequest, SquadResponse
from backend.services.prediction_service import PredictionService

app = FastAPI(title="injury-monitor backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Demo/hackathon friendly; lock down for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_prediction_service = PredictionService()


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/squad", response_model=SquadResponse)
def get_squad(
    count: int = Query(default=6, ge=1, le=30),
    use_ai_advice: bool = Query(default=False),
) -> SquadResponse:
    athletes = _prediction_service.generate_demo_squad(count=count, use_ai_advice=use_ai_advice)
    return SquadResponse(athletes=athletes, meta={"source": "demo-pipeline"})


@app.post("/api/predict/squad", response_model=SquadResponse)
def predict_squad(req: PredictRequest) -> SquadResponse:
    # For now, only demo mode is supported because input schema for real-world play data
    # isn't defined in the repo yet.
    athletes = _prediction_service.generate_demo_squad(count=req.count, use_ai_advice=req.use_ai_advice)
    return SquadResponse(athletes=athletes, meta={"mode": req.mode})


@app.get("/")
def root() -> Dict[str, Any]:
    return {"name": "injury-monitor backend", "routes": ["/api/health", "/api/squad", "/api/predict/squad"]}

