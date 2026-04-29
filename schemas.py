from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class AthleteHistoryPoint(BaseModel):
    day: str = Field(..., description="Short day label, e.g. Mon/Tue/Wed")
    load: float = Field(..., description="Normalized load (approx. 0.5-2.0 for UI)")
    recovery: float = Field(..., description="Recovery score (0-100)")


class Athlete(BaseModel):
    id: int
    name: str
    riskScore: int = Field(..., ge=0, le=100)
    advice: str
    history: List[AthleteHistoryPoint]


class SquadResponse(BaseModel):
    athletes: List[Athlete]
    meta: dict = Field(default_factory=dict)


class PredictRequest(BaseModel):
    mode: Literal["demo"] = "demo"
    count: int = Field(default=6, ge=1, le=30)
    use_ai_advice: bool = Field(default=False, description="If true, call the Gemini advisor if configured.")

