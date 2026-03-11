"""Pydantic templates describing common persona outputs."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class NormalResponse(BaseModel):
    response: str
    signature: str = "Normal"


class IcchaResponse(BaseModel):
    goal_detected: bool
    goal: Optional[str] = None
    details: Optional[str] = None
    response: Optional[str] = None
    signature: str = "Iccha"


class KaryaResponse(BaseModel):
    goal: str
    prompt: str
    timestamp: datetime
    signature: str = "Karya"

    @field_validator("timestamp", mode="before")
    def parse_timestamp(cls, value):
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value


class NiyatiResponse(BaseModel):
    route_to: Literal["Iccha", "Karya", "Karma"]
    user_prompt: str
    signature: str = "Niyati"


class KarmaResponse(BaseModel):
    action: str
    tool: str
    parameters: dict
    signature: str = "Karma"
