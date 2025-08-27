"""Module containing response templates for different personas.

Returns:
    Various response templates as Pydantic models.
"""

from datetime import datetime
from typing import Literal, Optional
import numpy
from pydantic import BaseModel, field_validator


class NormalResponse(BaseModel):
    """Normal Response Template

    Args:
        BaseModel: Pydantic BaseModel
    """

    response: str
    signature: str = "Normal"


class IcchaResponse(BaseModel):
    """Iccha Response Template

    Args:
        BaseModel: Pydantic BaseModel
    """

    goal_detected: bool
    goal: Optional[str] = None
    details: Optional[str] = None
    response: Optional[str] = None
    signature: str = "Iccha"


class KaryaResponse(BaseModel):
    """Karya Response Template

    Args:
        BaseModel: Pydantic BaseModel
    """

    goal: str
    prompt: str
    timestamp: datetime
    signature: str = "Karya"

    @field_validator("timestamp", mode="before")
    def parse_timestanp(cls, v):  # pylint: disable=no-self-argument
        """Parses timestamp from various formats

        Args:
            v: _input value

        Returns:
            v: parsed value
        """
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        if isinstance(v, numpy.datetime64):
            return v.astype("M8[ms]").astype(datetime)
        return v


class NiyatiResponse(BaseModel):
    """Niyati Response Template

    Args:
        BaseModel: Pydantic BaseModel
    """

    route_to: Literal["Iccha", "Karya", "Karma"]
    user_prompt: str
    signature: str = "Niyati"


class KarmaResponse(BaseModel):
    """Karma Response Template

    Args:
        BaseModel: Pydantic BaseModel
    """

    task: str
    status: Literal["Completed", "Failed", "In Progress"]
    result: Optional[str] = None
    signature: str = "Karma"
