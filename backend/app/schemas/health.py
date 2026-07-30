"""Response schemas for the health and meta endpoints.

Schemas are the API's public contract. Keeping them separate from ORM models
means the database can be refactored without silently changing what clients
receive - and means an internal column is never leaked by accident.
"""

from typing import Literal

from pydantic import BaseModel, Field

ComponentStatus = Literal["ok", "degraded", "down"]


class LivenessResponse(BaseModel):
    """Is the process alive? Deliberately checks no dependencies."""

    status: Literal["ok"] = "ok"


class DependencyCheck(BaseModel):
    name: str = Field(..., examples=["postgres"])
    status: ComponentStatus
    latency_ms: float | None = Field(None, description="Round-trip time of the probe")
    error: str | None = None


class ReadinessResponse(BaseModel):
    """Can the process serve traffic? Checks every hard dependency."""

    status: ComponentStatus
    dependencies: list[DependencyCheck]


class AppInfoResponse(BaseModel):
    name: str
    version: str
    environment: str
    api_version: str
