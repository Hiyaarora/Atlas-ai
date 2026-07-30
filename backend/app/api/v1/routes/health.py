"""Health endpoints.

Liveness vs readiness is not pedantry - it is how orchestrators avoid
outages:

* `/health/live`  - "is the process running?" If this fails, Kubernetes or
  Docker *restarts* the container. It must therefore check nothing external:
  a Postgres outage should not cause a restart loop of a healthy API.
* `/health/ready` - "should traffic be routed here?" If this fails, the load
  balancer *removes the instance from rotation* without killing it. This one
  checks the hard dependencies and recovers on its own once they return.
"""

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.schemas.health import AppInfoResponse, LivenessResponse, ReadinessResponse
from app.services import health_service

router = APIRouter(prefix="/health", tags=["health"])


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
)
async def liveness() -> LivenessResponse:
    return LivenessResponse()


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "One or more dependencies are unavailable"}},
)
async def readiness(
    response: Response,
    session: AsyncSession = Depends(get_db),
) -> ReadinessResponse:
    result = await health_service.get_readiness(session)
    if result.status == "down":
        # Body still returns the per-dependency detail, so an operator can
        # see *what* is broken from the probe response alone.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


@router.get(
    "/info",
    response_model=AppInfoResponse,
    summary="Build and environment information",
)
async def info() -> AppInfoResponse:
    return AppInfoResponse(
        name=settings.app_name,
        version="0.1.0",
        environment=settings.app_env,
        api_version="v1",
    )
