"""Aggregates every v1 route module into a single router.

New capability = new module in `routes/` + one `include_router` line here.
`app/main.py` never changes.
"""

from fastapi import APIRouter

from app.api.v1.routes import auth, conversations, documents, health

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(conversations.router)
api_router.include_router(documents.router)
