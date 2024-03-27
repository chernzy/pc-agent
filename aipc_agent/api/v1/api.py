from fastapi import APIRouter

from api.v1.endpoint import agent

api_router = APIRouter()

api_router.include_router(agent.router, prefix="/agent", tags=["agent"])