from fastapi import APIRouter

from api.v1.endpoint import agent
from api.v1.endpoint import chat_completions

api_router = APIRouter()

api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
api_router.include_router(chat_completions.router, prefix="/chat/completions", tags=["chat completions"])