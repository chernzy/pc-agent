from fastapi import APIRouter

# from api.v1.endpoint import agent
from api.v1.endpoint import chat

api_router = APIRouter()

# api_router.include_router(agent.router, prefix="/agent", tags=["agent"])
# api_router.include_router(chat_completions.router, prefix="/deprecated/chat/completions", tags=["chat completions"])
api_router.include_router(chat.chat_router, prefix="", tags=["chat"])