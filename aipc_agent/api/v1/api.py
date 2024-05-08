from fastapi import APIRouter

# from api.v1.endpoint import agent
from api.v1.endpoint import chat
from api.v1.endpoint import embeddings

api_router = APIRouter()

api_router.include_router(chat.chat_router, prefix="", tags=["chat"])
api_router.include_router(embeddings.embedding_router, prefix="", tags=["embeddings"])