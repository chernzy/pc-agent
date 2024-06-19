from core.config import SETTINGS
from core.models import (
    app,
    EMBEDDING_MODEL,
    # LLM_ENGINE,
    RERANK_MODEL
)

prefix = SETTINGS.api_prefix

if EMBEDDING_MODEL is not None:
    from api.v1.endpoint import embeddings
    app.include_router(embeddings.embedding_router, prefix=prefix, tags=["Embedding"])
    
if SETTINGS.engine == "trtllm":
    from api.v1.endpoint.trtllm_routes import chat_router
else:
    from api.v1.endpoint.chat import chat_router

app.include_router(chat_router, prefix=prefix, tags=["Chat Completion"])