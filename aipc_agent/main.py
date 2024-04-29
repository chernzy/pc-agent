from core.config import SETTINGS
from core.models import (
    app,
    EMBEDDING_MODEL,
    LLM_ENGINE,
    RERANK_MODEL
)

prefix = SETTINGS.api_prefix

if EMBEDDING_MODEL is not None:
    pass

if LLM_ENGINE is not None:
    from api.v1.api import api_router
    app.include_router(api_router, prefix=prefix, tags=["Chat Completion"])