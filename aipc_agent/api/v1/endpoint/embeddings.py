import tiktoken
from fastapi import APIRouter, Depends, status

from core.config import SETTINGS
from core.models import EMBEDDING_MODEL
from utils.rag import RAGEmbedding
from utils.protocol import EmbeddingCreateParams
from utils.request import check_api_key

embedding_router = APIRouter()

def get_embedding_engine():
    yield EMBEDDING_MODEL


