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


@embedding_router.post("/embeddings", status_code=status.HTTP_200_OK)
@embedding_router.post("/engines/{model_name}/embeddings")
async def create_embeddings(
    request: EmbeddingCreateParams,
    model_name: str = None,
    client: RAGEmbedding = Depends(get_embedding_engine)
):
    if request.model is None:
        request.model = model_name
    
    request.input = request.input
    if isinstance(request.input, str):
        request.input = [request.input]
    elif isinstance(request.input, list):
        if isinstance(request.input[0], int):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(request.input)]
        elif isinstance(request.input[0], list):
            decoding = tiktoken.model.encoding_for_model(request.model)
            request.input = [decoding.decode(text) for text in request.input]

    request.dimensions = request.dimensions or getattr(SETTINGS, "embedding_size", -1)
    print(client, " -------- embedding model ----------")
    return client.embed(
        texts=request.input,
        model=request.model,
        encoding_format=request.encoding_format,
        dimensions=request.dimensions
    )