from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config import SETTINGS
from utils.compat import dictify

def create_app() -> FastAPI:
    import gc
    import torch

    def torch_gc() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @asynccontextmanager
    async def lifespan(app: "FastAPI"):
        yield
        torch_gc()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app


def create_rag_models():
    rag_models = []
    if "rag" in SETTINGS.tasks and SETTINGS.activate_inference:
        if SETTINGS.embedding_name:
            from utils.rag import RAGEmbedding
            rag_models.append(
                RAGEmbedding(SETTINGS.embedding_name, SETTINGS.embedding_device)
            )
        else:
            rag_models.append(None)

        if SETTINGS.rerank_name:
            from utils.rag import RAGReranker
            rag_models.append(RAGReranker(SETTINGS.rerank_name, device=SETTINGS.rerank_device))
        else:
            rag_models.append(None)

    return rag_models if len(rag_models) == 2 else [None, None]


def create_hf_llm():
    from core.default_engine import DefaultEngine
    from utils.loader import load_model_and_tokenizer

    include = {
        "device_map",
        "load_in_8bit",
        "load_in_4bit",
        "dtype",
        "rope_scaling",
        "flash_attn"
    }
    kwargs = dictify(SETTINGS, include=include)

    model, tokenizer = load_model_and_tokenizer(model_name_or_path=SETTINGS.model_path, **kwargs)

    logger.info("Using default engine")

    return DefaultEngine(
        model,
        tokenizer,
        model_name=SETTINGS.model_name,
        context_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        prompt_name=SETTINGS.chat_template,
        use_streamer_v2=SETTINGS.use_streamer_v2,
    )

app = create_app()

EMBEDDING_MODEL, RERANK_MODEL = create_rag_models()

LLM_ENGINE = create_hf_llm()

EXCLUDE_MODELS = ["baichuan-13b", "baichuan2-13b", "qwen", "chatglm3"]