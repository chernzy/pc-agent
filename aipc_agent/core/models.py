from contextlib import asynccontextmanager
import gc
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from core.config import SETTINGS
from utils.compat import dictify

model = None
tokenizer = None

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


def create_hf_llm(model_name: str = None, model_path: str = None, lora_path: str = None):
    from core.default_engine import DefaultEngine
    from utils.loader import load_model_and_tokenizer, load_lora_model_and_tokenizer
    # from utils.sqlite_utils import SqliteSqlalchemy
    # from models.model_management import ModelManagement

    # session = SqliteSqlalchemy().session

    # Debug mode
    # result = session.query(ModelManagement).filter_by(ID=1).first()

    # Production mode
    '''
    all_models = session.query(ModelManagement).all()
    result = None
    for i in all_models:
        if i.STATUS == 1:
            result = i
    '''


    include = {
        "device",
        "load_in_8bit",
        "load_in_4bit",
        "dtype",
        "rope_scaling",
        "flash_attn"
    }
    kwargs = dictify(SETTINGS, include=include)
    global model, tokenizer


    if SETTINGS.lora_path:
        model, tokenizer = load_lora_model_and_tokenizer(model_name_or_path=SETTINGS.model_path, **kwargs)
        logger.info("Using LoRA model")
    else:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path=SETTINGS.model_path, **kwargs)
    
    '''
    if lora_path:
        
        model, tokenizer = load_lora_model_and_tokenizer(model_name_or_path=result.PATH, **kwargs)
        logger.info("Using LoRA model")
    else:
        model, tokenizer = load_model_and_tokenizer(model_name_or_path=result.PATH, **kwargs)
    '''
    
    logger.info("Using default engine")

    return DefaultEngine(
        model,
        tokenizer,
        model_name=SETTINGS.model_name,
        context_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,
        prompt_name=SETTINGS.chat_template,
        use_streamer_v2=SETTINGS.use_streamer_v2,
    )


def create_trtllm_engine():
    # try:
    #     from core.trt_llm_engine import TrtLLMEngine
    # except ImportError:
    #     return None
    from core.trt_llm_engine import TrtLLMEngine
    
    return TrtLLMEngine(
        model_path=SETTINGS.trt_engine_path,
        engine_name=SETTINGS.trt_engine_name,
        tokenizer_dir=SETTINGS.tokenizer_dir_path,
        temperature=0.1,
        max_new_tokens=SETTINGS.max_output_tokens,
        context_window=SETTINGS.max_input_tokens,
        use_py_session=True,
        add_special_tokens=False,
        trtLlm_debug_mode=False,
        verbose=False
    )



def del_model():
    global model, tokenizer
    if 'model' in globals():
        del model
        del tokenizer
    model = None
    tokenizer = None
    with torch.no_grad():
        torch.cuda.empty_cache()
        gc.collect()
'''
def init_sqlite():
    from utils.sqlite_utils import SqliteSqlalchemy
    from models.model_management import ModelManagement

    session = SqliteSqlalchemy().session
    model_db = ModelManagement(ID=1, NAME=SETTINGS.model_name, PATH=SETTINGS.model_path)
    # Debug Mode
    default_model = session.query(ModelManagement).filter_by(ID=1).first()
    if default_model is None:
        session.add(model_db)
        session.commit()
        session.close()

init_sqlite()
'''
app = create_app()

EMBEDDING_MODEL, RERANK_MODEL = create_rag_models()

EXCLUDE_MODELS = ["baichuan-13b", "baichuan2-13b", "qwen", "chatglm3"]