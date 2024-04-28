from functools import partial
from typing import Iterator

import anyio
from fastapi import (
    APIRouter,
    Depends,
    Request,
    HTTPException,
    status,
)
from loguru import logger
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from core.default_engine import DefaultEngine
from core.models import LLM_ENGINE
from utils.compat import dictify
from utils.protocol import ChatCompletionCreateParams, Role
from utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher
)

chat_router = APIRouter(prefix="/chat")

def get_engine():
    yield LLM_ENGINE


@chat_router.post(
    "/completions",
    status_code=status.HTTP_200_OK
)
