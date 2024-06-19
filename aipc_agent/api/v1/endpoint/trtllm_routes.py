from functools import partial
from typing import Iterator

import gc
import torch

import os
import subprocess
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
from core.models import create_trtllm_engine
from models.model_management import ModelManagement
from utils.sqlite_utils import SqliteSqlalchemy
from utils.compat import dictify
from utils.protocol import ChatCompletionCreateParams, Role
from utils.request import (
    handle_request,
    check_api_key,
    get_event_publisher
)

chat_router = APIRouter(prefix="/chat")


engine = create_trtllm_engine()

print(engine, " ------- trt engine --------")

def get_engine():
    yield engine


@chat_router.post(
    "/completions",
    status_code=status.HTTP_200_OK
)
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
):
    global engine
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    request = await handle_request(request, {
            "strings": ["<reserved_106>", "<reserved_107>"],
            "token_ids": [195, 196],
        })
    request.max_tokens = request.max_tokens or 1024

    params = dictify(request, exclude={"messages"})
    params.update(dict(prompt_or_messages=request.messages, echo=False))
    logger.debug(f" ==== request ===== \n{params}")

    iterator_or_completion = await run_in_threadpool(engine.create_chat_completion, params)

    if isinstance(iterator_or_completion, Iterator):
        first_response = await run_in_threadpool(next, iterator_or_completion)

        def iterator() -> Iterator:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(
                get_event_publisher,
                request=raw_request,
                inner_send_chan=send_chan,
                iterator=iterator()
            )
        )
    else:
        return iterator_or_completion