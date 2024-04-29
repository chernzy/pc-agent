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
async def create_chat_completion(
    request: ChatCompletionCreateParams,
    raw_request: Request,
    engine: DefaultEngine = Depends(get_engine)
):
    if (not request.messages) or request.messages[-1]["role"] == Role.ASSISTANT:
        raise HTTPException(status_code=400, detail="Invalid request")
    
    request = await handle_request(request, engine.stop)
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