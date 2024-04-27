import traceback
from abc import ABC
from typing import (
    Optional,
    List,
    Union,
    Tuple,
    Dict,
    Iterator,
    Any,
)

import torch
from fastapi.responses import JSONResponse
from loguru import logger
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall
)
from openai.types.chat.chat_completion_message import FunctionCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.completion import Completion
from openai.types.completion_choice import CompletionChoice, Logprobs
from openai.types.completion_usage import CompletionUsage
from transformers import PreTrainedModel, PreTrainedTokenizer

