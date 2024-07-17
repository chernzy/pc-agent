from enum import Enum
from typing import (
    Optional,
    Dict,
    List,
    Union,
    Literal,
    Any,
    TypedDict
)

# from openai.types.chat import (
#     ChatCompletionMessageParam,
#     ChatCompletionToolChoiceOptionParam
# )

from schemas.openai_completion_params import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam
)

from openai.types.chat.completion_create_params import FunctionCall, ResponseFormat
from openai.types.create_embedding_response import Usage
from pydantic import BaseModel

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int

class ChatCompletionCreateParams(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = 0.
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = "none"
    tools: Optional[List] = None
    functions: Optional[List] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = False
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    temperature: Optional[float] = 0.9
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = 1.0
    user: Optional[str] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.03
    typical_p: Optional[float] = None
    watermark: Optional[bool] = False
    best_of: Optional[int] = 1
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = None
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    min_p: Optional[float] = 0.0
    include_stop_str_in_output: Optional[bool] = False
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None

class CompletionCreateParams(BaseModel):
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]], None]
    best_of: Optional[int] = 1
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 16
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.
    top_p: Optional[float] = 1.
    user: Optional[str] = None
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.03
    typical_p: Optional[float] = None
    watermark: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    use_beam_search: Optional[bool] = False
    stop_token_ids: Optional[List[int]] = None
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    min_p: Optional[float] = 0.0
    include_stop_str_in_output: Optional[bool] = False
    length_penalty: Optional[float] = 1.0
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    guided_regex: Optional[str] = None
    guided_choice: Optional[List[str]] = None
    guided_grammar: Optional[str] = None

class EmbeddingCreateParams(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

class Embedding(BaseModel):
    embedding: Any
    index: int
    object: Literal["embedding"]

class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]
    model: str
    object: Literal["list"]
    usage: Usage


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: List[str]
    top_n: Optional[int] = None
    return_documents: Optional[bool] = False


class Document(TypedDict):
    text: str


class DocumentObj(TypedDict):
    index: int
    relevance_score: float
    document: Optional[Document]


class RerankResponse(TypedDict):
    id: Optional[str]
    results: List[DocumentObj]