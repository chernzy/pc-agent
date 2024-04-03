from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Union

class Message(BaseModel):
    content: str
    role: str

class Parameter(BaseModel):
    type: str
    required: Optional[List[str]]
    properties: dict

class Function(BaseModel):
    name: str
    description: str
    parameters: Parameter

class Tool(BaseModel):
    function: Function
    type: str

class InputData(BaseModel):
    messages: List[Message]
    temperature: float = Field(0.1, alias="temperature")
    top_p: int = Field(1, alias="top_p")
    n: int = Field(1, alias="n")
    presence_penalty: float = Field(0, alias="presence_penalty")
    frequency_penalty: float = Field(0, alias="frequency_penalty")
    stream: bool = Field(False, alias="stream")
    model: str
    tools: List[Tool]
    tool_choice: str

class ContentFilterResults(BaseModel):
    pass

class MessageResponse(BaseModel):
    content: Optional[str]
    role: str
    tool_calls: List[Dict[str, str]]

class Choices(BaseModel):
    content_filter_results: Optional[ContentFilterResults]
    finish_reason: Optional[str]
    index: Optional[int]
    message: MessageResponse

class ContentFilterResultsInner(BaseModel):
    filtered: bool
    severity: str

class PromptFilterResults(BaseModel):
    prompt_index: int
    content_filter_results: Dict[str, ContentFilterResultsInner]

class ChatCompletionResponse(BaseModel):
    choices: List[Choices]
    created: int
    id: str
    model: str
    object: str
    prompt_filter_results: List[PromptFilterResults]
    system_fingerprint: Optional[str] = None
    usage: Optional[Dict[str, int]] = None

# ==================================== chatglm3 6b ===============================================
class FunctionCallResponse(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None    

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "function"]
    content: str = None
    name: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCallResponse] = None

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length", "function_call"]]
    index: int


class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None