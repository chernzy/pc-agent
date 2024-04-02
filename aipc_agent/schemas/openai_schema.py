from pydantic import BaseModel, Field
from typing import List, Optional, Dict

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