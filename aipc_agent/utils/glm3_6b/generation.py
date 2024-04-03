import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from schemas.openai_schema import ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse


async def predict(model_id: str, params: dict):
    choice_data = ChatCompletionResponseStreamChoice(
        index = 0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        pass

@torch.inference_mode()
def generate_stream_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    tools = params["tools"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    