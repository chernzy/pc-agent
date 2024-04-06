import torch
import gc
import time
from loguru import logger
from transformers import PreTrainedModel, PreTrainedTokenizer

from schemas.openai_schema import ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionResponse, FunctionCallResponse
from utils.glm3_6b.utils import process_chatglm_messages, InvalidScoreLogitsProcessor, apply_stopping_strings, process_response, contains_custom_function


async def predict(model_id: str, params: dict):
    global model, tokenizer
    choice_data = ChatCompletionResponseStreamChoice(
        index = 0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue
        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response(decoded_unicode, use_tool = True)
            except:
                logger.warning("Failed to parse tool call, maybe the response is not a tool call or have been answered")

        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index = 0,
            delta=delta,
            finish_reason=finish_reason,
        )

        chunk = ChatCompletionResponse(
            model=model_id,
            id="",
            choices=[choice_data],
            object="chat.completion.chunk"
        )

        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id="",
        choices=[choice_data],
        object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def predict_stream(model_id, gen_params):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    for new_response in generate_stream_chatglm3(model, tokenizer, gen_params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        if not is_function_call and len(output) > 7:
            is_function_call = contains_custom_function(output)

            if is_function_call:
                continue

            finish_reason = new_response["finish_reason"]

            if not has_send_first_chunk:
                message=DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data=ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=message,
                    finish_reason=finish_reason
                )
                chunk=ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk"
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))
            
            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message=DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data=ChatCompletionResponseStreamChoice(
                index=0,
                delta=message,
                finish_reason=finish_reason
            )
            chunk=ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk"
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    if is_function_call:
        yield output
    else:
        yield '[DONE]'


@torch.inference_mode()
def generate_stream_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    tools = params["tools"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    messages = process_chatglm_messages(messages, tools = tools)
    query, role = messages[-1]["content"], messages[-1]["role"]

    inputs = tokenizer.build_chat_input(query, history=messages[:-1], role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len >= model.config.seq_length:
        print(f"Input length larger than {model.config.seq_length}")
    
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }

    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len: -1]
        
        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            response, stop_found = apply_stopping_strings(response, ["<|observation|>"])

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": "function_call" if stop_found else None,
            }

            if stop_found:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()

def generate_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    for response in generate_stream_chatglm3(model, tokenizer, params):
        pass
    return response