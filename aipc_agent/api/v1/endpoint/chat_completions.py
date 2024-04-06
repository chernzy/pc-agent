from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from queue import Queue
import uuid
import time
from loguru import logger
from sse_starlette.sse import EventSourceResponse

from core.llm import llm_gemma2b_local, llm_glm3_6b_local
from schemas.openai_schema import InputData, ChatCompletionResponse, ChatCompletionRequest, FunctionCallResponse, ChatMessage, UsageInfo,ChatCompletionResponseChoice
from utils.streamer import CustomStreamer
from utils.glm3_6b.generation import predict_stream, predict, generate_chatglm3
from utils.glm3_6b.utils import contains_custom_function, process_response, parse_output_text
from threading import Thread
import asyncio

router = APIRouter()

'''
TODO: 
different open source models have different input and response formats. modify according to the local llm
'''
tokenizer, model = llm_glm3_6b_local()

# ================================================== GEMMA 2B ==================================================
"""
tokenizer, model = llm_gemma2b_local()

def start_generation(input_text, streamer):
    input_ids = tokenizer([input_text], return_tensors="pt").to("cuda:0")
    generation_args = dict(input_ids, streamer=streamer,)
    thread = Thread(target=model.generate, kwargs=generation_args)
    thread.start()

async def response_generator(streamer_queue, input_text, streamer):
    start_generation(input_text, streamer)
    while True:
        value = streamer_queue.get()
        res_data = ChatCompletionResponse(
            choices=[{
                "message": {"content": value, "role": "assistant", "tool_calls": []},
                "finish_reason": "completed",
                "index": 0,
                "content_filter_results": None
            }],
            created=int(time.time()),
            id=str(uuid.uuid4()),
            model="gemma-2b-it",
            object="chat_completion",
            prompt_filter_results=[],
        )
        
        yield f"data: {res_data.json()}\n\n"
        if value == None:
            yield f"data: [DONE]\n\n"
            break
        streamer_queue.task_done()
        await asyncio.sleep(0.1)


@router.post("")
async def chat(data: InputData):
    try:
        input_text = data.messages[-1].content
        if data.stream:
            streamer_queue = Queue()
            streamer = CustomStreamer(streamer_queue, tokenizer, skip_prompt=True)
            return StreamingResponse(response_generator(streamer_queue, input_text, streamer), media_type='text/event-stream')
        else:
            input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")
            res = model.generate(
                **input_ids,
                temperature=data.temperature,
                top_p=data.top_p,
                # some models don't support n, presence_penalty, frequency_penalty, etc. params
                # n=data.n,
                # presence_penalty=data.presence_penalty,
                # frequency_penalty=data.frequency_penalty,
            )
            return ChatCompletionResponse(
                choices=[{
                    "message": {"content": tokenizer.decode(res[0]), "role": "assistant", "tool_calls": []},
                    "finish_reason": "completed",
                    "index": 0,
                    "content_filter_results": None
                }],
                created=int(time.time()),
                id=str(uuid.uuid4()),
                model="gemma-2b-it",
                object="chat_completion",
                prompt_filter_results=[],
            )
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
"""

@router.post("/chatglm3", response_model=ChatCompletionResponse)
async def chatglm3_completions(request: ChatCompletionRequest):
    global model, tokenizer
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
    )
    logger.debug(f"request \n {gen_params}")

    if request.stream:
        predict_stream_generator = predict_stream(request.model, gen_params)
        output = next(predict_stream_generator)
        if not contains_custom_function(output):
            raise EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        
        logger.debug(f"first result output: \n {output}")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response(output, use_tools=True)
            except:
                logger.warning("failed to parse tool call")
        
        if isinstance(function_call, dict):
            function_call = FunctionCallResponse(**function_call)
            tool_response = ""

            if not gen_params.get("messages"):
                gen_params["messages"] = []

            gen_params["messages"].append(ChatMessage(role="assistant", content=output))
            gen_params["messages"].append(ChatMessage(role="function", name=function_call.name, content=tool_response))

            generate = predict(request.model, gen_params)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            generate = parse_output_text(request.model, output)
            return EventSourceResponse(generate, media_type="text/event-stream")
        
    response = generate_chatglm3(model, tokenizer, gen_params)

    usage = UsageInfo()
    function_call, finish_reason = None, "stop"
    if request.tools:
        try:
            function_call = process_response(response["text"], use_tool=True)
        except:
            logger.warning("failed to parse tool call, maybe the response is not a tool call or have been answered")

    if isinstance(function_call, dict):
        finish_reason = "function_call"
        function_call = FunctionCallResponse(**function_call)

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None
    )

    logger.debug(f" message \n {message}")

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=request.model,
        id="",
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )



# =====================================================================================================
# sample of streaming output
# @router.post("/stream")
# async def stream_chat(data: InputData):
#     input_text = data.messages[-1].content
#     streamer_queue = Queue()
#     streamer = CustomStreamer(streamer_queue, tokenizer, skip_prompt=True)
#     return StreamingResponse(response_generator(streamer_queue, input_text, streamer), media_type='text/event-stream')