from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from queue import Queue
import uuid
import time

from core.llm import llm_gemma2b_local
from schemas.openai_schema import InputData, ChatCompletionResponse
from utils.streamer import CustomStreamer
from threading import Thread
import asyncio

router = APIRouter()

'''
TODO: 
different open source models have different input and response formats. modify according to the local llm
'''
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

# =====================================================================================================
# sample of streaming output
# @router.post("/stream")
# async def stream_chat(data: InputData):
#     input_text = data.messages[-1].content
#     streamer_queue = Queue()
#     streamer = CustomStreamer(streamer_queue, tokenizer, skip_prompt=True)
#     return StreamingResponse(response_generator(streamer_queue, input_text, streamer), media_type='text/event-stream')