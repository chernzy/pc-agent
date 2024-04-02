from fastapi import APIRouter

from core.llm import llm_gemma2b_local

router = APIRouter()

tokenizer, model = llm_gemma2b_local()

@router.post("")
async def chat():
    input_text = "Hello world"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:0")
    res = model.generate(**input_ids)
    return {"code": 200, "msg": tokenizer.decode(res[0])}