from fastapi import APIRouter

from agent.custom_agent import custom_math_agent
from core.llm import llm_openai

router = APIRouter()

@router.post("")
async def agent_completions():
    llm = llm_openai()
    agent = custom_math_agent(llm)
    res = agent.invoke({
        	"input": "I have 3 apples and 4 oranges. I give half of my oranges away and buy two dozen new ones, alongwith three packs of strawberries. Each pack of strawberry has 30 strawberries. How  many total pieces of fruit do I have at the end?"
        })
    return {"code": 200, "msg": res}