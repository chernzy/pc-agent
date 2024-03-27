from langchain_openai import OpenAI
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from dotenv import load_dotenv
from fastapi import FastAPI

from aipc_agent.custom_tools import get_wiki_tool, get_math_tool, get_word_problem_tool

load_dotenv()

app = FastAPI()

llm = OpenAI()

problem_chain = LLMMathChain.from_llm(llm=llm)

word_problem_template = """You are a reasoning agent tasked with solving 
the user's logic-based questions. Logically arrive at the solution, and be 
factual. In your answers, clearly detail the steps involved and give the 
final answer. Provide the response in bullet points. 
Question  {question} Answer"""

math_assistant_prompt = PromptTemplate(input_variables=["question"],
                                       template=word_problem_template
                                       )
word_problem_chain = LLMChain(llm=llm,
                              prompt=math_assistant_prompt)

wikipedia_tool = get_wiki_tool()
math_tool = get_math_tool(problem_chain)
word_problem_tool = get_word_problem_tool(word_problem_chain)


agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, word_problem_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

@app.post("/generate")
def generate():
    res = agent.invoke({"input": "I have 3 apples and 4 oranges. I give half of my oranges away and buy two dozen new ones, alongwith three packs of strawberries. Each pack of strawberry has 30 strawberries. How  many total pieces of fruit do I have at the end?"})
    return {"code": 200, "msg": res}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)