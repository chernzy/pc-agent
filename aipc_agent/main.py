from dotenv import load_dotenv
from fastapi import FastAPI

from api.v1.api import api_router
from core.llm import llm_gemma2b_local

load_dotenv()

app = FastAPI()

app.include_router(api_router, prefix="/api/v1")
