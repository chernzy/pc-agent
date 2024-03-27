from dotenv import load_dotenv
from fastapi import FastAPI

from api.v1.api import api_router

load_dotenv()

app = FastAPI()

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)