import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

app = FastAPI(title="Sales Dashboard Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if client is None:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set in backend/.env")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": req.message}],
    )
    return ChatResponse(answer=completion.choices[0].message.content)
