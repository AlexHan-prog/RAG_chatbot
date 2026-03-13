from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from src.rag_chatbot.rag.RAG_bot import chat_loop

router = APIRouter()

class ChatIn(BaseModel):
    message: str

@router.post("/chat")
async def chat(chat_in: ChatIn):
    response = await chat_loop(chat_in.message)
    return response