"""
api/v1/routes/chat.py — POST /api/v1/chat

AI-powered scam detection chat endpoint backed by GPT-4o vision.
Accepts plain text or a base64-encoded image alongside the message.
"""

import os

import openai
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    image_base64: str | None = None


class ChatResponse(BaseModel):
    reply: str


_SYSTEM_PROMPT = (
    "You are a scam detection assistant. Help users determine if a person, message, "
    "or situation is a scam. If the user sends a photo, analyze it for signs of fraud: "
    "fake documents, suspicious profiles, fraudulent offers. Answer clearly and concisely."
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="AI scam-detection chat (text + optional image)",
    tags=["AI Chat"],
)
async def chat(request: ChatRequest) -> ChatResponse:
    client = openai.OpenAI(api_key=os.environ["CHATGPT"])

    messages: list = [{"role": "system", "content": _SYSTEM_PROMPT}]

    if request.image_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": request.message},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{request.image_base64}"
                }},
            ],
        })
    else:
        messages.append({"role": "user", "content": request.message})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI service error",
        )

    return ChatResponse(reply=response.choices[0].message.content)
