from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from agent import answer, make_client

from retriever import PFRRetriever

app = FastAPI(title="PFR Agent API", version="1.0")
retriever = PFRRetriever(Path(__file__).parent / "data")
client = make_client()
model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {
        "ok": True,
        "chunks": len(retriever.documents),
        "firms": retriever.firms(),
        "ai_enabled": client is not None,
    }

@app.post("/chat")
def chat(request: ChatRequest):
    result = answer(request.message, retriever, client, model)
    return {
        "answer": result["answer"],
        "sources": [
            {"firm": s["firm"], "title": s["title"], "url": s["url"]}
            for s in result["sources"]
        ],
    }
