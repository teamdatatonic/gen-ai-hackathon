from pathlib import Path
import chains
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    question: str
    chat_history: tuple


@app.on_event("startup")
async def startup_event():
    if not Path(chains.PERSIST_DIR).exists():
        chains.create_embeddings("datatonic.com")


@app.post("/query")
async def query_model(message: Message):
    try:
        response = chains.qa_with_sources_chain()(
            {
                "question": message.question,
                "chat_history": message.chat_history,
            }
        )
    except Exception as e:
        error_msg = f"Error querying model request, with following error: {e}"
        raise HTTPException(status_code=500, detail=error_msg)

    return response
