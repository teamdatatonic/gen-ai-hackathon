from pathlib import Path
import chains
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Message(BaseModel):
    question: str
    chat_history: list


@app.on_event("startup")
async def startup_event():
    if not Path(chains.PERSIST_DIR).exists():
        chains.create_vector_store()


@app.post("/query")
async def query_model(message: Message):
    try:
        # map history (list of lists) to expected format of chat_history (list of tuples)
        chat_history = map(tuple, message.chat_history)

        response = chains.qa_with_sources_chain()(
            {
                "question": message.question,
                "chat_history": chat_history,
            }
        )
    except Exception as e:
        error_msg = f"Error querying model request, with following error: {e}"
        raise HTTPException(status_code=500, detail=error_msg)

    return response
