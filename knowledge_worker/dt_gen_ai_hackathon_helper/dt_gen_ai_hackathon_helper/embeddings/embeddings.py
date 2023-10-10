from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings


def load_embeddings(persist_directory):
    # We use VertexAI embeddings model, however other models can be substituted here
    embeddings = VertexAIEmbeddings()

    # Creating embeddings with each re-run is highly inefficient and costly.
    # We instead aim to embed once, then load these embeddings from storage.
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vector_store