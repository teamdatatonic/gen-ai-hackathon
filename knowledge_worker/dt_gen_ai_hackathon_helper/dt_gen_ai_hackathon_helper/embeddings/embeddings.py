from typing import List
from langchain.vectorstores import Chroma
from langchain.embeddings import VertexAIEmbeddings
from dt_gen_ai_hackathon_helper.rate_limiter import rate_limit


class CustomVertexAIEmbeddings(VertexAIEmbeddings):
    requests_per_minute: int
    num_instances_per_batch: int

    # Overriding embed_documents method
    def embed_documents(self, texts: List[str]):
        limiter = rate_limit(self.requests_per_minute)
        results = []
        docs = list(texts)

        while docs:
            # Working in batches because the API accepts maximum 5
            # documents per request to get embeddings
            head, docs = (
                docs[: self.num_instances_per_batch],
                docs[self.num_instances_per_batch :],
            )
            chunk = self.client.get_embeddings(head)
            results.extend(chunk)
            next(limiter)

        return [r.values for r in results]


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
