import logging
from typing import Any, Dict, List, Mapping, Optional

from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from pydantic import BaseModel, root_validator
from vertexai.preview.language_models import TextEmbeddingModel, TextGenerationModel

BATCH_SIZE = 5


class GooglePalmEmbeddings(BaseModel, Embeddings):
    model_name: str = "textembedding-gecko@001"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates that the python package exists in environment."""
        
        values["client"] = TextEmbeddingModel.from_pretrained(
            values["model_name"])
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings.
        Args:
            texts: List[str] The list of strings to embed.
        Returns:
            List of embeddings, one for each text.
        """
        logging.info(
            "API calls restricted to 5 instances per call, batching documents to embed..."
        )
        texts_batched = [
            texts[i: i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)
        ]
        embeddings = [self.client.get_embeddings(x) for x in texts_batched]
        logging.info("Embeddings received!")
        return [el.values for batch in embeddings for el in batch]

    def embed_query(self, text: str) -> List[float]:
        """Embed a text.
        Args:
            text: The text to embed.
        Returns:
            Embedding for the text.
        """
        embeddings = self.client.get_embeddings([text])
        return embeddings[0].values


class GoogleLLM(LLM):
    
    # Model name options {text-bison-alpha, text-bison@001}
    model_name: str="text-bison@001"
    _llm = TextGenerationModel.from_pretrained(model_name)
    max_output_tokens:int = 256
    temperature:float = 0.3
    top_p:float = 0.8
    top_k:int = 40

    @property
    def _llm_type(self) -> str:
        """Return type of llm"""
        return "google"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        text = str(self._llm.predict(
            prompt, 
            max_output_tokens=self.max_output_tokens, 
            temperature=self.temperature, 
            top_p=self.top_p, 
            top_k=self.top_k,
        ))

        return text
