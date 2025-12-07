import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union
from utils.logger import logger
from config import TEXT_EMBEDDING_MODEL


class EmbeddingModels:

    def __init__(self):
        self.text_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._initialized = False

    def initialize(self):
        if self._initialized:
            logger.warning("Embedding models already initialized")
            return

        logger.info(f"Loading text embedding model: {TEXT_EMBEDDING_MODEL}")
        self.text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=self.device)

        self._initialized = True
        logger.success(f"Embedding models initialized on {self.device}")

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("Embedding models not initialized")

        if isinstance(text, str):
            text = [text]

        embeddings = self.text_model.encode(text, convert_to_numpy=True)

        if len(text) == 1:
            return embeddings[0]
        return embeddings

