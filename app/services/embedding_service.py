from sentence_transformers import SentenceTransformer
import numpy as np
import os
from transformers import logging
from typing import Sequence

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    def generate_embedding(self, text: str)-> list[float]:
        if not text:
            return []
        embedding = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        return compute_semantic_similarity(vec1, vec2)


def compute_semantic_similarity(
    vec1: Sequence[float] | np.ndarray | None,
    vec2: Sequence[float] | np.ndarray | None,
) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    v1 = np.asarray(vec1, dtype=np.float32).reshape(-1)
    v2 = np.asarray(vec1, dtype=np.float32).reshape(-1)

    if v1.size == 0 or v2.size == 0:
        return 0.0
    if v1.shape != v2.shape:
        return 0.0
    
    return float(np.dot(v1,v2))