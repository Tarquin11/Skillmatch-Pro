from sentence_transformers import SentenceTransformer
import numpy as np
import os
from transformers import logging

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
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        if len(v1) == 0 or len(v2) == 0:
            return 0.0
        
        return float(np.dot(v1, v2))
    
def compute_semantic_similarity(vec1, vec2):
        if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float (np.dot(v1, v2))
    
embedding_service = EmbeddingService() 
