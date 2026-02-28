import numpy as np
from app.services.embedding_service import EmbeddingService

class MatchingService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    def rank_candidates(self, job_text: str, candidates: list[dict], threshold: float = 0.0) -> list[dict]:
        if not candidates or not job_text.strip():
            return []
        job_embedding = np.array(
            self.embedding_service.generate_embedding(job_text), 
            dtype=np.float32
        )
        norm = np.linalg.norm(job_embedding)
        if norm > 0:
            job_embedding = job_embedding / norm

        candidate_texts = [self._build_candidate_text(c) for c in candidates]

        try:
            candidate_embeddings = self.embedding_service.model.encode(
                candidate_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
        similarity_scores = candidate_embeddings @ job_embedding

        ranked = [
            {"candidate": c, "score": float(s)}
            for c, s in zip(candidates, similarity_scores)
            if s >= threshold
        ]

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked

    def _build_candidate_text(self, candidate: dict) -> str:
        title = (candidate.get('title') or '').strip()
        skills_list = candidate.get('skills') or []
        skills = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)
        summary = (candidate.get('summary') or '').strip()
        
        if not (title or skills or summary):
            return "empty profile"
            
        return f"Role: {title}. Skills: {skills}. Summary: {summary}"