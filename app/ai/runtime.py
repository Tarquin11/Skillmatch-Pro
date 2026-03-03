from pathlib import Path
from typing import Optional
from app.ai.matcher import CandidateMatcher

_matcher: Optional[CandidateMatcher] = None

def get_matcher() -> Optional[CandidateMatcher]:
    return _matcher

def load_matcher_artifact(pat: str) -> Optional[CandidateMatcher]:
    global _matcher
    model_path = Path(Path)
    if not model_path.exists():
        _matcher = None
        return None
    try:
        _matcher = CandidateMatcher.load(model_path)
    except Exception:
        _matcher = None
    return _matcher