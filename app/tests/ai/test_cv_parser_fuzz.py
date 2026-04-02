import random
import string

import pytest

from app.schemas.candidate import CandidateUploadRespose
from app.services import cv_parser


KNOWN_SKILLS = ["python", "sql", "docker", "java", "react"]


def _assert_contract(payload: dict) -> None:
    CandidateUploadRespose(**payload)
    assert isinstance(payload["errors"], list)
    assert isinstance(payload["warnings"], list)
    assert isinstance(payload["skills"], list)
    assert isinstance(payload["extracted_skills"], list)


def _random_malformed_text(rng: random.Random, max_len: int = 500) -> str:
    chunks = [
        string.ascii_letters,
        string.digits,
        string.punctuation,
        " \n\t\r",
        "\x00\x01\x02",
        "éèêàç",
        "ΑΒΓΔ",
        "漢字かな",
        "🧪🤖",
    ]
    length = rng.randint(0, max_len)
    out = []
    for _ in range(length):
        pool = chunks[rng.randrange(len(chunks))]
        out.append(pool[rng.randrange(len(pool))])
    return "".join(out)


def test_parse_cv_safe_fuzz_malformed_text_never_crashes(monkeypatch):
    rng = random.Random(1337)
    for idx in range(200):
        text = _random_malformed_text(rng)
        monkeypatch.setattr(cv_parser, "extract_text", lambda *_args, _t=text: _t)
        try:
            payload = cv_parser.parse_cv_safe(
                file_bytes=b"fuzz",
                filename=f"fuzz_{idx}.pdf",
                known_skills=KNOWN_SKILLS,
                min_confidence=0.6,
                use_semantic=False,
            )
        except Exception as exc:  # pragma: no cover - failure guard
            pytest.fail(f"Crash on malformed text case {idx}: {exc!r}")
        _assert_contract(payload)


def test_parse_cv_safe_fuzz_random_binary_pdf_never_crashes():
    rng = random.Random(2026)
    for idx in range(100):
        payload_bytes = b"%PDF-1.7\n" + bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 1024)))
        try:
            payload = cv_parser.parse_cv_safe(
                file_bytes=payload_bytes,
                filename=f"binary_{idx}.pdf",
                known_skills=KNOWN_SKILLS,
                min_confidence=0.6,
                use_semantic=False,
            )
        except Exception as exc:  # pragma: no cover - failure guard
            pytest.fail(f"Crash on binary case {idx}: {exc!r}")
        _assert_contract(payload)
