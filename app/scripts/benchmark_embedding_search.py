import argparse
import ast
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable
import numpy as np
import pandas as pd
from app.services.embedding_service import EmbeddingService
from app.services.matching_service import MatchingService

logger = logging.getLogger(__name__)

DEFAULT_DATASET_CANDIDATES = [
    Path("app/Datasets/French jobs dataset.csv"),
    Path("app/Datasets/french_jobs_dataset.csv"),
    Path(r"C:\Users\zarro\OneDrive\Desktop\French jobs dataset.csv"),
]


@dataclass
class BenchmarkResult:
    size: int
    end_to_end_avg_ms: float
    end_to_end_p95_ms: float
    precompute_build_s: float
    precomputed_avg_ms: float
    precomputed_p95_ms: float
    speedup: float


def _safe_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_listish(value) -> list[str]:
    text = _safe_text(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    text = text.strip("[]")
    items = [token.strip(" '\"") for token in text.split(",")]
    return [item for item in items if item]


def _build_candidate(row: pd.Series) -> dict:
    title = _safe_text(row.get("intitule"))
    skills = _parse_listish(row.get("SAVOIR_FAIRE"))
    summary_parts = [
        _safe_text(row.get("description")),
        _safe_text(row.get("profil")),
        _safe_text(row.get("entreprise_description")),
    ]
    summary = " ".join(part for part in summary_parts if part)

    return {
        "title": title,
        "skills": skills,
        "summary": summary,
    }


def _candidate_to_text(candidate: dict) -> str:
    title = (candidate.get("title") or "").strip()
    skills_list = candidate.get("skills") or []
    skills = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)
    summary = (candidate.get("summary") or "").strip()

    if not (title or skills or summary):
        return "empty profile"
    return f"Role: {title}. Skills: {skills}. Summary: {summary}"


def _read_dataset(path: Path) -> list[dict]:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    candidates: list[dict] = []
    for _, row in df.iterrows():
        candidate = _build_candidate(row)
        if candidate["title"] or candidate["skills"] or candidate["summary"]:
            candidates.append(candidate)
    return candidates


def _resolve_dataset_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No default dataset file found. Pass --dataset-path explicitly."
    )


def _build_queries(candidates: list[dict], requested: Iterable[str], query_count: int) -> list[str]:
    explicit = [query.strip() for query in requested if query.strip()]
    if explicit:
        return explicit

    titles = []
    seen = set()
    for candidate in candidates:
        title = (candidate.get("title") or "").strip()
        if not title:
            continue
        lowered = title.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        titles.append(title)
        if len(titles) >= query_count:
            break

    return titles


def _latency_stats(latencies: list[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0
    avg_ms = (sum(latencies) / len(latencies)) * 1000
    p95_ms = statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else max(latencies) * 1000
    return round(avg_ms, 2), round(p95_ms, 2)


def _benchmark_end_to_end(
    matching_service: MatchingService,
    queries: list[str],
    candidates: list[dict],
    threshold: float,
) -> tuple[float, float]:
    latencies = []
    for query in queries:
        start = perf_counter()
        matching_service.rank_candidates(job_text=query, candidates=candidates, threshold=threshold)
        latencies.append(perf_counter() - start)
    return _latency_stats(latencies)


def _benchmark_precomputed(
    embedding_service: EmbeddingService,
    queries: list[str],
    candidate_texts: list[str],
    top_k: int,
    batch_size: int,
) -> tuple[float, float, float]:
    build_start = perf_counter()
    candidate_embeddings = embedding_service.model.encode(
        candidate_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    build_time = perf_counter() - build_start

    latencies = []
    for query in queries:
        start = perf_counter()
        query_embedding = np.array(embedding_service.generate_embedding(query), dtype=np.float32)
        if query_embedding.size == 0:
            latencies.append(perf_counter() - start)
            continue

        similarities = candidate_embeddings @ query_embedding
        np.argsort(similarities)[::-1][:top_k]
        latencies.append(perf_counter() - start)

    avg_ms, p95_ms = _latency_stats(latencies)
    return round(build_time, 3), avg_ms, p95_ms


def run_benchmark(
    dataset_path: Path,
    all_candidates: list[dict],
    sizes: list[int],
    queries: list[str],
    threshold: float,
    top_k: int,
    batch_size: int,
    expand_dataset: bool,
) -> list[BenchmarkResult]:
    logger.info("Loaded %d candidate rows from %s", len(all_candidates), dataset_path)

    if not all_candidates:
        raise RuntimeError("No usable rows were parsed from the dataset.")

    matching_service = MatchingService()
    embedding_service = matching_service.embedding_service

    logger.info("Warm-up embedding model")
    embedding_service.generate_embedding("warmup query")

    results: list[BenchmarkResult] = []
    for size in sizes:
        if size > len(all_candidates):
            if not expand_dataset:
                logger.warning("Requested size %d exceeds dataset rows %d, skipping", size, len(all_candidates))
                continue
            logger.info(
                "Requested size %d exceeds dataset rows %d; using synthetic expansion.",
                size,
                len(all_candidates),
            )
            repeat = (size // len(all_candidates)) + 1
            subset = (all_candidates * repeat)[:size]
        else:
            subset = all_candidates[:size]
        candidate_texts = [_candidate_to_text(candidate) for candidate in subset]

        e2e_avg, e2e_p95 = _benchmark_end_to_end(
            matching_service=matching_service,
            queries=queries,
            candidates=subset,
            threshold=threshold,
        )
        build_s, pre_avg, pre_p95 = _benchmark_precomputed(
            embedding_service=embedding_service,
            queries=queries,
            candidate_texts=candidate_texts,
            top_k=top_k,
            batch_size=batch_size,
        )

        speedup = round((e2e_avg / pre_avg), 2) if pre_avg > 0 else 0.0
        results.append(
            BenchmarkResult(
                size=size,
                end_to_end_avg_ms=e2e_avg,
                end_to_end_p95_ms=e2e_p95,
                precompute_build_s=build_s,
                precomputed_avg_ms=pre_avg,
                precomputed_p95_ms=pre_p95,
                speedup=speedup,
            )
        )
    return results


def _parse_sizes(raw: str) -> list[int]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return sorted(set(values))


def _print_results(results: list[BenchmarkResult]) -> None:
    if not results:
        print("No benchmark results produced.")
        return

    headers = [
        "size",
        "e2e_avg_ms",
        "e2e_p95_ms",
        "build_s",
        "pre_avg_ms",
        "pre_p95_ms",
        "speedup_x",
    ]
    print(" | ".join(headers))
    print("-" * 86)
    for result in results:
        row = [
            str(result.size),
            f"{result.end_to_end_avg_ms:.2f}",
            f"{result.end_to_end_p95_ms:.2f}",
            f"{result.precompute_build_s:.3f}",
            f"{result.precomputed_avg_ms:.2f}",
            f"{result.precomputed_p95_ms:.2f}",
            f"{result.speedup:.2f}",
        ]
        print(" | ".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark EmbeddingService and search scaling on the French jobs dataset."
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the CSV dataset.")
    parser.add_argument(
        "--sizes",
        type=str,
        default="100,500,1000,2500,5000,9000",
        help="Comma-separated dataset sizes to benchmark.",
    )
    parser.add_argument(
        "--query",
        dest="queries",
        action="append",
        default=[],
        help="Repeatable query text. If omitted, queries are sampled from job titles.",
    )
    parser.add_argument("--query-count", type=int, default=10, help="Number of sampled queries.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Score threshold for rank_candidates.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K slice for precomputed search.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding encode.")
    parser.add_argument("--out-csv", type=str, default=None, help="Optional output CSV report path.")
    parser.add_argument(
        "--no-expand-dataset",
        action="store_true",
        help="Skip sizes greater than dataset rows instead of synthetic expansion.",
    )
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset_path)
    sizes = _parse_sizes(args.sizes)
    candidates = _read_dataset(dataset_path)
    queries = _build_queries(candidates, args.queries, args.query_count)

    if not queries:
        raise RuntimeError("No queries available. Provide --query values explicitly.")

    results = run_benchmark(
        dataset_path=dataset_path,
        all_candidates=candidates,
        sizes=sizes,
        queries=queries,
        threshold=args.threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
        expand_dataset=(not args.no_expand_dataset),
    )
    _print_results(results)

    if args.out_csv:
        df = pd.DataFrame([result.__dict__ for result in results])
        df.to_csv(args.out_csv, index=False)
        logger.info("Saved benchmark report: %s", args.out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    main()
