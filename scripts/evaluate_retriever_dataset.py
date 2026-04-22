from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from core.llm import LLMConfig
from core.pipeline_v4 import UnetiDocumentAgentV4Max
from scripts.build_retriever_dataset import _retrieve_candidates


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _recall_at(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(1 for row_id in ranked_ids[:k] if row_id in relevant_ids)
    return hits / len(relevant_ids)


def _mrr_at(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    for idx, row_id in enumerate(ranked_ids[:k], start=1):
        if row_id in relevant_ids:
            return 1.0 / idx
    return 0.0


def _ndcg_at(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for idx, row_id in enumerate(ranked_ids[:k], start=1):
        if row_id in relevant_ids:
            dcg += 1.0 / math.log2(idx + 1.0)
    ideal_hits = min(k, len(relevant_ids))
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _metrics_for_queries(results: List[Dict[str, Any]], k_values: Sequence[int]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f'recall@{k}'] = round(_mean([float(item.get(f'recall@{k}', 0.0) or 0.0) for item in results]), 6)
    max_k = max(k_values) if k_values else 10
    metrics[f'mrr@{max_k}'] = round(_mean([float(item.get(f'mrr@{max_k}', 0.0) or 0.0) for item in results]), 6)
    metrics[f'ndcg@{max_k}'] = round(_mean([float(item.get(f'ndcg@{max_k}', 0.0) or 0.0) for item in results]), 6)
    return metrics


def evaluate_retriever_dataset(
    dataset_name: str = 'retriever_unified_v3',
    split: str = 'val',
    top_k: int = 20,
    k_values: Sequence[int] | None = None,
) -> Dict[str, Any]:
    dataset_dir = RLHF_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_dir}')

    queries = [
        row for row in _read_jsonl(dataset_dir / 'queries.jsonl')
        if str(row.get('split', '') or '') == split
    ]
    qrels = [
        row for row in _read_jsonl(dataset_dir / 'qrels.jsonl')
        if str(row.get('split', '') or '') == split and int(row.get('label', 0) or 0) > 0
    ]
    qrels_by_query: Dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        qrels_by_query[str(row.get('query_id', '') or '')].add(str(row.get('corpus_id', '') or ''))

    k_values = sorted({int(k) for k in (k_values or [1, 3, 5, 10]) if int(k) > 0})
    max_metric_k = max(k_values) if k_values else 10
    top_k = max(top_k, max_metric_k)

    agent = UnetiDocumentAgentV4Max(LLMConfig(enabled=False))
    per_query: List[Dict[str, Any]] = []

    for row in queries:
        query_id = str(row.get('query_id', '') or '')
        relevant_ids = set(qrels_by_query.get(query_id, set()))
        if not relevant_ids:
            continue
        candidates = _retrieve_candidates(
            agent=agent,
            query=str(row.get('query', '') or ''),
            domain=str(row.get('domain', '') or ''),
            max_candidates=top_k,
        )
        ranked_ids = [str(getattr(item, 'source_id', '') or '') for item in candidates]
        result = {
            'query_id': query_id,
            'query': str(row.get('query', '') or ''),
            'origin': str(row.get('origin', '') or ''),
            'domain': str(row.get('domain', '') or ''),
            'query_type': str(row.get('query_type', '') or ''),
            'quality': float(row.get('quality', 1.0) or 1.0),
            'relevant_count': len(relevant_ids),
            'top_ids': ranked_ids[:max_metric_k],
        }
        for k in k_values:
            result[f'recall@{k}'] = round(_recall_at(ranked_ids, relevant_ids, k), 6)
        result[f'mrr@{max_metric_k}'] = round(_mrr_at(ranked_ids, relevant_ids, max_metric_k), 6)
        result[f'ndcg@{max_metric_k}'] = round(_ndcg_at(ranked_ids, relevant_ids, max_metric_k), 6)
        per_query.append(result)

    overall = _metrics_for_queries(per_query, k_values)
    by_origin: Dict[str, Dict[str, float]] = {}
    by_type: Dict[str, Dict[str, float]] = {}
    by_domain: Dict[str, Dict[str, float]] = {}
    for key in sorted({str(item.get('origin', '') or '') for item in per_query}):
        by_origin[key] = _metrics_for_queries([item for item in per_query if str(item.get('origin', '') or '') == key], k_values)
    for key in sorted({str(item.get('query_type', '') or '') for item in per_query}):
        by_type[key] = _metrics_for_queries([item for item in per_query if str(item.get('query_type', '') or '') == key], k_values)
    for key in sorted({str(item.get('domain', '') or '') for item in per_query}):
        by_domain[key] = _metrics_for_queries([item for item in per_query if str(item.get('domain', '') or '') == key], k_values)

    summary = {
        'dataset_name': dataset_name,
        'split': split,
        'top_k': int(top_k),
        'k_values': list(k_values),
        'counts': {
            'queries': len(queries),
            'evaluated_queries': len(per_query),
            'qrels': len(qrels),
        },
        'overall': overall,
        'by_origin': by_origin,
        'by_type': by_type,
        'by_domain': by_domain,
        'query_origins': dict(Counter(str(item.get('origin', '') or '') for item in per_query)),
    }
    out_path = dataset_dir / f'eval_hybrid_{split}.json'
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate current hybrid retrieval against unified retriever dataset.')
    parser.add_argument('--dataset-name', default='retriever_unified_v3')
    parser.add_argument('--split', default='val')
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--k-values', nargs='*', type=int, default=[1, 3, 5, 10])
    args = parser.parse_args()

    summary = evaluate_retriever_dataset(
        dataset_name=str(args.dataset_name),
        split=str(args.split),
        top_k=max(1, int(args.top_k)),
        k_values=[int(k) for k in args.k_values],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
