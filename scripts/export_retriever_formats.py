from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR


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


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in materialized), encoding='utf-8')
    return len(materialized)


def _write_tsv(path: Path, header: str, rows: Iterable[str]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [header, *materialized] if header else materialized
    path.write_text('\n'.join(payload), encoding='utf-8')
    return len(materialized)


def _split_filter(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [row for row in rows if str(row.get('split', '') or '') == split]


def _difficulty_from_negative_scores(scores: Sequence[float]) -> float:
    values = [float(score) for score in scores if float(score) > 0.0]
    if not values:
        return 0.5
    avg_score = sum(values) / len(values)
    return round(min(1.0, max(0.0, math.log1p(avg_score) / math.log(4.0))), 6)


def _sample_weight(query_quality: float, negative_scores: Sequence[float], origin: str) -> float:
    difficulty = _difficulty_from_negative_scores(negative_scores)
    origin_boost = 1.0 if origin == 'seed_qa' else 0.92
    weight = max(0.15, min(1.25, float(query_quality) * (0.75 + 0.5 * difficulty) * origin_boost))
    return round(weight, 6)


def export_retriever_formats(
    dataset_name: str = 'retriever_unified_v3',
    export_name: str = '',
) -> Dict[str, Any]:
    dataset_dir = RLHF_DIR / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_dir}')

    corpus_rows = _read_jsonl(dataset_dir / 'corpus.jsonl')
    query_rows = _read_jsonl(dataset_dir / 'queries.jsonl')
    triplet_rows = _read_jsonl(dataset_dir / 'embedding_triplets.jsonl')
    pair_rows = _read_jsonl(dataset_dir / 'reranker_pairs.jsonl')
    qrel_rows = _read_jsonl(dataset_dir / 'qrels.jsonl')
    corpus_by_id = {str(row.get('id', '') or ''): row for row in corpus_rows}
    query_by_id = {str(row.get('query_id', '') or ''): row for row in query_rows}

    export_root = RLHF_DIR / 'exports' / (export_name.strip() or dataset_name)
    export_root.mkdir(parents=True, exist_ok=True)

    # BEIR-style export
    beir_root = export_root / 'beir'
    beir_corpus = [
        {
            '_id': str(row.get('id', '') or ''),
            'title': str(row.get('title', '') or ''),
            'text': str(row.get('text', '') or ''),
            'metadata': {
                'domain': str(row.get('domain', '') or ''),
                'chunk_type': str(row.get('chunk_type', '') or ''),
                'entity_name': str(row.get('entity_name', '') or ''),
                'source_kind': str(row.get('source_kind', '') or ''),
            },
        }
        for row in corpus_rows
    ]
    beir_queries = [
        {
            '_id': str(row.get('query_id', '') or ''),
            'text': str(row.get('query', '') or ''),
            'metadata': {
                'domain': str(row.get('domain', '') or ''),
                'origin': str(row.get('origin', '') or ''),
                'query_type': str(row.get('query_type', '') or ''),
                'quality': float(row.get('quality', 1.0) or 1.0),
                'split': str(row.get('split', '') or ''),
            },
        }
        for row in query_rows
    ]
    _write_jsonl(beir_root / 'corpus.jsonl', beir_corpus)
    _write_jsonl(beir_root / 'queries.jsonl', beir_queries)
    for split in ['train', 'val']:
        split_qrels = [row for row in qrel_rows if str(row.get('split', '') or '') == split]
        _write_tsv(
            beir_root / f'qrels_{split}.tsv',
            'query-id\tcorpus-id\tscore',
            (f"{row.get('query_id', '')}\t{row.get('corpus_id', '')}\t{row.get('label', 0)}" for row in split_qrels),
        )

    # Sentence-transformers bi-encoder export
    st_root = export_root / 'sentence_transformers'
    bi_encoder_rows: List[Dict[str, Any]] = []
    for row in query_rows:
        negatives = list(row.get('hard_negatives', []) or [])
        negative_ids = [str(item.get('corpus_id', '') or '') for item in negatives if str(item.get('corpus_id', '') or '') in corpus_by_id]
        negative_scores = [float(item.get('score', 0.0) or 0.0) for item in negatives]
        if not negative_ids:
            continue
        positives = [pid for pid in row.get('positives', []) or [] if pid in corpus_by_id]
        if not positives:
            continue
        bi_encoder_rows.append({
            'query_id': str(row.get('query_id', '') or ''),
            'query': str(row.get('query', '') or ''),
            'query_type': str(row.get('query_type', '') or ''),
            'query_quality': float(row.get('quality', 1.0) or 1.0),
            'origin': str(row.get('origin', '') or ''),
            'domain': str(row.get('domain', '') or ''),
            'positive_ids': positives,
            'positives': [str(corpus_by_id[pid].get('text', '') or '') for pid in positives],
            'negative_ids': negative_ids,
            'negatives': [str(corpus_by_id[nid].get('text', '') or '') for nid in negative_ids],
            'sample_weight': _sample_weight(float(row.get('quality', 1.0) or 1.0), negative_scores, str(row.get('origin', '') or '')),
            'difficulty': _difficulty_from_negative_scores(negative_scores),
            'split': str(row.get('split', '') or ''),
        })
    _write_jsonl(st_root / 'bi_encoder_train.jsonl', _split_filter(bi_encoder_rows, 'train'))
    _write_jsonl(st_root / 'bi_encoder_val.jsonl', _split_filter(bi_encoder_rows, 'val'))
    _write_jsonl(st_root / 'triplets_train.jsonl', _split_filter(triplet_rows, 'train'))
    _write_jsonl(st_root / 'triplets_val.jsonl', _split_filter(triplet_rows, 'val'))

    # Cross-encoder export
    ce_root = export_root / 'cross_encoder'
    cross_rows: List[Dict[str, Any]] = []
    for row in pair_rows:
        query = query_by_id.get(str(row.get('query_id', '') or ''), {})
        negative_score = float(row.get('negative_score', 0.0) or 0.0)
        weight = _sample_weight(
            float(row.get('query_quality', query.get('quality', 1.0)) or 1.0),
            [negative_score] if negative_score > 0.0 else [],
            str(row.get('origin', query.get('origin', '')) or ''),
        )
        cross_rows.append({
            'id': str(row.get('pair_id', '') or ''),
            'text1': str(row.get('query', '') or ''),
            'text2': str(row.get('text', '') or ''),
            'label': int(row.get('label', 0) or 0),
            'query_id': str(row.get('query_id', '') or ''),
            'corpus_id': str(row.get('corpus_id', '') or ''),
            'query_type': str(row.get('query_type', query.get('query_type', '')) or ''),
            'query_quality': float(row.get('query_quality', query.get('quality', 1.0)) or 1.0),
            'negative_source': str(row.get('negative_source', '') or ''),
            'negative_score': negative_score,
            'origin': str(row.get('origin', query.get('origin', '')) or ''),
            'domain': str(row.get('domain', query.get('domain', '')) or ''),
            'sample_weight': weight,
            'split': str(row.get('split', '') or ''),
        })
    _write_jsonl(ce_root / 'train.jsonl', _split_filter(cross_rows, 'train'))
    _write_jsonl(ce_root / 'val.jsonl', _split_filter(cross_rows, 'val'))

    # Curriculum slices
    curriculum_root = export_root / 'curriculum'
    high_conf_rows = [
        row for row in bi_encoder_rows
        if float(row.get('query_quality', 0.0) or 0.0) >= 0.9 and str(row.get('split', '') or '') == 'train'
    ]
    _write_jsonl(curriculum_root / 'bi_encoder_high_conf_train.jsonl', high_conf_rows)
    _write_jsonl(
        curriculum_root / 'cross_encoder_high_conf_train.jsonl',
        [
            row for row in cross_rows
            if float(row.get('query_quality', 0.0) or 0.0) >= 0.9 and str(row.get('split', '') or '') == 'train'
        ],
    )

    summary = {
        'dataset_name': dataset_name,
        'export_root': str(export_root),
        'counts': {
            'corpus': len(corpus_rows),
            'queries': len(query_rows),
            'triplets': len(triplet_rows),
            'pairs': len(pair_rows),
            'bi_encoder_train': len(_split_filter(bi_encoder_rows, 'train')),
            'bi_encoder_val': len(_split_filter(bi_encoder_rows, 'val')),
            'cross_encoder_train': len(_split_filter(cross_rows, 'train')),
            'cross_encoder_val': len(_split_filter(cross_rows, 'val')),
            'high_conf_bi_encoder_train': len(high_conf_rows),
        },
        'query_types': dict(Counter(str(row.get('query_type', '') or '') for row in query_rows)),
        'origins': dict(Counter(str(row.get('origin', '') or '') for row in query_rows)),
    }
    (export_root / 'manifest.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Export unified retriever dataset to training-friendly formats.')
    parser.add_argument('--dataset-name', default='retriever_unified_v3')
    parser.add_argument('--export-name', default='')
    args = parser.parse_args()

    summary = export_retriever_formats(
        dataset_name=str(args.dataset_name),
        export_name=str(args.export_name),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
