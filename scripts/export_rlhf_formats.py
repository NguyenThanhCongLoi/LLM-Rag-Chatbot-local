from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from scripts.build_rlhf_dataset import build_rlhf_dataset


FRAMEWORKS = ('trl', 'axolotl', 'unsloth')


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


def _normalize_csv(values: Sequence[str] | None) -> List[str]:
    out: List[str] = []
    for value in values or []:
        for item in str(value or '').split(','):
            text = item.strip()
            if text and text not in out:
                out.append(text)
    return out


def _stable_bucket(value: str, salt: str = '') -> float:
    key = f'{salt}:{value}'.encode('utf-8')
    digest = hashlib.sha1(key).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _split_rows(rows: List[Dict[str, Any]], val_ratio: float, salt: str) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if val_ratio <= 0:
        return rows, []
    if val_ratio >= 1:
        return [], rows
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for row in rows:
        key = str(row.get('turn_id', '') or row.get('question', '') or row.get('prompt', '') or '')
        if _stable_bucket(key, salt=salt) < val_ratio:
            val_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, val_rows


def _row_domain(row: Dict[str, Any]) -> str:
    meta = row.get('meta') or {}
    return str(row.get('domain', '') or meta.get('domain', '') or '').strip()


def _row_rating(row: Dict[str, Any]) -> int:
    meta = row.get('meta') or {}
    try:
        return int(meta.get('feedback_rating', 0) or 0)
    except Exception:
        return 0


def _row_verdicts(row: Dict[str, Any]) -> List[str]:
    meta = row.get('meta') or {}
    verdicts = meta.get('review_verdicts', []) or []
    return [str(item or '').strip() for item in verdicts if str(item or '').strip()]


def _matches_filters(
    row: Dict[str, Any],
    domains: Sequence[str],
    verdicts: Sequence[str],
    min_rating: int | None,
    max_rating: int | None,
) -> bool:
    domain_filters = set(_normalize_csv(domains))
    verdict_filters = set(_normalize_csv(verdicts))
    row_domain = _row_domain(row)
    if domain_filters and row_domain not in domain_filters:
        return False

    rating = _row_rating(row)
    if min_rating is not None:
        if rating <= 0 or rating < min_rating:
            return False
    if max_rating is not None:
        if rating <= 0 or rating > max_rating:
            return False

    row_verdicts = set(_row_verdicts(row))
    if verdict_filters and not (row_verdicts & verdict_filters):
        return False
    return True


def _trl_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(row.get('prompt', '') or '')
    response = str(row.get('response', '') or '')
    return {
        'messages': [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response},
        ],
        'prompt': prompt,
        'completion': response,
        'meta': row.get('meta', {}),
    }


def _trl_preference(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'prompt': str(row.get('question', '') or ''),
        'chosen': str(row.get('chosen', '') or ''),
        'rejected': str(row.get('rejected', '') or ''),
        'meta': row.get('meta', {}),
    }


def _axolotl_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'instruction': str(row.get('prompt', '') or ''),
        'input': '',
        'output': str(row.get('response', '') or ''),
        'meta': row.get('meta', {}),
    }


def _axolotl_preference(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'prompt': str(row.get('question', '') or ''),
        'chosen': str(row.get('chosen', '') or ''),
        'rejected': str(row.get('rejected', '') or ''),
        'meta': row.get('meta', {}),
    }


def _unsloth_sft(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt = str(row.get('prompt', '') or '')
    response = str(row.get('response', '') or '')
    return {
        'messages': [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': response},
        ],
        'meta': row.get('meta', {}),
    }


def _unsloth_preference(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'prompt': str(row.get('question', '') or ''),
        'chosen': str(row.get('chosen', '') or ''),
        'rejected': str(row.get('rejected', '') or ''),
        'meta': row.get('meta', {}),
    }


FORMATTERS = {
    'trl': {
        'sft': _trl_sft,
        'preference': _trl_preference,
    },
    'axolotl': {
        'sft': _axolotl_sft,
        'preference': _axolotl_preference,
    },
    'unsloth': {
        'sft': _unsloth_sft,
        'preference': _unsloth_preference,
    },
}


def export_rlhf_formats(
    dataset_prefix: str = 'dataset',
    export_prefix: str = '',
    frameworks: Sequence[str] | None = None,
    domains: Sequence[str] | None = None,
    verdicts: Sequence[str] | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    val_ratio: float = 0.1,
    split_salt: str = 'uneti-rlhf-v1',
    build_if_missing: bool = True,
) -> Dict[str, Any]:
    frameworks = _normalize_csv(frameworks) or list(FRAMEWORKS)
    invalid = [item for item in frameworks if item not in FORMATTERS]
    if invalid:
        raise ValueError(f'Unsupported frameworks: {invalid}')

    candidates_path = RLHF_DIR / f'{dataset_prefix}_candidates.jsonl'
    preferences_path = RLHF_DIR / f'{dataset_prefix}_preferences.jsonl'
    sft_path = RLHF_DIR / f'{dataset_prefix}_sft.jsonl'
    if build_if_missing and (not preferences_path.exists() or not sft_path.exists()):
        build_rlhf_dataset(output_prefix=dataset_prefix)

    preference_rows = [
        row for row in _read_jsonl(preferences_path)
        if _matches_filters(row, domains or [], verdicts or [], min_rating, max_rating)
    ]
    sft_rows = [
        row for row in _read_jsonl(sft_path)
        if _matches_filters(row, domains or [], verdicts or [], min_rating, max_rating)
    ]
    candidate_rows = [
        row for row in _read_jsonl(candidates_path)
        if _matches_filters(row, domains or [], verdicts or [], min_rating, max_rating)
    ]

    export_name = export_prefix.strip() or dataset_prefix
    export_root = RLHF_DIR / 'exports' / export_name
    summary: Dict[str, Any] = {
        'dataset_prefix': dataset_prefix,
        'export_prefix': export_name,
        'filters': {
            'domains': _normalize_csv(domains),
            'verdicts': _normalize_csv(verdicts),
            'min_rating': min_rating,
            'max_rating': max_rating,
            'val_ratio': val_ratio,
        },
        'source_counts': {
            'candidates': len(candidate_rows),
            'preferences': len(preference_rows),
            'sft': len(sft_rows),
        },
        'frameworks': {},
    }

    sft_train, sft_val = _split_rows(sft_rows, val_ratio=val_ratio, salt=f'{split_salt}:sft')
    pref_train, pref_val = _split_rows(preference_rows, val_ratio=val_ratio, salt=f'{split_salt}:pref')

    for framework in frameworks:
        formatter = FORMATTERS[framework]
        framework_root = export_root / framework

        counts = {
            'sft_train': _write_jsonl(
                framework_root / 'sft_train.jsonl',
                (formatter['sft'](row) for row in sft_train),
            ),
            'sft_val': _write_jsonl(
                framework_root / 'sft_val.jsonl',
                (formatter['sft'](row) for row in sft_val),
            ),
            'preference_train': _write_jsonl(
                framework_root / 'preference_train.jsonl',
                (formatter['preference'](row) for row in pref_train),
            ),
            'preference_val': _write_jsonl(
                framework_root / 'preference_val.jsonl',
                (formatter['preference'](row) for row in pref_val),
            ),
        }
        summary['frameworks'][framework] = {
            'path': str(framework_root),
            'counts': counts,
        }

    manifest_path = export_root / 'manifest.json'
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['manifest_path'] = str(manifest_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Filter, split, and export RLHF datasets for common training frameworks.')
    parser.add_argument('--dataset-prefix', default='dataset', help='Base RLHF dataset prefix inside storage/rlhf')
    parser.add_argument('--export-prefix', default='', help='Output folder name under storage/rlhf/exports')
    parser.add_argument('--frameworks', nargs='*', default=list(FRAMEWORKS), help='trl, axolotl, unsloth')
    parser.add_argument('--domains', nargs='*', default=[], help='Optional domain filters')
    parser.add_argument('--verdicts', nargs='*', default=[], help='Optional review verdict filters')
    parser.add_argument('--min-rating', type=int, default=None, help='Minimum feedback rating to include')
    parser.add_argument('--max-rating', type=int, default=None, help='Maximum feedback rating to include')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation split ratio between 0 and 1')
    parser.add_argument('--split-salt', default='uneti-rlhf-v1', help='Salt for deterministic split hashing')
    parser.add_argument('--no-build-if-missing', action='store_true', help='Do not auto-build the base RLHF dataset if files are missing')
    args = parser.parse_args()

    summary = export_rlhf_formats(
        dataset_prefix=args.dataset_prefix,
        export_prefix=args.export_prefix,
        frameworks=args.frameworks,
        domains=args.domains,
        verdicts=args.verdicts,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        val_ratio=args.val_ratio,
        split_salt=args.split_salt,
        build_if_missing=not args.no_build_if_missing,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
