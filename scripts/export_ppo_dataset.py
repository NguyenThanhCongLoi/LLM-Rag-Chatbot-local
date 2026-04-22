from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from scripts.build_rlhf_dataset import build_rlhf_dataset


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in materialized), encoding='utf-8')
    return len(materialized)


def _norm(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _stable_id(prompt: str, reference: str) -> str:
    return hashlib.sha1(f'{prompt}\n{reference}'.encode('utf-8')).hexdigest()[:16]


def export_ppo_dataset(
    dataset_prefix: str = 'dataset',
    output_prefix: str = '',
    build_if_missing: bool = True,
    include_sft: bool = True,
) -> Dict[str, Any]:
    preferences_path = RLHF_DIR / f'{dataset_prefix}_preferences.jsonl'
    sft_path = RLHF_DIR / f'{dataset_prefix}_sft.jsonl'
    if build_if_missing and (not preferences_path.exists() or (include_sft and not sft_path.exists())):
        build_rlhf_dataset(output_prefix=dataset_prefix)

    preference_rows = _read_jsonl(preferences_path)
    sft_rows = _read_jsonl(sft_path) if include_sft else []
    rows: List[Dict[str, Any]] = []
    seen = set()

    for row in preference_rows:
        prompt = _norm(row.get('question', ''))
        chosen = _norm(row.get('chosen', ''))
        rejected = _norm(row.get('rejected', ''))
        if not prompt or not chosen:
            continue
        key = (prompt, chosen)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'id': _stable_id(prompt, chosen),
            'prompt': prompt,
            'reference': chosen,
            'rejected': rejected,
            'source': 'preference',
            'meta': row.get('meta', {}),
        })

    for row in sft_rows:
        prompt = _norm(row.get('prompt', ''))
        response = _norm(row.get('response', ''))
        if not prompt or not response:
            continue
        key = (prompt, response)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'id': _stable_id(prompt, response),
            'prompt': prompt,
            'reference': response,
            'rejected': '',
            'source': 'sft',
            'meta': row.get('meta', {}),
        })

    export_name = output_prefix.strip() or dataset_prefix
    out_path = RLHF_DIR / f'{export_name}_ppo_prompts.jsonl'
    count = _write_jsonl(out_path, rows)
    summary = {
        'dataset_prefix': dataset_prefix,
        'output_path': str(out_path),
        'prompt_rows': count,
        'source_counts': {
            'preferences': len(preference_rows),
            'sft': len(sft_rows),
        },
    }
    summary_path = RLHF_DIR / f'{export_name}_ppo_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['summary_path'] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Export PPO prompts from RLHF SFT/preference data.')
    parser.add_argument('--dataset-prefix', default='dataset', help='Base RLHF dataset prefix inside storage/rlhf')
    parser.add_argument('--output-prefix', default='', help='Output PPO prefix inside storage/rlhf')
    parser.add_argument('--no-build-if-missing', action='store_true', help='Do not build the base RLHF dataset if files are missing')
    parser.add_argument('--no-sft', action='store_true', help='Use preference rows only')
    args = parser.parse_args()

    summary = export_ppo_dataset(
        dataset_prefix=args.dataset_prefix,
        output_prefix=args.output_prefix,
        build_if_missing=not args.no_build_if_missing,
        include_sft=not args.no_sft,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
