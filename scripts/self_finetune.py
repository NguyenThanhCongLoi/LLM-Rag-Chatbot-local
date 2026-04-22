from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from core.pipeline_v4 import UnetiDocumentAgentV4Max
from core.reward_model import DEFAULT_REWARD_MODEL_PATH, load_reward_model, train_reward_model_from_rows
from scripts.build_rlhf_dataset import build_rlhf_dataset


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in materialized), encoding='utf-8')
    return len(materialized)


def _norm(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _verified(debug: Dict[str, Any]) -> bool:
    verification = debug.get('answer_verification') or {}
    contexts = verification.get('contexts') or []
    if not contexts:
        return False
    route = str(debug.get('route', '') or '')
    return route not in {'out_of_scope', 'policy_block', 'sensitive_block', 'meta', 'clarification'}


def build_self_preferences(
    dataset_prefix: str = 'dataset',
    output_prefix: str = 'self',
    reward_model_path: Path = DEFAULT_REWARD_MODEL_PATH,
    max_rows: int = 50,
    min_reward: float = 0.52,
    margin: float = 0.08,
    build_if_missing: bool = True,
) -> Dict[str, Any]:
    preferences_path = RLHF_DIR / f'{dataset_prefix}_preferences.jsonl'
    if build_if_missing and not preferences_path.exists():
        build_rlhf_dataset(output_prefix=dataset_prefix)
    seed_rows = _read_jsonl(preferences_path)
    scorer = load_reward_model(str(reward_model_path))
    agent = UnetiDocumentAgentV4Max()
    out_rows: List[Dict[str, Any]] = []

    for row in seed_rows:
        if len(out_rows) >= max_rows:
            break
        question = _norm(row.get('question', ''))
        rejected = _norm(row.get('rejected', ''))
        baseline = _norm(row.get('chosen', ''))
        if not question:
            continue
        answer, debug, _memory = agent.answer(question, {})
        answer = _norm(answer)
        if not answer or not _verified(debug):
            continue
        answer_reward = float(scorer.score(question, answer))
        baseline_reward = float(scorer.score(question, baseline)) if baseline else 0.0
        rejected_reward = float(scorer.score(question, rejected)) if rejected else 0.0

        chosen = answer
        loser = rejected
        reason = 'generated_vs_rejected'
        if baseline and baseline_reward >= answer_reward + margin:
            chosen = baseline
            loser = answer
            reason = 'baseline_vs_generated'
        elif answer_reward < min_reward:
            continue
        elif not loser and baseline and answer_reward >= baseline_reward + margin:
            loser = baseline
            reason = 'generated_vs_baseline'

        if not loser or _norm(loser) == _norm(chosen):
            continue

        out_rows.append({
            'turn_id': f"self:{row.get('turn_id', '')}",
            'student_id': 'self_finetune',
            'question': question,
            'chosen': chosen,
            'rejected': loser,
            'meta': {
                **(row.get('meta') or {}),
                'source': 'self_finetune',
                'reason': reason,
                'generated_reward': round(answer_reward, 6),
                'baseline_reward': round(baseline_reward, 6),
                'rejected_reward': round(rejected_reward, 6),
                'reward_model_kind': getattr(scorer, 'metadata', {}).get('kind', ''),
            },
        })

    output_path = RLHF_DIR / f'{output_prefix}_preferences.jsonl'
    count = _write_jsonl(output_path, out_rows)
    summary = {
        'dataset_prefix': dataset_prefix,
        'output_path': str(output_path),
        'preference_rows': count,
        'reward_model_kind': getattr(scorer, 'metadata', {}).get('kind', ''),
    }
    summary_path = RLHF_DIR / f'{output_prefix}_summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    summary['summary_path'] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build conservative self-training preference pairs from verified answers.')
    parser.add_argument('--dataset-prefix', default='dataset')
    parser.add_argument('--output-prefix', default='self')
    parser.add_argument('--reward-model-path', default=str(DEFAULT_REWARD_MODEL_PATH))
    parser.add_argument('--max-rows', type=int, default=50)
    parser.add_argument('--min-reward', type=float, default=0.52)
    parser.add_argument('--margin', type=float, default=0.08)
    parser.add_argument('--no-build-if-missing', action='store_true')
    parser.add_argument('--auto-train-reward', action='store_true', help='Train reward model on generated self-preference rows after export')
    args = parser.parse_args()

    summary = build_self_preferences(
        dataset_prefix=args.dataset_prefix,
        output_prefix=args.output_prefix,
        reward_model_path=Path(args.reward_model_path),
        max_rows=args.max_rows,
        min_reward=args.min_reward,
        margin=args.margin,
        build_if_missing=not args.no_build_if_missing,
    )
    if args.auto_train_reward and summary.get('preference_rows', 0):
        rows = _read_jsonl(Path(summary['output_path']))
        summary['reward_train'] = train_reward_model_from_rows(rows, model_path=Path(args.reward_model_path))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
