from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from core.reward_model import DEFAULT_REWARD_MODEL_PATH, train_reward_model
from scripts.build_rlhf_dataset import build_rlhf_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a lightweight local reward model from RLHF preference pairs.')
    parser.add_argument('--dataset-prefix', default='dataset', help='RLHF dataset prefix inside storage/rlhf')
    parser.add_argument('--preferences-path', default='', help='Explicit preference JSONL path')
    parser.add_argument('--model-path', default=str(DEFAULT_REWARD_MODEL_PATH), help='Output reward model pickle path')
    parser.add_argument('--no-build-if-missing', action='store_true', help='Do not build RLHF dataset when preferences are missing')
    args = parser.parse_args()

    preferences_path = Path(args.preferences_path) if args.preferences_path else RLHF_DIR / f'{args.dataset_prefix}_preferences.jsonl'
    if not preferences_path.exists() and not args.no_build_if_missing:
        build_rlhf_dataset(output_prefix=args.dataset_prefix)
    if not preferences_path.exists():
        raise FileNotFoundError(f'Missing preference dataset: {preferences_path}')

    summary = train_reward_model(preferences_path=preferences_path, model_path=Path(args.model_path))
    summary['preferences_path'] = str(preferences_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
