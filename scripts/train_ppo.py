from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from core.reward_model import DEFAULT_REWARD_MODEL_PATH, load_reward_model


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _dependency_error() -> str:
    missing = []
    for module in ['torch', 'transformers', 'datasets', 'trl']:
        try:
            __import__(module)
        except Exception:
            missing.append(module)
    if not missing:
        return ''
    return 'Missing optional PPO dependencies: ' + ', '.join(missing)


def dry_run(dataset_path: Path, reward_model_path: Path) -> Dict[str, Any]:
    rows = _read_jsonl(dataset_path)
    scorer = load_reward_model(str(reward_model_path))
    samples = []
    for row in rows[:5]:
        prompt = str(row.get('prompt', '') or '')
        reference = str(row.get('reference', '') or '')
        samples.append({
            'id': row.get('id', ''),
            'prompt': prompt[:120],
            'reference_reward': round(float(scorer.score(prompt, reference)), 6),
            'source': row.get('source', ''),
        })
    return {
        'dataset_path': str(dataset_path),
        'rows': len(rows),
        'reward_model_kind': getattr(scorer, 'metadata', {}).get('kind', ''),
        'dependency_status': _dependency_error() or 'ok',
        'samples': samples,
    }


def run_ppo(args: argparse.Namespace) -> Dict[str, Any]:
    dependency_error = _dependency_error()
    if dependency_error:
        raise RuntimeError(
            dependency_error
            + '. Install optional dependencies from requirements-rlhf.txt before running PPO.'
        )

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    rows = _read_jsonl(Path(args.dataset_path))
    if not rows:
        raise ValueError(f'No PPO prompt rows in {args.dataset_path}')

    scorer = load_reward_model(str(args.reward_model_path))
    tokenizer = AutoTokenizer.from_pretrained(args.model_ref)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_ref)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args.model_ref)
    dataset = Dataset.from_list([{'query': str(row.get('prompt', '') or '')} for row in rows])

    config = PPOConfig(
        model_name=args.model_ref,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=max(1, min(args.batch_size, args.mini_batch_size)),
    )
    trainer = PPOTrainer(config=config, model=model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset)
    generation_kwargs = {
        'min_length': -1,
        'top_k': 0.0,
        'top_p': 0.9,
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'max_new_tokens': args.max_new_tokens,
    }

    steps = 0
    for batch in trainer.dataloader:
        queries = batch['query']
        query_tensors = [tokenizer.encode(query, return_tensors='pt').squeeze(0) for query in queries]
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        responses = [tokenizer.decode(resp.squeeze(), skip_special_tokens=True) for resp in response_tensors]
        rewards = [torch.tensor(float(scorer.score(query, response))) for query, response in zip(queries, responses)]
        trainer.step(query_tensors, response_tensors, rewards)
        steps += 1
        if steps >= args.max_steps:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_pretrained(str(output_dir))
    return {
        'output_dir': str(output_dir),
        'steps': steps,
        'rows': len(rows),
        'reward_model_kind': getattr(scorer, 'metadata', {}).get('kind', ''),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Optional PPO trainer using TRL and the local reward model.')
    parser.add_argument('--dataset-path', default=str(RLHF_DIR / 'dataset_ppo_prompts.jsonl'))
    parser.add_argument('--reward-model-path', default=str(DEFAULT_REWARD_MODEL_PATH))
    parser.add_argument('--model-ref', default='', help='Hugging Face/local causal LM path for PPO')
    parser.add_argument('--output-dir', default=str(RLHF_DIR / 'ppo_model'))
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--mini-batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--max-new-tokens', type=int, default=180)
    parser.add_argument('--max-steps', type=int, default=20)
    parser.add_argument('--dry-run', action='store_true', help='Validate dataset/reward model without training')
    args = parser.parse_args()

    if args.dry_run:
        summary = dry_run(Path(args.dataset_path), Path(args.reward_model_path))
    else:
        if not args.model_ref:
            raise ValueError('--model-ref is required when not using --dry-run')
        summary = run_ppo(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
