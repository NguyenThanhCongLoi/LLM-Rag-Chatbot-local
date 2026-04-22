from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import RLHF_DIR
from .normalize import norm_text_ascii


DEFAULT_REWARD_MODEL_PATH = RLHF_DIR / 'reward_model.pkl'

STOPWORDS = {
    'ban', 'toi', 'minh', 'cho', 'hoi', 've', 'la', 'gi', 'nao', 'o', 'dau', 'co', 'cua',
    'trong', 'va', 'hoac', 'thi', 'nhu', 'the', 'can', 'hay', 'duoc', 'khong', 'mot',
    'cac', 'nhung', 'nay', 'do', 'day', 'nay', 'uneti', 'truong',
}

BAD_DIRECT_MARKERS = [
    'lich su hinh thanh',
    'ngay nay da co lich su',
    'song song voi viec giang day',
    'khong tim thay du thong tin',
]


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


def _norm(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _pair_text(question: str, answer: str) -> str:
    return f'question: {_norm(question)}\nanswer: {_norm(answer)}'


def _tokens(text: str) -> List[str]:
    return [
        tok for tok in re.findall(r'[a-z0-9]+', norm_text_ascii(text))
        if len(tok) >= 2 and tok not in STOPWORDS
    ]


def _clamp01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.5
    return max(0.0, min(1.0, float(value)))


class HeuristicRewardScorer:
    """Fallback scorer used before a trained reward model exists."""

    metadata = {
        'kind': 'heuristic',
        'trained': False,
        'example_count': 0,
    }

    def score(self, question: str, answer: str) -> float:
        q_tokens = set(_tokens(question))
        a_text = norm_text_ascii(answer)
        a_tokens = _tokens(answer)
        if not _norm(answer):
            return 0.0

        coverage = 0.0
        if q_tokens:
            coverage = sum(1 for tok in q_tokens if tok in a_text) / max(1, len(q_tokens))

        length = len(_norm(answer))
        if length <= 220:
            brevity = 1.0
        elif length <= 600:
            brevity = 0.82
        elif length <= 1200:
            brevity = 0.62
        else:
            brevity = 0.42

        focus = 0.0
        if a_tokens and q_tokens:
            focus = sum(1 for tok in a_tokens if tok in q_tokens) / max(1, len(a_tokens))

        score = 0.45 + coverage * 0.24 + brevity * 0.18 + min(0.18, focus * 1.4)

        q = norm_text_ascii(question)
        direct_query = any(marker in q for marker in [' la gi', ' o dau', ' day mon gi', ' dao tao gi', ' ai'])
        if direct_query and any(marker in a_text for marker in BAD_DIRECT_MARKERS):
            score -= 0.28
        if 'khong tim thay' in a_text and coverage < 0.25:
            score -= 0.18
        if len(set(a_tokens)) <= 2:
            score -= 0.15
        return _clamp01(score)

    def score_many(self, question: str, answers: Sequence[str]) -> List[float]:
        return [self.score(question, answer) for answer in answers]


@dataclass
class RewardScorer:
    pipeline: Any
    metadata: Dict[str, Any]

    def score(self, question: str, answer: str) -> float:
        if not _norm(answer):
            return 0.0
        try:
            if hasattr(self.pipeline, 'predict_proba'):
                proba = self.pipeline.predict_proba([_pair_text(question, answer)])[0]
                classes = list(getattr(self.pipeline, 'classes_', [0, 1]))
                positive_idx = classes.index(1) if 1 in classes else len(proba) - 1
                return _clamp01(float(proba[positive_idx]))
            pred = self.pipeline.decision_function([_pair_text(question, answer)])[0]
            return _clamp01(1.0 / (1.0 + math.exp(-float(pred))))
        except Exception:
            return HeuristicRewardScorer().score(question, answer)

    def score_many(self, question: str, answers: Sequence[str]) -> List[float]:
        return [self.score(question, answer) for answer in answers]

    def save(self, path: Path = DEFAULT_REWARD_MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as fh:
            pickle.dump({'pipeline': self.pipeline, 'metadata': self.metadata}, fh)
        meta_path = path.with_suffix('.meta.json')
        meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        load_reward_model.cache_clear()
        return path


def preference_examples(rows: Iterable[Dict[str, Any]]) -> tuple[List[str], List[int]]:
    examples: List[str] = []
    labels: List[int] = []
    seen = set()
    for row in rows:
        question = _norm(str(row.get('question', '') or row.get('prompt', '') or ''))
        chosen = _norm(str(row.get('chosen', '') or row.get('preferred', '') or row.get('response', '') or ''))
        rejected = _norm(str(row.get('rejected', '') or ''))
        if not question:
            continue
        if chosen:
            key = (question, chosen, 1)
            if key not in seen:
                seen.add(key)
                examples.append(_pair_text(question, chosen))
                labels.append(1)
        if rejected:
            key = (question, rejected, 0)
            if key not in seen:
                seen.add(key)
                examples.append(_pair_text(question, rejected))
                labels.append(0)
    return examples, labels


def train_reward_model_from_rows(rows: Iterable[Dict[str, Any]], model_path: Path = DEFAULT_REWARD_MODEL_PATH) -> Dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    materialized = list(rows)
    examples, labels = preference_examples(materialized)
    if len(examples) < 2 or len(set(labels)) < 2:
        raise ValueError('Need at least one chosen and one rejected answer to train reward model.')

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=1,
            max_features=12000,
            lowercase=True,
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipeline.fit(examples, labels)
    scorer = RewardScorer(
        pipeline=pipeline,
        metadata={
            'kind': 'sklearn_logreg_tfidf',
            'trained': True,
            'preference_rows': len(materialized),
            'example_count': len(examples),
            'positive_examples': int(sum(labels)),
            'negative_examples': int(len(labels) - sum(labels)),
            'model_path': str(model_path),
        },
    )
    scorer.save(model_path)
    return dict(scorer.metadata)


def train_reward_model(preferences_path: Path, model_path: Path = DEFAULT_REWARD_MODEL_PATH) -> Dict[str, Any]:
    rows = _read_jsonl(preferences_path)
    return train_reward_model_from_rows(rows, model_path=model_path)


@lru_cache(maxsize=4)
def load_reward_model(path: str = '') -> Any:
    model_path = Path(path) if path else DEFAULT_REWARD_MODEL_PATH
    if not model_path.exists():
        return HeuristicRewardScorer()
    try:
        with model_path.open('rb') as fh:
            payload = pickle.load(fh)
        return RewardScorer(
            pipeline=payload.get('pipeline'),
            metadata=payload.get('metadata') or {'kind': 'unknown', 'trained': True},
        )
    except Exception:
        return HeuristicRewardScorer()


def score_answer(question: str, answer: str, model_path: Path = DEFAULT_REWARD_MODEL_PATH) -> float:
    return float(load_reward_model(str(model_path)).score(question, answer))
