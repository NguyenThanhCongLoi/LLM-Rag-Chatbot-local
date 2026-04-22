from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import HISTORY_DIR, REVIEW_DB_PATH
from .history import load_history
from .normalize import norm_text_ascii


DEFAULT_REVIEW_DB = {'reviews': []}
VERDICTS = ['correct', 'incorrect', 'missing', 'excessive', 'partial']
VERDICT_LABELS = {
    'correct': 'Đúng',
    'incorrect': 'Sai',
    'missing': 'Thiếu',
    'excessive': 'Thừa',
    'partial': 'Một phần',
}
REVIEW_KINDS = ['answer', 'policy']
REVIEW_KIND_LABELS = {
    'answer': 'Đáp án',
    'policy': 'Chính sách',
}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def _dedupe_strings(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or '').strip()
        norm = norm_text_ascii(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _review_variants(review: Dict[str, Any]) -> List[str]:
    return _dedupe_strings([
        str(review.get('user_question', '') or '').strip(),
        *[str(x or '').strip() for x in review.get('match_questions', []) or []],
        *[str(x or '').strip() for x in review.get('auto_match_questions', []) or []],
    ])


def load_review_db() -> Dict[str, Any]:
    if not REVIEW_DB_PATH.exists():
        return {'reviews': []}
    try:
        data = json.loads(REVIEW_DB_PATH.read_text(encoding='utf-8'))
        reviews = data.get('reviews', [])
        if not isinstance(reviews, list):
            reviews = []
        return {'reviews': reviews}
    except Exception:
        return {'reviews': []}


def save_review_db(data: Dict[str, Any]) -> None:
    payload = {'reviews': data.get('reviews', [])}
    REVIEW_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def list_history_student_ids() -> List[str]:
    return sorted(path.stem for path in HISTORY_DIR.glob('*.json'))


def history_turns(student_id: str) -> List[Dict[str, Any]]:
    hist = load_history(student_id)
    messages = hist.get('messages', [])
    turns: List[Dict[str, Any]] = []
    last_user = ''
    last_user_index = -1
    for idx, msg in enumerate(messages):
        role = str(msg.get('role', '') or '')
        if role == 'user':
            last_user = str(msg.get('content', '') or '')
            last_user_index = idx
            continue
        if role != 'assistant':
            continue
        turns.append({
            'student_id': student_id,
            'assistant_index': idx,
            'user_index': last_user_index,
            'turn_id': f'{student_id}:{idx}',
            'question': last_user,
            'answer': str(msg.get('content', '') or ''),
            'debug': str(msg.get('debug', '') or ''),
        })
    return turns


def get_review(turn_id: str) -> Dict[str, Any]:
    db = load_review_db()
    for item in db.get('reviews', []):
        if str(item.get('turn_id', '')) == turn_id:
            return item
    return {}


def upsert_review(review: Dict[str, Any]) -> Dict[str, Any]:
    db = load_review_db()
    reviews = db.get('reviews', [])
    turn_id = str(review.get('turn_id', '') or '').strip()
    if not turn_id:
        raise ValueError('turn_id is required')

    normalized_review = {
        'turn_id': turn_id,
        'student_id': str(review.get('student_id', '') or '').strip(),
        'assistant_index': int(review.get('assistant_index', 0) or 0),
        'user_question': str(review.get('user_question', '') or '').strip(),
        'assistant_answer': str(review.get('assistant_answer', '') or '').strip(),
        'verdict': str(review.get('verdict', 'partial') or 'partial').strip(),
        'review_kind': str(review.get('review_kind', 'answer') or 'answer').strip(),
        'policy_code': str(review.get('policy_code', '') or '').strip(),
        'approved_answer': str(review.get('approved_answer', '') or '').strip(),
        'retrieval_hint': str(review.get('retrieval_hint', '') or '').strip(),
        'domain_hint': str(review.get('domain_hint', '') or '').strip(),
        'notes': str(review.get('notes', '') or '').strip(),
        'active': bool(review.get('active', True)),
        'match_questions': _dedupe_strings([
            str(review.get('user_question', '') or '').strip(),
            *[str(x or '').strip() for x in review.get('match_questions', []) or []],
        ]),
        'auto_match_questions': _dedupe_strings([str(x or '').strip() for x in review.get('auto_match_questions', []) or []])[:40],
        'usage_count': int(review.get('usage_count', 0) or 0),
        'last_used_at': str(review.get('last_used_at', '') or '').strip(),
        'updated_at': _utc_now(),
    }
    if normalized_review['verdict'] not in VERDICTS:
        normalized_review['verdict'] = 'partial'
    if normalized_review['review_kind'] not in REVIEW_KINDS:
        normalized_review['review_kind'] = 'answer'
    if not normalized_review['approved_answer'] and normalized_review['verdict'] == 'correct':
        normalized_review['approved_answer'] = normalized_review['assistant_answer']

    for idx, item in enumerate(reviews):
        if str(item.get('turn_id', '')) == turn_id:
            normalized_review['created_at'] = str(item.get('created_at', '') or normalized_review['updated_at'])
            normalized_review['auto_match_questions'] = _dedupe_strings([
                *[str(x or '').strip() for x in item.get('auto_match_questions', []) or []],
                *[str(x or '').strip() for x in normalized_review.get('auto_match_questions', []) or []],
            ])[:40]
            normalized_review['usage_count'] = int(review.get('usage_count', item.get('usage_count', 0)) or 0)
            normalized_review['last_used_at'] = str(review.get('last_used_at', item.get('last_used_at', '')) or '').strip()
            reviews[idx] = normalized_review
            save_review_db(db)
            return normalized_review

    normalized_review['created_at'] = normalized_review['updated_at']
    reviews.append(normalized_review)
    save_review_db(db)
    return normalized_review


def all_reviews() -> List[Dict[str, Any]]:
    db = load_review_db()
    reviews = list(db.get('reviews', []))
    reviews.sort(key=lambda x: str(x.get('updated_at', '')), reverse=True)
    return reviews


def _question_score(question: str, candidate: Dict[str, Any]) -> float:
    q_norm = norm_text_ascii(question)
    if not q_norm:
        return 0.0
    q_tokens = set(q_norm.split())
    best = 0.0
    for variant in _review_variants(candidate):
        v_norm = norm_text_ascii(variant)
        if not v_norm:
            continue
        if v_norm == q_norm:
            return 10.0
        v_tokens = set(v_norm.split())
        overlap = len(q_tokens & v_tokens) / max(1, len(q_tokens))
        contains = 0.5 if q_norm in v_norm or v_norm in q_norm else 0.0
        best = max(best, overlap + contains)
    return best


def lookup_review(question: str) -> Dict[str, Any]:
    candidates = []
    for item in all_reviews():
        if not item.get('active', True):
            continue
        if str(item.get('review_kind', 'answer') or 'answer') != 'answer':
            continue
        score = _question_score(question, item)
        if score <= 0:
            continue
        candidates.append((score, item))
    if not candidates:
        return {}
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, best = candidates[0]
    result = dict(best)
    result['match_score'] = score
    return result


def lookup_policy_review(policy_code: str) -> Dict[str, Any]:
    if not policy_code:
        return {}
    for item in all_reviews():
        if not item.get('active', True):
            continue
        if str(item.get('review_kind', 'answer') or 'answer') != 'policy':
            continue
        if str(item.get('policy_code', '') or '').strip() != policy_code:
            continue
        return dict(item)
    return {}


def record_review_usage(review: Dict[str, Any], question: str) -> Dict[str, Any]:
    turn_id = str(review.get('turn_id', '') or '').strip()
    asked = str(question or '').strip()
    if not turn_id or not asked:
        return {}

    db = load_review_db()
    reviews = db.get('reviews', [])
    for idx, item in enumerate(reviews):
        if str(item.get('turn_id', '') or '').strip() != turn_id:
            continue
        updated = dict(item)
        updated['usage_count'] = int(updated.get('usage_count', 0) or 0) + 1
        updated['last_used_at'] = _utc_now()
        variants = _dedupe_strings([
            *[str(x or '').strip() for x in updated.get('auto_match_questions', []) or []],
            asked,
        ])[:40]
        updated['auto_match_questions'] = variants
        reviews[idx] = updated
        save_review_db(db)
        return updated
    return {}
