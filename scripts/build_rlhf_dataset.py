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
from core.contribution_store import list_contributions
from core.feedback_store import is_feedback_eligible, list_feedback
from core.history import load_history
from core.review_store import all_reviews, list_history_student_ids


def _norm(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _load_debug(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _history_turns() -> Dict[str, Dict[str, Any]]:
    turns: Dict[str, Dict[str, Any]] = {}
    for student_id in list_history_student_ids():
        hist = load_history(student_id)
        messages = hist.get('messages', []) or []
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
            turn_id = f'{student_id}:{idx}'
            debug = _load_debug(msg.get('debug'))
            turns[turn_id] = {
                'turn_id': turn_id,
                'student_id': student_id,
                'assistant_index': idx,
                'user_index': last_user_index,
                'question': str(last_user or '').strip(),
                'answer': str(msg.get('content', '') or '').strip(),
                'debug': debug,
                'route': str(debug.get('route', '') or ''),
                'domain': str((debug.get('plan') or {}).get('domain', '') or ''),
            }
    return turns


def _is_rlhf_eligible(turn: Dict[str, Any]) -> bool:
    debug = turn.get('debug') or {}
    if not is_feedback_eligible(debug):
        return False
    question = _norm(turn.get('question', ''))
    answer = _norm(turn.get('answer', ''))
    return bool(question and answer)


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    label_rank = {'preferred': 3, 'rejected': 2, 'original': 1}
    best_by_text: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        text = _norm(item.get('text', ''))
        if not text:
            continue
        current = {**item, 'text': text}
        existing = best_by_text.get(text)
        if not existing:
            best_by_text[text] = current
            continue
        current_key = (label_rank.get(str(current.get('label', '') or ''), 0), float(current.get('score', 0.0)))
        existing_key = (label_rank.get(str(existing.get('label', '') or ''), 0), float(existing.get('score', 0.0)))
        if current_key > existing_key:
            best_by_text[text] = current
    return sorted(best_by_text.values(), key=lambda x: (label_rank.get(str(x.get('label', '') or ''), 0), float(x.get('score', 0.0))), reverse=True)


def _candidate(label: str, text: str, score: float, source: str, meta: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        'label': label,
        'text': _norm(text),
        'score': float(score),
        'source': source,
        'meta': meta or {},
    }


def _collect_turn_meta(
    turn: Dict[str, Any],
    reviews: List[Dict[str, Any]],
    feedback: Dict[str, Any],
    contributions: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    review_verdicts = sorted({
        str(item.get('verdict', '') or '').strip()
        for item in reviews
        if str(item.get('verdict', '') or '').strip()
    })
    review_kinds = sorted({
        str(item.get('review_kind', '') or '').strip()
        for item in reviews
        if str(item.get('review_kind', '') or '').strip()
    })
    candidate_sources = sorted({
        str(item.get('source', '') or '').strip()
        for item in candidates
        if str(item.get('source', '') or '').strip()
    })
    feedback_rating = int(feedback.get('rating', 0) or 0)
    return {
        'domain': str(turn.get('domain', '') or ''),
        'route': str(turn.get('route', '') or ''),
        'feedback_rating': feedback_rating,
        'has_feedback': bool(feedback),
        'review_verdicts': review_verdicts,
        'review_kinds': review_kinds,
        'has_review': bool(reviews),
        'contribution_count': len(contributions),
        'candidate_sources': candidate_sources,
    }


def _build_candidates(turn: Dict[str, Any], reviews: List[Dict[str, Any]], feedback: Dict[str, Any], contributions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    original_answer = str(turn.get('answer', '') or '').strip()
    if original_answer:
        candidates.append(_candidate('original', original_answer, 0.4, 'history', {'route': turn.get('route', ''), 'domain': turn.get('domain', '')}))

    for review in reviews:
        approved = str(review.get('approved_answer', '') or '').strip()
        verdict = str(review.get('verdict', '') or '').strip()
        if approved:
            score = 0.98 if verdict in {'correct', 'missing', 'partial'} else 0.96
            candidates.append(_candidate('preferred', approved, score, 'review', {'turn_id': review.get('turn_id', ''), 'verdict': verdict}))
        rejected = str(review.get('assistant_answer', '') or '').strip()
        if rejected and verdict in {'incorrect', 'excessive', 'missing', 'partial'}:
            candidates.append(_candidate('rejected', rejected, 0.1, 'review', {'turn_id': review.get('turn_id', ''), 'verdict': verdict}))

    rating = int(feedback.get('rating', 0) or 0)
    suggestion = str(feedback.get('answer_suggestion', '') or '').strip()
    if suggestion:
        score = 0.88 if rating <= 2 else 0.7
        candidates.append(_candidate('preferred', suggestion, score, 'feedback', {'rating': rating}))
    if rating >= 4 and original_answer:
        candidates.append(_candidate('preferred', original_answer, 0.85, 'feedback', {'rating': rating}))
    if rating <= 2 and original_answer:
        candidates.append(_candidate('rejected', original_answer, 0.15, 'feedback', {'rating': rating}))

    for contrib in contributions:
        text = str(contrib.get('user_contribution', '') or '').strip()
        if text:
            candidates.append(_candidate('preferred', text, 0.88, 'contribution', {'contribution_id': contrib.get('contribution_id', '')}))

    return _dedupe_candidates(candidates)


def _reviews_for_turn(turn: Dict[str, Any], all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    question = _norm(turn.get('question', ''))
    turn_id = str(turn.get('turn_id', '') or '')
    matches = []
    for item in all_items:
        if str(item.get('review_kind', 'answer') or 'answer') != 'answer':
            continue
        if str(item.get('turn_id', '') or '') == turn_id:
            matches.append(item)
            continue
        variants = [str(item.get('user_question', '') or '').strip(), *[str(x or '').strip() for x in item.get('match_questions', []) or []]]
        if question and question in variants:
            matches.append(item)
    return matches


def _feedback_for_turn(turn_id: str, all_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    for item in all_items:
        if str(item.get('turn_id', '') or '') == turn_id:
            return item
    return {}


def _contributions_for_turn(turn: Dict[str, Any], all_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    question = _norm(turn.get('question', ''))
    domain = str(turn.get('domain', '') or '')
    out = []
    for item in all_items:
        if str(item.get('status', '') or '') != 'approved':
            continue
        if question and _norm(item.get('original_question', '')) == question:
            out.append(item)
            continue
        if domain and str(item.get('domain_hint', '') or '') == domain and question and _norm(item.get('original_question', '')) == question:
            out.append(item)
    return out


def _to_preference_row(turn: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferred = [item for item in candidates if item.get('label') == 'preferred']
    rejected = [item for item in candidates if item.get('label') == 'rejected']
    if not preferred or not rejected:
        return {}
    chosen = preferred[0]
    loser = rejected[0]
    return {
        'turn_id': turn.get('turn_id', ''),
        'student_id': turn.get('student_id', ''),
        'question': turn.get('question', ''),
        'chosen': chosen.get('text', ''),
        'rejected': loser.get('text', ''),
        'meta': {
            'domain': turn.get('domain', ''),
            'route': turn.get('route', ''),
            'chosen_source': chosen.get('source', ''),
            'rejected_source': loser.get('source', ''),
        },
    }


def _to_sft_row(turn: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    preferred = [item for item in candidates if item.get('label') == 'preferred']
    if not preferred:
        return {}
    target = preferred[0]
    text = str(target.get('text', '') or '').strip()
    if not text:
        return {}
    return {
        'turn_id': turn.get('turn_id', ''),
        'student_id': turn.get('student_id', ''),
        'prompt': turn.get('question', ''),
        'response': text,
        'meta': {
            'domain': turn.get('domain', ''),
            'route': turn.get('route', ''),
            'source': target.get('source', ''),
            'label': target.get('label', ''),
        },
    }


def build_rlhf_dataset(output_prefix: str = 'dataset') -> Dict[str, Any]:
    turns = _history_turns()
    reviews = all_reviews()
    feedback_items = list_feedback()
    contributions = list_contributions()

    candidate_rows: List[Dict[str, Any]] = []
    preference_rows: List[Dict[str, Any]] = []
    sft_rows: List[Dict[str, Any]] = []

    for turn in turns.values():
        if not _is_rlhf_eligible(turn):
            continue
        turn_reviews = _reviews_for_turn(turn, reviews)
        turn_feedback = _feedback_for_turn(str(turn.get('turn_id', '') or ''), feedback_items)
        turn_contributions = _contributions_for_turn(turn, contributions)
        candidates = _build_candidates(turn, turn_reviews, turn_feedback, turn_contributions)
        if not candidates:
            continue
        turn_meta = _collect_turn_meta(turn, turn_reviews, turn_feedback, turn_contributions, candidates)
        candidate_rows.append({
            'turn_id': turn.get('turn_id', ''),
            'student_id': turn.get('student_id', ''),
            'question': turn.get('question', ''),
            'domain': turn.get('domain', ''),
            'route': turn.get('route', ''),
            'meta': turn_meta,
            'candidates': candidates,
        })
        pref = _to_preference_row(turn, candidates)
        if pref:
            pref['meta'].update(turn_meta)
            preference_rows.append(pref)
        sft = _to_sft_row(turn, candidates)
        if sft:
            sft['meta'].update(turn_meta)
            sft_rows.append(sft)

    RLHF_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = RLHF_DIR / f'{output_prefix}_candidates.jsonl'
    preferences_path = RLHF_DIR / f'{output_prefix}_preferences.jsonl'
    sft_path = RLHF_DIR / f'{output_prefix}_sft.jsonl'
    summary_path = RLHF_DIR / f'{output_prefix}_summary.json'

    for path, rows in [
        (candidates_path, candidate_rows),
        (preferences_path, preference_rows),
        (sft_path, sft_rows),
    ]:
        path.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')

    summary = {
        'eligible_turns': len(candidate_rows),
        'preference_rows': len(preference_rows),
        'sft_rows': len(sft_rows),
        'candidates_path': str(candidates_path),
        'preferences_path': str(preferences_path),
        'sft_path': str(sft_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build RLHF-style datasets from UNETI chat history, reviews, feedback, and contributions.')
    parser.add_argument('--output-prefix', default='dataset', help='Prefix for output files inside storage/rlhf')
    args = parser.parse_args()
    summary = build_rlhf_dataset(output_prefix=args.output_prefix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
