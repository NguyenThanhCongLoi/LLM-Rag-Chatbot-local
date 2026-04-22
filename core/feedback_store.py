from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .config import FEEDBACK_DB_PATH


DEFAULT_FEEDBACK_DB = {'items': []}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def load_feedback_db() -> Dict[str, Any]:
    if not FEEDBACK_DB_PATH.exists():
        return dict(DEFAULT_FEEDBACK_DB)
    try:
        data = json.loads(FEEDBACK_DB_PATH.read_text(encoding='utf-8'))
        items = data.get('items', [])
        if not isinstance(items, list):
            items = []
        return {'items': items}
    except Exception:
        return dict(DEFAULT_FEEDBACK_DB)


def save_feedback_db(data: Dict[str, Any]) -> None:
    payload = {'items': data.get('items', [])}
    FEEDBACK_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def get_feedback(turn_id: str) -> Dict[str, Any]:
    for item in load_feedback_db().get('items', []):
        if str(item.get('turn_id', '') or '') == turn_id:
            return dict(item)
    return {}


def upsert_feedback(item: Dict[str, Any]) -> Dict[str, Any]:
    db = load_feedback_db()
    items = db.get('items', [])
    turn_id = str(item.get('turn_id', '') or '').strip()
    if not turn_id:
        raise ValueError('turn_id is required')

    payload = {
        'turn_id': turn_id,
        'student_id': str(item.get('student_id', '') or '').strip(),
        'assistant_index': int(item.get('assistant_index', 0) or 0),
        'question': str(item.get('question', '') or '').strip(),
        'answer': str(item.get('answer', '') or '').strip(),
        'rating': max(1, min(5, int(item.get('rating', 3) or 3))),
        'answer_suggestion': str(item.get('answer_suggestion', '') or '').strip(),
        'notes': str(item.get('notes', '') or '').strip(),
        'route': str(item.get('route', '') or '').strip(),
        'domain': str(item.get('domain', '') or '').strip(),
        'updated_at': _utc_now(),
    }

    for idx, existing in enumerate(items):
        if str(existing.get('turn_id', '') or '') != turn_id:
            continue
        payload['created_at'] = str(existing.get('created_at', '') or payload['updated_at'])
        items[idx] = payload
        save_feedback_db(db)
        return payload

    payload['created_at'] = payload['updated_at']
    items.append(payload)
    save_feedback_db(db)
    return payload


def list_feedback(student_id: str = '') -> List[Dict[str, Any]]:
    items = list(load_feedback_db().get('items', []))
    if student_id:
        items = [item for item in items if str(item.get('student_id', '') or '') == student_id]
    items.sort(key=lambda x: str(x.get('updated_at', '') or ''), reverse=True)
    return items


def is_feedback_eligible(debug: Dict[str, Any]) -> bool:
    route = str(debug.get('route', '') or '')
    if route in {'meta', 'out_of_scope', 'control', 'clarification', 'web_notice_only'}:
        return False
    query_family = debug.get('query_family') or {}
    family = str(query_family.get('family', '') or '')
    if family in {'meta', 'out_of_scope', 'clarification'}:
        return False
    if route in {'policy_block', 'sensitive_block', 'partial_scope', 'admin_review_override'}:
        return True
    domain = str((debug.get('plan') or {}).get('domain', '') or '')
    return bool(domain and domain != 'general_docs')
