from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from .config import CONTRIBUTION_DB_PATH
from .normalize import norm_text_ascii
from .routing import detect_meta_query, is_control, is_low_signal_query


DEFAULT_CONTRIBUTION_DB = {'items': []}
PENDING_STATUSES = ['pending', 'approved', 'rejected']

QUESTIONISH_HINTS = [
    'la ai', 'la gi', 'o dau', 'nhu the nao', 'cach', 'bao nhieu', 'co khong',
    'giup', 'hoi', 'cho em hoi', 'cho minh hoi',
]
DECLINE_HINTS = [
    'toi khong biet', 'em khong biet', 'minh khong biet', 'khong biet',
    'khong ro', 'ko biet', 'khong co', 'chiu', 'thoi',
]
CONTRIBUTION_CUES = [
    'la ', 'email', '@', 'website', 'http', 'https', 'dia chi', 'ha noi', 'nam dinh',
    'linh nam', 'minh khai', 'tran hung dao', 'truong khoa', 'pho truong khoa',
    'truong phong', 'pho truong phong', 'hieu truong', 'pho hieu truong',
]
MISSING_INFO_HINTS = [
    'chua thay thong tin',
    'khong co trong tri thuc',
    'thieu thong tin',
    'chua co du lieu',
    'chua the xac nhan',
]


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def load_contribution_db() -> Dict[str, Any]:
    if not CONTRIBUTION_DB_PATH.exists():
        return dict(DEFAULT_CONTRIBUTION_DB)
    try:
        data = json.loads(CONTRIBUTION_DB_PATH.read_text(encoding='utf-8'))
        items = data.get('items', [])
        if not isinstance(items, list):
            items = []
        return {'items': items}
    except Exception:
        return dict(DEFAULT_CONTRIBUTION_DB)


def save_contribution_db(data: Dict[str, Any]) -> None:
    payload = {'items': data.get('items', [])}
    CONTRIBUTION_DB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def list_contributions(status: str = '') -> List[Dict[str, Any]]:
    items = list(load_contribution_db().get('items', []))
    if status:
        items = [item for item in items if str(item.get('status', '') or '') == status]
    items.sort(key=lambda x: str(x.get('updated_at', '') or ''), reverse=True)
    return items


def submit_contribution(payload: Dict[str, Any]) -> Dict[str, Any]:
    db = load_contribution_db()
    items = db.get('items', [])
    contribution_id = str(payload.get('contribution_id', '') or '').strip() or f"contrib:{len(items) + 1}:{int(datetime.utcnow().timestamp())}"
    item = {
        'contribution_id': contribution_id,
        'student_id': str(payload.get('student_id', '') or '').strip(),
        'original_question': str(payload.get('original_question', '') or '').strip(),
        'bot_answer': str(payload.get('bot_answer', '') or '').strip(),
        'user_contribution': str(payload.get('user_contribution', '') or '').strip(),
        'domain_hint': str(payload.get('domain_hint', '') or '').strip(),
        'route': str(payload.get('route', '') or '').strip(),
        'status': 'pending',
        'admin_note': '',
        'linked_review_id': '',
        'created_at': _utc_now(),
        'updated_at': _utc_now(),
    }
    items.append(item)
    save_contribution_db(db)
    return item


def update_contribution(contribution_id: str, **changes: Any) -> Dict[str, Any]:
    db = load_contribution_db()
    items = db.get('items', [])
    for idx, item in enumerate(items):
        if str(item.get('contribution_id', '') or '') != contribution_id:
            continue
        updated = dict(item)
        updated.update(changes)
        status = str(updated.get('status', 'pending') or 'pending')
        updated['status'] = status if status in PENDING_STATUSES else 'pending'
        updated['updated_at'] = _utc_now()
        items[idx] = updated
        save_contribution_db(db)
        return updated
    return {}


def should_offer_contribution(answer: str, debug: Dict[str, Any]) -> bool:
    route = str(debug.get('route', '') or '')
    if route in {'policy_block', 'sensitive_block', 'out_of_scope', 'meta', 'clarification', 'control'}:
        return False
    answer_norm = norm_text_ascii(answer)
    if not answer_norm:
        return False
    if any(hint in answer_norm for hint in [
        'thong tin ca nhan rieng tu',
        'ngoai pham vi chatbot',
        'khong cung cap',
        'vuot qua muc chi tiet chatbot duoc phep xac nhan',
    ]):
        return False
    return any(hint in answer_norm for hint in MISSING_INFO_HINTS)


def contribution_invitation(answer: str) -> str:
    base = str(answer or '').strip()
    invite = 'Nếu bạn có thông tin chính xác để bổ sung, bạn có thể gửi ngay tại đây. Tôi sẽ cảm ơn bạn và chuyển cho admin kiểm tra lại trước khi dùng.'
    if not base:
        return invite
    if invite in base:
        return base
    return f'{base}\n\n{invite}'


def classify_contribution_reply(text: str) -> str:
    q = norm_text_ascii(text)
    if not q:
        return 'other'
    if any(hint == q or hint in q for hint in DECLINE_HINTS):
        return 'decline'
    if is_control(q) or detect_meta_query(q) or is_low_signal_query(q):
        return 'other'
    if '?' in text:
        return 'other'
    if any(hint in q for hint in QUESTIONISH_HINTS):
        return 'other'
    tokens = q.split()
    if len(tokens) < 3:
        return 'other'
    if any(cue in q for cue in CONTRIBUTION_CUES):
        return 'provide'
    if len(tokens) >= 6:
        return 'provide'
    return 'other'


def contribution_thanks_reply() -> str:
    return 'Cảm ơn bạn đã bổ sung thông tin. Tôi đã ghi nhận và chuyển cho admin kiểm tra, xác nhận lại trước khi dùng cho chatbot.'


def contribution_decline_reply() -> str:
    return 'Không sao. Cảm ơn bạn đã phản hồi. Bạn có thể tiếp tục hỏi nội dung khác về UNETI.'
