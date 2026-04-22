import json
from pathlib import Path
from typing import Any, Dict, List
from .config import HISTORY_DIR

DEFAULT_MEMORY = {
    'active_domain': '',
    'active_doc': '',
    'last_entity': '',
    'last_named_unit': '',
    'last_question_type': 'factoid',
    'last_retrieved_ids': [],
    'last_user_query': '',
    'last_assistant_answer': '',
    'last_topic': '',
    'last_channel': 'docs',
    'context_turns': 0,
    'awaiting_user_contribution': False,
    'pending_contribution_question': '',
    'pending_contribution_answer': '',
    'pending_contribution_domain': '',
}

def history_path(student_id: str) -> Path:
    safe = ''.join(ch for ch in str(student_id) if ch.isalnum() or ch in ('-', '_')).strip() or 'anonymous'
    return HISTORY_DIR / f'{safe}.json'

def load_history(student_id: str) -> Dict[str, Any]:
    path = history_path(student_id)
    if not path.exists():
        return {'messages': [], 'memory': dict(DEFAULT_MEMORY)}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        mem = dict(DEFAULT_MEMORY)
        mem.update(data.get('memory', {}))
        return {'messages': data.get('messages', []), 'memory': mem}
    except Exception:
        return {'messages': [], 'memory': dict(DEFAULT_MEMORY)}

def save_history(student_id: str, messages: List[Dict[str, Any]], memory: Dict[str, Any]) -> None:
    path = history_path(student_id)
    path.write_text(json.dumps({'messages': messages, 'memory': memory}, ensure_ascii=False, indent=2), encoding='utf-8')
