from __future__ import annotations

import re
from typing import Any, Dict, List

TOKEN_RE = re.compile(r'[A-Za-z0-9@._:/-]+')


def important_tokens(text: str) -> set[str]:
    keep = set()
    for tok in TOKEN_RE.findall(text):
        lower = tok.lower()
        if '@' in lower or any(ch.isdigit() for ch in lower) or lower.startswith('http') or len(lower) >= 6:
            keep.add(lower)
    return keep


def answer_guard(answer: str, contexts: List[Dict[str, Any]]) -> bool:
    if not answer or not contexts:
        return False
    ctx = ' '.join(c.get('text', '') for c in contexts)
    answer_tokens = important_tokens(answer)
    ctx_tokens = important_tokens(ctx)

    exact_lock_tokens = {tok for tok in answer_tokens if '@' in tok or tok.startswith('http') or any(ch.isdigit() for ch in tok)}
    if any(tok not in ctx_tokens for tok in exact_lock_tokens):
        return False

    missing = [tok for tok in answer_tokens if tok not in ctx_tokens]
    if not answer_tokens:
        return True
    return len(missing) <= max(4, len(answer_tokens) // 3)
