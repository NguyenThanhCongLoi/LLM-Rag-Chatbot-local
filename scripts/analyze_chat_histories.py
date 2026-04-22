from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.routing import detect_query_family
HISTORY_DIR = ROOT / 'storage' / 'chat_histories'


def _load_debug(message: dict) -> dict:
    debug = message.get('debug')
    if isinstance(debug, dict):
        return debug
    if isinstance(debug, str):
        try:
            return json.loads(debug)
        except Exception:
            return {}
    return {}


def _family(question: str, route: str, domain: str, answer: str) -> str:
    q = (question or '').lower()
    a = (answer or '').lower()
    detected = detect_query_family(question or '', {})
    detected_family = str(detected.get('family', '') or '')
    if detected_family == 'meta' and route != 'meta':
        return 'meta_missed'
    if detected_family == 'clarification' and route != 'clarification':
        return str(detected.get('detail', '') or 'clarification_missed')
    if detected_family == 'policy' and route != 'policy_block':
        return str(detected.get('policy_code', '') or 'policy_missed')
    if detected_family == 'sensitive' and route != 'sensitive_block':
        return str(detected.get('policy_code', '') or 'sensitive_missed')
    if detected_family == 'domain' and domain == 'general_docs':
        return 'general_docs_drift'
    if detected_family == 'out_of_scope' and route != 'out_of_scope':
        return 'out_of_scope_missed'
    if any(h in q for h in ['gioi thieu ve truong', 'noi ve truong', 'thong tin ve truong', 'tong quan ve truong']):
        return 'summary_school_overview'
    if route == 'out_of_scope' and 'uneti' in q:
        return 'out_of_scope_with_uneti_keyword'
    if route == 'out_of_scope' and 'tôi chỉ hỗ trợ' in a:
        return 'generic_out_of_scope'
    if detected_family == 'clarification':
        return 'ambiguous_follow_up'
    if any(h in q for h in ['do ai day', 'ai day mon', 'giang vien nao day']):
        return 'teaching_assignment'
    if any(h in q for h in ['bao nhieu lop', 'bao nhieu sinh vien', 'si so']):
        return 'class_or_student_detail'
    return 'other'


def main() -> None:
    files = sorted(HISTORY_DIR.glob('*.json'))
    route_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    policy_counter: Counter[str] = Counter()
    question_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    file_samples: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    total_messages = 0
    total_turns = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            print(f'[skip] {path.name}: {exc}')
            continue
        messages = payload.get('messages') or []
        total_messages += len(messages)
        total_turns += len(messages) // 2
        for i, message in enumerate(messages):
            if message.get('role') != 'assistant':
                continue
            debug = _load_debug(message)
            route = str(debug.get('route', '') or '')
            policy = str(debug.get('policy_code', '') or '')
            plan = debug.get('plan') or {}
            domain = str(plan.get('domain', '') or '')
            question = str(debug.get('question', '') or '').strip()
            answer = str(message.get('content', '') or '').strip()

            if route:
                route_counter[route] += 1
            if policy:
                policy_counter[policy] += 1
            if domain:
                domain_counter[domain] += 1
            if question:
                question_counter[question] += 1
            family = _family(question, route, domain, answer)
            family_counter[family] += 1

            if route or policy:
                file_samples[path.name].append((question[:100], route or '-', (policy or domain or '-')[:60]))

    print('=== CHAT HISTORY REPORT ===')
    print(f'Files: {len(files)}')
    print(f'Messages: {total_messages}')
    print(f'Turns: {total_turns}')
    print()
    print('Top routes:')
    for name, count in route_counter.most_common(12):
        print(f'- {name}: {count}')
    print()
    print('Top policy codes:')
    for name, count in policy_counter.most_common(12):
        print(f'- {name}: {count}')
    print()
    print('Top domains:')
    for name, count in domain_counter.most_common(12):
        print(f'- {name}: {count}')
    print()
    print('Top repeated questions:')
    for text, count in question_counter.most_common(12):
        print(f'- {count}x: {text}')
    print()
    print('Top error families:')
    for name, count in family_counter.most_common(12):
        print(f'- {name}: {count}')
    print()
    print('Per-file samples:')
    for filename in sorted(file_samples):
        print(f'- {filename}')
        for question, route, extra in file_samples[filename][:6]:
            print(f'  route={route} extra={extra} q={question}')


if __name__ == '__main__':
    main()
