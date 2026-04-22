from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import RLHF_DIR
from core.feedback_store import list_feedback
from core.history import load_history
from core.llm import LLMConfig
from core.normalize import norm_text_ascii
from core.pipeline_v4 import UnetiDocumentAgentV4Max, _domain_doc_tiers
from core.review_store import all_reviews, list_history_student_ids
from core.seed_loader import load_seed_chunks


DOC_ID_TO_DOMAIN = {
    'ban-giam-hieu': 'ban_giam_hieu',
    'co-so-vat-chat': 'co_so_vat_chat',
    'danh-sach-cac-thanh-vien-hoi-dong-truong': 'hoi_dong_truong',
    'huong-dan-chuc-nang-cong-thong-tin-sv': 'portal_howto',
    'khoa-chuyen-mon': 'khoa_chuyen_mon',
    'lich-su-hinh-thanh': 'lich_su_hinh_thanh',
    'phong-ban-va-chuc-nang': 'phong_ban_va_chuc_nang',
}
BAD_HISTORY_ROUTES = {
    'admin_review_override',
    'clarification',
    'control',
    'meta',
    'out_of_scope',
    'partial_scope',
    'policy_block',
    'sensitive_block',
    'web_notice_only',
}
QUERY_COMMON_TOKENS = {
    'ai', 'bao', 'chi', 'cho', 'co', 'cua', 'cach', 'dang', 'dau', 'dia', 'em', 'gi', 'giup',
    'hoc', 'hoi', 'ky', 'la', 'lien', 'minh', 'nao', 'nay', 'nhieu', 'nhu', 'o', 'phan', 'tai',
    'the', 'thong', 'tin', 'toi', 'truong', 'uneti', 've', 'vien', 'xem', 'huong', 'dan', 'sinh',
    'khoa', 'phong',
}
QUERY_TYPE_MARKERS = {
    'contact': ['email', 'mail', 'website', 'web', 'lien he', 'dien thoai', 'sdt'],
    'location': ['dia chi', 'o dau', 'co so', 'ha noi', 'nam dinh', 'minh khai', 'linh nam', 'tran hung dao'],
    'role': ['la ai', 'hieu truong', 'pho hieu truong', 'truong khoa', 'truong phong', 'chu tich'],
    'howto': ['cach', 'huong dan', 'dang nhap', 'dang ky', 'tra cuu', 'xem ', 'vao ', 'chon '],
    'history': ['lich su', 'thanh lap', 'tien than', 'giai doan'],
    'function': ['lam gi', 'chuc nang', 'nhiem vu', 'phu trach'],
}
QUERY_TYPE_TO_CHUNK_TYPES = {
    'contact': {'contact', 'location'},
    'location': {'location', 'contact', 'overview'},
    'role': {'role', 'record'},
    'howto': {'howto'},
    'history': {'history', 'overview'},
    'function': {'function', 'duty', 'overview'},
    'generic': {'overview', 'fact', 'contact', 'location', 'role', 'history', 'function', 'duty', 'howto', 'record'},
}


def _norm(text: str) -> str:
    return ' '.join(str(text or '').split()).strip()


def _hash_id(*parts: str, prefix: str) -> str:
    sha = hashlib.sha1()
    for part in parts:
        sha.update(str(part or '').encode('utf-8', errors='ignore'))
        sha.update(b'\0')
    return f'{prefix}-{sha.hexdigest()[:16]}'


def _load_debug(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _dedupe_preserve(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = str(item or '').strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def _write_beir_qrels(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as fh:
        fh.write('query-id\tcorpus-id\tscore\n')
        for row in rows:
            fh.write(f"{row.get('query_id', '')}\t{row.get('corpus_id', '')}\t{row.get('label', 0)}\n")


def _split_for_id(value: str, val_ratio: float) -> str:
    bucket = int(hashlib.sha1(value.encode('utf-8', errors='ignore')).hexdigest()[:8], 16) % 10000
    return 'val' if bucket < int(max(0.0, min(1.0, val_ratio)) * 10000) else 'train'


def _query_type(query: str) -> str:
    query_ascii = norm_text_ascii(query)
    for query_type, markers in QUERY_TYPE_MARKERS.items():
        if any(marker in query_ascii for marker in markers):
            return query_type
    return 'generic'


def _seed_domain_from_doc_id(doc_id: str) -> str:
    if str(doc_id).startswith('seed-'):
        return str(doc_id)[5:]
    return ''


def _chunk_row_from_seed(chunk: Dict[str, Any]) -> Dict[str, Any]:
    doc_id = str(chunk.get('doc_id', '') or '')
    metadata = dict(chunk.get('metadata', {}) or {})
    return {
        'id': str(chunk.get('chunk_id', '') or ''),
        'doc_id': doc_id,
        'domain': _seed_domain_from_doc_id(doc_id),
        'title': str(chunk.get('title', '') or ''),
        'text': str(chunk.get('text', '') or ''),
        'kind': str(chunk.get('kind', 'chunk') or 'chunk'),
        'chunk_type': str(chunk.get('chunk_type', '') or ''),
        'entity_type': str(chunk.get('entity_type', '') or ''),
        'entity_name': str(chunk.get('entity_name', '') or ''),
        'priority': int(chunk.get('priority', 0) or 0),
        'summary': str(chunk.get('summary', '') or ''),
        'dedupe_key': str(chunk.get('dedupe_key', '') or ''),
        'parent_chunk_id': str(chunk.get('parent_chunk_id', '') or ''),
        'source_kind': str(chunk.get('source_kind', 'seed') or 'seed'),
        'is_authoritative': bool(chunk.get('is_authoritative', True)),
        'section': str(chunk.get('section', '') or ''),
        'keywords': list(chunk.get('keywords', []) or []),
        'keyword_ids': list(chunk.get('keyword_ids', []) or []),
        'source_url': str(chunk.get('source_url', '') or ''),
        'metadata': metadata,
    }


def _chunk_row_from_doc(chunk: Any) -> Dict[str, Any]:
    metadata = dict(getattr(chunk, 'metadata', {}) or {})
    doc_id = str(getattr(chunk, 'doc_id', '') or '')
    return {
        'id': str(getattr(chunk, 'chunk_id', '') or ''),
        'doc_id': doc_id,
        'domain': DOC_ID_TO_DOMAIN.get(doc_id, ''),
        'title': str(getattr(chunk, 'title', '') or ''),
        'text': str(getattr(chunk, 'text', '') or ''),
        'kind': 'chunk',
        'chunk_type': str(getattr(chunk, 'chunk_type', '') or ''),
        'entity_type': str(getattr(chunk, 'entity_type', '') or ''),
        'entity_name': str(getattr(chunk, 'entity_name', '') or ''),
        'priority': int(getattr(chunk, 'priority', 0) or 0),
        'summary': str(getattr(chunk, 'summary', '') or ''),
        'dedupe_key': str(getattr(chunk, 'dedupe_key', '') or ''),
        'parent_chunk_id': str(getattr(chunk, 'parent_chunk_id', '') or ''),
        'source_kind': str(getattr(chunk, 'source_kind', 'local') or 'local'),
        'is_authoritative': bool(getattr(chunk, 'is_authoritative', True)),
        'section': str(getattr(chunk, 'section', '') or ''),
        'keywords': list(metadata.get('keywords', []) or []),
        'keyword_ids': list(metadata.get('keyword_ids', []) or []),
        'source_url': str(metadata.get('source_url', '') or ''),
        'metadata': metadata,
    }


def _build_corpus(agent: UnetiDocumentAgentV4Max, include_web: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    corpus_rows: List[Dict[str, Any]] = []

    for chunk in load_seed_chunks():
        row = _chunk_row_from_seed(chunk)
        if row['id'] and row['text']:
            corpus_rows.append(row)

    doc_ids = agent.local_docs()
    if include_web:
        doc_ids += agent.web_docs()
    agent._ensure_docs_loaded(doc_ids)
    for doc_id in doc_ids:
        for chunk in agent.store.doc_chunks.get(doc_id, []) or []:
            row = _chunk_row_from_doc(chunk)
            if row['id'] and row['text']:
                corpus_rows.append(row)

    by_id: Dict[str, Dict[str, Any]] = {}
    for row in corpus_rows:
        by_id[str(row['id'])] = row
    return corpus_rows, by_id


def _build_corpus_lookup(corpus_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    by_dedupe: Dict[str, List[str]] = defaultdict(list)
    by_text: Dict[str, List[str]] = defaultdict(list)
    by_doc_text: Dict[str, List[str]] = defaultdict(list)
    by_question: Dict[str, List[str]] = defaultdict(list)
    by_title: Dict[str, List[str]] = defaultdict(list)

    for row in corpus_rows:
        row_id = str(row.get('id', '') or '')
        if not row_id:
            continue
        dedupe_key = str(row.get('dedupe_key', '') or '').strip()
        text_norm = norm_text_ascii(row.get('text', ''))
        doc_key = f"{row.get('doc_id', '')}::{text_norm}"
        if dedupe_key:
            by_dedupe[dedupe_key].append(row_id)
        if text_norm:
            by_text[text_norm].append(row_id)
            by_doc_text[doc_key].append(row_id)
        title_norm = norm_text_ascii(row.get('title', ''))
        if title_norm:
            by_title[title_norm].append(row_id)
        question_norm = norm_text_ascii((row.get('metadata') or {}).get('question', ''))
        if question_norm:
            by_question[question_norm].append(row_id)

    return {
        'by_dedupe': {k: _dedupe_preserve(v) for k, v in by_dedupe.items()},
        'by_text': {k: _dedupe_preserve(v) for k, v in by_text.items()},
        'by_doc_text': {k: _dedupe_preserve(v) for k, v in by_doc_text.items()},
        'by_question': {k: _dedupe_preserve(v) for k, v in by_question.items()},
        'by_title': {k: _dedupe_preserve(v) for k, v in by_title.items()},
    }


def _history_turns() -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for student_id in list_history_student_ids():
        hist = load_history(student_id)
        messages = hist.get('messages', []) or []
        last_user = ''
        last_user_index = -1
        for idx, msg in enumerate(messages):
            role = str(msg.get('role', '') or '')
            if role == 'user':
                last_user = str(msg.get('content', '') or '').strip()
                last_user_index = idx
                continue
            if role != 'assistant':
                continue
            debug = _load_debug(msg.get('debug'))
            turns.append({
                'turn_id': f'{student_id}:{idx}',
                'student_id': student_id,
                'assistant_index': idx,
                'user_index': last_user_index,
                'question': last_user,
                'answer': str(msg.get('content', '') or '').strip(),
                'debug': debug,
                'route': str(debug.get('route', '') or ''),
                'domain': str((debug.get('plan') or {}).get('domain', '') or ''),
                'contexts': list(debug.get('contexts', []) or []),
            })
    return turns


def _reviews_by_turn() -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in all_reviews():
        grouped[str(item.get('turn_id', '') or '')].append(item)
    return grouped


def _feedback_by_turn() -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in list_feedback():
        turn_id = str(item.get('turn_id', '') or '')
        if turn_id and turn_id not in grouped:
            grouped[turn_id] = item
    return grouped


def _history_turn_is_usable(turn: Dict[str, Any], reviews: List[Dict[str, Any]], feedback: Dict[str, Any]) -> Tuple[bool, str]:
    question = _norm(turn.get('question', ''))
    contexts = list(turn.get('contexts', []) or [])
    route = str(turn.get('route', '') or '')
    domain = str(turn.get('domain', '') or '')
    if not question:
        return False, 'empty_question'
    if not contexts:
        return False, 'no_contexts'
    if route in BAD_HISTORY_ROUTES:
        return False, f'bad_route:{route}'

    verdicts = {
        str(item.get('verdict', '') or '').strip()
        for item in reviews
        if str(item.get('review_kind', 'answer') or 'answer') == 'answer'
    }
    if verdicts & {'incorrect', 'excessive'}:
        return False, 'review_rejected'

    rating = int(feedback.get('rating', 0) or 0)
    if rating and rating <= 2:
        return False, 'feedback_rejected'

    top_score = max(float(ctx.get('score', 0.0) or 0.0) for ctx in contexts)
    confidence = 0
    if rating >= 4:
        confidence += 2
    elif rating == 3:
        confidence += 1
    if 'correct' in verdicts:
        confidence += 2
    elif verdicts & {'partial', 'missing'}:
        confidence += 1
    if top_score >= 0.75:
        confidence += 2
    elif top_score >= 0.55:
        confidence += 1
    if domain and domain != 'general_docs':
        confidence += 1

    if confidence < 2:
        return False, 'low_confidence'
    return True, 'ok'


def _query_quality(origin: str, meta: Dict[str, Any]) -> float:
    if origin == 'seed_qa':
        return 1.0
    score = 0.45
    feedback_rating = int(meta.get('feedback_rating', 0) or 0)
    if feedback_rating >= 4:
        score += 0.20
    elif feedback_rating == 3:
        score += 0.08
    verdicts = set(meta.get('review_verdicts', []) or [])
    if 'correct' in verdicts:
        score += 0.22
    elif verdicts & {'partial', 'missing'}:
        score += 0.10
    top_context_score = float(meta.get('top_context_score', 0.0) or 0.0)
    if top_context_score >= 1.0:
        score += 0.13
    elif top_context_score >= 0.7:
        score += 0.08
    elif top_context_score >= 0.5:
        score += 0.04
    return round(min(1.0, max(0.0, score)), 4)


def _resolve_context_ids(
    context: Dict[str, Any],
    corpus_by_id: Dict[str, Dict[str, Any]],
    corpus_lookup: Dict[str, Dict[str, List[str]]],
) -> List[str]:
    source_id = str(context.get('source_id', '') or '').strip()
    if source_id and source_id in corpus_by_id:
        return [source_id]

    metadata = dict(context.get('metadata', {}) or {})
    kind = str(context.get('kind', '') or '').strip()
    title = str(context.get('title', '') or '').strip()
    question = str(metadata.get('question', '') or '').strip()
    answer = str(metadata.get('answer', '') or '').strip()
    dedupe_key = str(metadata.get('dedupe_key', '') or '').strip()
    candidate_ids: List[str] = []

    if question:
        candidate_ids.extend(corpus_lookup['by_question'].get(norm_text_ascii(question), []))
    if title:
        candidate_ids.extend(corpus_lookup['by_title'].get(norm_text_ascii(title), []))
    if dedupe_key:
        candidate_ids.extend(corpus_lookup['by_dedupe'].get(dedupe_key, []))

    doc_id = str(metadata.get('doc_id', '') or '').strip()
    text_norm = norm_text_ascii(context.get('text', ''))
    if doc_id and text_norm:
        candidate_ids.extend(corpus_lookup['by_doc_text'].get(f'{doc_id}::{text_norm}', []))

    if text_norm:
        candidate_ids.extend(corpus_lookup['by_text'].get(text_norm, []))
    if answer:
        candidate_ids.extend(corpus_lookup['by_text'].get(norm_text_ascii(answer), []))

    candidate_ids = _dedupe_preserve(candidate_ids)
    if not candidate_ids:
        return []

    title_norm = norm_text_ascii(title)
    question_norm = norm_text_ascii(question)
    answer_norm = norm_text_ascii(answer)

    def _candidate_score(row_id: str) -> Tuple[float, str]:
        row = corpus_by_id.get(row_id, {})
        row_meta = dict(row.get('metadata', {}) or {})
        row_title = norm_text_ascii(row.get('title', ''))
        row_text = norm_text_ascii(row.get('text', ''))
        row_question = norm_text_ascii(row_meta.get('question', ''))
        score = 0.0
        if doc_id and str(row.get('doc_id', '') or '') == doc_id:
            score += 1.0
        if kind and str(row.get('kind', '') or '') == kind:
            score += 0.6
        if title_norm and row_title == title_norm:
            score += 0.9
        if question_norm and row_question == question_norm:
            score += 1.2
        if answer_norm and row_text == answer_norm:
            score += 1.0
        if text_norm and row_text == text_norm:
            score += 0.8
        return score, row_id

    ranked = sorted((_candidate_score(row_id) for row_id in candidate_ids), reverse=True)
    if not ranked:
        return []
    best_score = float(ranked[0][0])
    if best_score <= 0.0:
        return []
    return [str(ranked[0][1])]


def _history_positive_ids(
    contexts: List[Dict[str, Any]],
    corpus_by_id: Dict[str, Dict[str, Any]],
    corpus_lookup: Dict[str, Dict[str, List[str]]],
    max_positives: int,
) -> List[str]:
    if not contexts:
        return []
    top_score = float(contexts[0].get('score', 0.0) or 0.0)
    min_score = max(0.32, top_score * 0.68)
    positives: List[str] = []
    for idx, context in enumerate(contexts[:8]):
        score = float(context.get('score', 0.0) or 0.0)
        if idx > 0 and score < min_score:
            continue
        positives.extend(_resolve_context_ids(context, corpus_by_id, corpus_lookup))
        positives = _dedupe_preserve(positives)
        if len(positives) >= max_positives:
            return positives[:max_positives]
    return positives[:max_positives]


def _positive_matches_query(query: str, row: Dict[str, Any]) -> bool:
    query_ascii = norm_text_ascii(query)
    tokens = [
        tok for tok in query_ascii.split()
        if len(tok) >= 3 and tok not in QUERY_COMMON_TOKENS
    ]
    if not tokens:
        return True
    haystack = norm_text_ascii(' '.join([
        str(row.get('title', '') or ''),
        str(row.get('entity_name', '') or ''),
        str(row.get('text', '') or ''),
        str((row.get('metadata') or {}).get('question', '') or ''),
    ]))
    hits = sum(1 for tok in set(tokens) if tok in haystack)
    return hits >= 1


def _expand_positive_ids(
    query: str,
    positive_ids: List[str],
    corpus_rows: List[Dict[str, Any]],
    corpus_by_id: Dict[str, Dict[str, Any]],
    max_extra: int = 3,
) -> List[str]:
    if not positive_ids:
        return []
    query_type = _query_type(query)
    accepted_chunk_types = QUERY_TYPE_TO_CHUNK_TYPES.get(query_type, QUERY_TYPE_TO_CHUNK_TYPES['generic'])
    positives = _dedupe_preserve(positive_ids)
    positive_rows = [corpus_by_id[pid] for pid in positives if pid in corpus_by_id]
    if not positive_rows:
        return positives

    positive_entities = {
        norm_text_ascii(row.get('entity_name', ''))
        for row in positive_rows
        if norm_text_ascii(row.get('entity_name', ''))
    }
    positive_domains = {
        str(row.get('domain', '') or '')
        for row in positive_rows
        if str(row.get('domain', '') or '')
    }
    ranked: List[Tuple[float, str]] = []
    for row in corpus_rows:
        row_id = str(row.get('id', '') or '')
        if not row_id or row_id in positives:
            continue
        if str(row.get('domain', '') or '') not in positive_domains:
            continue
        if norm_text_ascii(row.get('entity_name', '')) not in positive_entities:
            continue
        chunk_type = str(row.get('chunk_type', '') or '')
        if chunk_type and chunk_type not in accepted_chunk_types:
            continue
        if not _positive_matches_query(query, row):
            continue
        score = 0.0
        if str(row.get('source_kind', '') or '') == 'seed':
            score += 0.25
        if chunk_type in accepted_chunk_types:
            score += 0.30
        if query_type != 'generic' and chunk_type == query_type:
            score += 0.35
        if norm_text_ascii(row.get('text', '')) and norm_text_ascii(row.get('text', '')) != norm_text_ascii(row.get('summary', '')):
            score += 0.05
        ranked.append((score, row_id))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, row_id in ranked[:max_extra]:
        positives.append(row_id)
    return _dedupe_preserve(positives)


def _seed_query_variants(chunk: Dict[str, Any], max_variants: int) -> List[str]:
    metadata = dict(chunk.get('metadata', {}) or {})
    question = _norm(metadata.get('question', ''))
    variants = [question] if question else []
    question_norm = norm_text_ascii(question)
    for keyword in list(chunk.get('keywords', []) or []):
        phrase = _norm(keyword)
        phrase_norm = norm_text_ascii(phrase)
        if len(phrase_norm) < 6 or phrase_norm == question_norm:
            continue
        if phrase_norm in question_norm or question_norm in phrase_norm:
            continue
        variants.append(phrase)
        if len(variants) >= max(1, max_variants + 1):
            break
    return _dedupe_preserve(variants)


def _build_seed_queries(corpus_rows: List[Dict[str, Any]], max_variants: int) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    for chunk in corpus_rows:
        if str(chunk.get('source_kind', '') or '') != 'seed':
            continue
        if str(chunk.get('kind', '') or '') != 'qa':
            continue
        variants = _seed_query_variants(chunk, max_variants=max_variants)
        if not variants:
            continue
        positive_ids = [str(chunk.get('id', '') or '')]
        for idx, query_text in enumerate(variants):
            query_type = _query_type(query_text)
            query_id = _hash_id('seed_qa', str(chunk.get('id', '')), str(idx), query_text, prefix='q')
            rows.append({
                'query_id': query_id,
                'query': query_text,
                'query_type': query_type,
                'origin': 'seed_qa',
                'origin_id': str(chunk.get('id', '') or ''),
                'domain': str(chunk.get('domain', '') or ''),
                'positives': positive_ids,
                'quality': 1.0,
                'meta': {
                    'kind': str(chunk.get('kind', '') or ''),
                    'chunk_type': str(chunk.get('chunk_type', '') or ''),
                    'entity_name': str(chunk.get('entity_name', '') or ''),
                    'title': str(chunk.get('title', '') or ''),
                    'variant_rank': idx,
                },
            })
            stats['seed_queries'] += 1
            if idx > 0:
                stats['seed_variant_queries'] += 1
    return rows, stats


def _build_history_queries(
    corpus_by_id: Dict[str, Dict[str, Any]],
    corpus_lookup: Dict[str, Dict[str, List[str]]],
    max_positives: int,
) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    reviews = _reviews_by_turn()
    feedback = _feedback_by_turn()

    for turn in _history_turns():
        stats['history_turns_seen'] += 1
        turn_id = str(turn.get('turn_id', '') or '')
        turn_reviews = reviews.get(turn_id, [])
        turn_feedback = feedback.get(turn_id, {})
        usable, reason = _history_turn_is_usable(turn, turn_reviews, turn_feedback)
        if not usable:
            stats[f'skip_{reason}'] += 1
            continue

        positives = _history_positive_ids(
            list(turn.get('contexts', []) or []),
            corpus_by_id=corpus_by_id,
            corpus_lookup=corpus_lookup,
            max_positives=max_positives,
        )
        positives = [
            positive_id for positive_id in positives
            if _positive_matches_query(str(turn.get('question', '') or ''), corpus_by_id.get(positive_id, {}))
        ]
        if not positives:
            stats['skip_no_positive_match'] += 1
            continue

        verdicts = sorted({
            str(item.get('verdict', '') or '').strip()
            for item in turn_reviews
            if str(item.get('review_kind', 'answer') or 'answer') == 'answer'
        })
        query_text = str(turn.get('question', '') or '').strip()
        meta = {
            'route': str(turn.get('route', '') or ''),
            'feedback_rating': int(turn_feedback.get('rating', 0) or 0),
            'review_verdicts': verdicts,
            'assistant_index': int(turn.get('assistant_index', 0) or 0),
            'top_context_score': float((turn.get('contexts', [{}]) or [{}])[0].get('score', 0.0) or 0.0),
        }
        query_id = _hash_id('history', turn_id, turn.get('question', ''), *positives, prefix='q')
        rows.append({
            'query_id': query_id,
            'query': query_text,
            'query_type': _query_type(query_text),
            'origin': 'history',
            'origin_id': turn_id,
            'domain': str(turn.get('domain', '') or ''),
            'positives': positives,
            'quality': _query_quality('history', meta),
            'meta': meta,
        })
        stats['history_queries'] += 1
    return rows, stats


def _merge_candidates(items: List[Any], top_k: int) -> List[Any]:
    merged = sorted(items, key=lambda item: float(item.score), reverse=True)
    out: List[Any] = []
    seen = set()
    for item in merged:
        metadata = getattr(item, 'metadata', {}) or {}
        dedupe_key = str(metadata.get('dedupe_key', '') or '').strip()
        key = ('dedupe', dedupe_key) if dedupe_key else ('id', str(getattr(item, 'source_id', '') or ''))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _retrieve_candidates(
    agent: UnetiDocumentAgentV4Max,
    query: str,
    domain: str,
    max_candidates: int,
) -> List[Any]:
    normalized_domain = domain or 'general_docs'
    seed_items = []
    if normalized_domain in agent.seed:
        seed_items = agent.store.seed_retrieve(query, normalized_domain, top_k=max_candidates)

    plan = {'domain': normalized_domain, 'intent': 'factoid'}
    doc_tiers = _domain_doc_tiers(normalized_domain, query, plan, agent.local_docs())
    doc_ids = _dedupe_preserve(doc_id for tier in doc_tiers for doc_id in tier)
    agent._ensure_docs_loaded(doc_ids)
    doc_items = agent.store.chunk_retrieve(query, doc_ids, top_k=max_candidates) if doc_ids else []
    return _merge_candidates(seed_items + doc_items, top_k=max_candidates)


def _token_overlap(query: str, text: str) -> float:
    query_tokens = [tok for tok in norm_text_ascii(query).split() if len(tok) >= 3]
    text_tokens = set(tok for tok in norm_text_ascii(text).split() if len(tok) >= 3)
    if not query_tokens or not text_tokens:
        return 0.0
    return sum(1 for tok in set(query_tokens) if tok in text_tokens) / max(1, len(set(query_tokens)))


def _candidate_negative_rows(
    query_row: Dict[str, Any],
    corpus_rows: List[Dict[str, Any]],
    corpus_by_id: Dict[str, Dict[str, Any]],
    positive_ids: List[str],
) -> List[Tuple[float, str, str]]:
    query = str(query_row.get('query', '') or '')
    query_type = str(query_row.get('query_type', '') or 'generic')
    positive_rows = [corpus_by_id[pid] for pid in positive_ids if pid in corpus_by_id]
    positive_entities = {
        norm_text_ascii(row.get('entity_name', ''))
        for row in positive_rows
        if norm_text_ascii(row.get('entity_name', ''))
    }
    positive_dedupes = {
        str(row.get('dedupe_key', '') or '').strip()
        for row in positive_rows
        if str(row.get('dedupe_key', '') or '').strip()
    }
    positives_domains = {
        str(row.get('domain', '') or '')
        for row in positive_rows
        if str(row.get('domain', '') or '')
    }

    ranked: List[Tuple[float, str, str]] = []
    for row in corpus_rows:
        row_id = str(row.get('id', '') or '')
        if not row_id or row_id in positive_ids:
            continue
        if str(row.get('dedupe_key', '') or '').strip() in positive_dedupes:
            continue
        row_entity = norm_text_ascii(row.get('entity_name', ''))
        row_domain = str(row.get('domain', '') or '')
        row_chunk_type = str(row.get('chunk_type', '') or '')

        score = _token_overlap(query, ' '.join([
            str(row.get('title', '') or ''),
            str(row.get('entity_name', '') or ''),
            str(row.get('text', '') or ''),
            str((row.get('metadata') or {}).get('question', '') or ''),
        ]))
        if score <= 0.0:
            continue
        source = 'lexical'
        if row_domain == str(query_row.get('domain', '') or ''):
            score += 0.18
            source = 'same_domain'
        if row_chunk_type == query_type:
            score += 0.16
            source = 'same_type'
        if row_entity and row_entity in positive_entities:
            score -= 0.22
        if positives_domains and row_domain in positives_domains:
            score += 0.05
        if score >= 0.18:
            ranked.append((score, row_id, source))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked


def _negative_ids_for_query(
    agent: UnetiDocumentAgentV4Max,
    query_row: Dict[str, Any],
    corpus_rows: List[Dict[str, Any]],
    corpus_by_id: Dict[str, Dict[str, Any]],
    max_negatives: int,
) -> List[Dict[str, Any]]:
    positive_ids = [pid for pid in query_row.get('positives', []) if pid in corpus_by_id]
    if not positive_ids:
        return []

    positive_dedupe = {
        str(corpus_by_id[pid].get('dedupe_key', '') or '').strip()
        for pid in positive_ids
        if str(corpus_by_id[pid].get('dedupe_key', '') or '').strip()
    }
    positive_entities = {
        norm_text_ascii(corpus_by_id[pid].get('entity_name', ''))
        for pid in positive_ids
        if norm_text_ascii(corpus_by_id[pid].get('entity_name', ''))
    }
    accepted_positive_types = QUERY_TYPE_TO_CHUNK_TYPES.get(
        str(query_row.get('query_type', '') or 'generic'),
        QUERY_TYPE_TO_CHUNK_TYPES['generic'],
    )
    negatives: List[Dict[str, Any]] = []
    seen_negative_ids = set()
    for item in _retrieve_candidates(
        agent,
        query=str(query_row.get('query', '') or ''),
        domain=str(query_row.get('domain', '') or ''),
        max_candidates=max(max_negatives * 6, 18),
    ):
        source_id = str(getattr(item, 'source_id', '') or '')
        if source_id not in corpus_by_id:
            continue
        if source_id in positive_ids:
            continue
        row = corpus_by_id[source_id]
        dedupe_key = str(row.get('dedupe_key', '') or '').strip()
        if dedupe_key and dedupe_key in positive_dedupe:
            continue
        row_entity = norm_text_ascii(row.get('entity_name', ''))
        row_chunk_type = str(row.get('chunk_type', '') or '')
        if row_entity and row_entity in positive_entities and row_chunk_type in accepted_positive_types:
            continue
        if source_id in seen_negative_ids:
            continue
        seen_negative_ids.add(source_id)
        negatives.append({
            'corpus_id': source_id,
            'source': 'retrieval',
            'score': round(float(getattr(item, 'score', 0.0) or 0.0), 6),
        })
        if len(negatives) >= max_negatives:
            break

    if len(negatives) < max_negatives:
        for score, row_id, source in _candidate_negative_rows(query_row, corpus_rows, corpus_by_id, positive_ids):
            if row_id in seen_negative_ids:
                continue
            seen_negative_ids.add(row_id)
            negatives.append({
                'corpus_id': row_id,
                'source': source,
                'score': round(float(score), 6),
            })
            if len(negatives) >= max_negatives:
                break
    return negatives


def build_retriever_dataset(
    output_name: str = 'retriever_unified_v1',
    include_history: bool = True,
    include_web: bool = False,
    max_negatives: int = 8,
    max_history_positives: int = 2,
    max_seed_variants: int = 3,
    val_ratio: float = 0.15,
) -> Dict[str, Any]:
    out_dir = RLHF_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = UnetiDocumentAgentV4Max(LLMConfig(enabled=False))
    corpus_rows, corpus_by_id = _build_corpus(agent, include_web=include_web)
    corpus_lookup = _build_corpus_lookup(corpus_rows)

    query_rows: List[Dict[str, Any]] = []
    stats = Counter()

    seed_queries, seed_stats = _build_seed_queries(corpus_rows, max_variants=max_seed_variants)
    query_rows.extend(seed_queries)
    stats.update(seed_stats)

    if include_history:
        history_queries, history_stats = _build_history_queries(
            corpus_by_id=corpus_by_id,
            corpus_lookup=corpus_lookup,
            max_positives=max_history_positives,
        )
        query_rows.extend(history_queries)
        stats.update(history_stats)

    clean_queries: List[Dict[str, Any]] = []
    for row in query_rows:
        positives = [pid for pid in _dedupe_preserve(row.get('positives', [])) if pid in corpus_by_id]
        if not positives:
            stats['skip_query_no_positive_in_corpus'] += 1
            continue
        row = dict(row)
        row['positives'] = _expand_positive_ids(
            query=str(row.get('query', '') or ''),
            positive_ids=positives,
            corpus_rows=corpus_rows,
            corpus_by_id=corpus_by_id,
        )
        row['quality'] = round(float(row.get('quality', 1.0) or 1.0), 4)
        row['query_type'] = str(row.get('query_type', '') or _query_type(str(row.get('query', '') or '')))
        row['split'] = _split_for_id(str(row.get('query_id', '') or ''), val_ratio=val_ratio)
        clean_queries.append(row)

    qrels: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []
    reranker_pairs: List[Dict[str, Any]] = []
    seen_pair_ids = set()

    for row in clean_queries:
        qid = str(row.get('query_id', '') or '')
        split = str(row.get('split', 'train') or 'train')
        query_text = str(row.get('query', '') or '')
        query_type = str(row.get('query_type', '') or 'generic')
        query_quality = float(row.get('quality', 1.0) or 1.0)
        positives = list(row.get('positives', []) or [])
        for positive_id in positives:
            qrels.append({
                'query_id': qid,
                'corpus_id': positive_id,
                'label': 1,
                'split': split,
            })

        negatives = _negative_ids_for_query(
            agent=agent,
            query_row=row,
            corpus_rows=corpus_rows,
            corpus_by_id=corpus_by_id,
            max_negatives=max_negatives,
        )
        row['hard_negatives'] = negatives
        if not negatives:
            stats['queries_without_negatives'] += 1
            continue

        for positive_id in positives:
            pair_pos_id = _hash_id(qid, positive_id, 'pos', prefix='pair')
            if pair_pos_id not in seen_pair_ids:
                seen_pair_ids.add(pair_pos_id)
                reranker_pairs.append({
                    'pair_id': pair_pos_id,
                    'query_id': qid,
                    'query': query_text,
                    'query_type': query_type,
                    'query_quality': query_quality,
                    'corpus_id': positive_id,
                    'text': str(corpus_by_id[positive_id].get('text', '') or ''),
                    'label': 1,
                    'origin': str(row.get('origin', '') or ''),
                    'domain': str(row.get('domain', '') or ''),
                    'split': split,
                })
            for negative in negatives:
                negative_id = str(negative.get('corpus_id', '') or '')
                pair_neg_id = _hash_id(qid, negative_id, 'neg', prefix='pair')
                if pair_neg_id not in seen_pair_ids:
                    seen_pair_ids.add(pair_neg_id)
                    reranker_pairs.append({
                        'pair_id': pair_neg_id,
                        'query_id': qid,
                        'query': query_text,
                        'query_type': query_type,
                        'query_quality': query_quality,
                        'corpus_id': negative_id,
                        'text': str(corpus_by_id[negative_id].get('text', '') or ''),
                        'label': 0,
                        'origin': str(row.get('origin', '') or ''),
                        'domain': str(row.get('domain', '') or ''),
                        'negative_source': str(negative.get('source', '') or ''),
                        'negative_score': float(negative.get('score', 0.0) or 0.0),
                        'split': split,
                    })
                triplets.append({
                    'triplet_id': _hash_id(qid, positive_id, negative_id, prefix='trip'),
                    'query_id': qid,
                    'query': query_text,
                    'query_type': query_type,
                    'query_quality': query_quality,
                    'positive_id': positive_id,
                    'positive': str(corpus_by_id[positive_id].get('text', '') or ''),
                    'negative_id': negative_id,
                    'negative': str(corpus_by_id[negative_id].get('text', '') or ''),
                    'origin': str(row.get('origin', '') or ''),
                    'domain': str(row.get('domain', '') or ''),
                    'negative_source': str(negative.get('source', '') or ''),
                    'negative_score': float(negative.get('score', 0.0) or 0.0),
                    'split': split,
                })

    corpus_rows.sort(key=lambda row: (str(row.get('source_kind', '') or ''), str(row.get('domain', '') or ''), str(row.get('id', '') or '')))
    clean_queries.sort(key=lambda row: str(row.get('query_id', '') or ''))
    qrels.sort(key=lambda row: (str(row.get('query_id', '') or ''), str(row.get('corpus_id', '') or '')))
    triplets.sort(key=lambda row: str(row.get('triplet_id', '') or ''))
    reranker_pairs.sort(key=lambda row: str(row.get('pair_id', '') or ''))

    _write_jsonl(out_dir / 'corpus.jsonl', corpus_rows)
    _write_jsonl(out_dir / 'queries.jsonl', clean_queries)
    _write_jsonl(out_dir / 'qrels.jsonl', qrels)
    _write_beir_qrels(out_dir / 'qrels.tsv', qrels)
    _write_jsonl(out_dir / 'embedding_triplets.jsonl', triplets)
    _write_jsonl(out_dir / 'reranker_pairs.jsonl', reranker_pairs)
    _write_jsonl(out_dir / 'queries_train.jsonl', [row for row in clean_queries if row.get('split') == 'train'])
    _write_jsonl(out_dir / 'queries_val.jsonl', [row for row in clean_queries if row.get('split') == 'val'])
    _write_jsonl(out_dir / 'embedding_triplets_train.jsonl', [row for row in triplets if row.get('split') == 'train'])
    _write_jsonl(out_dir / 'embedding_triplets_val.jsonl', [row for row in triplets if row.get('split') == 'val'])
    _write_jsonl(out_dir / 'reranker_pairs_train.jsonl', [row for row in reranker_pairs if row.get('split') == 'train'])
    _write_jsonl(out_dir / 'reranker_pairs_val.jsonl', [row for row in reranker_pairs if row.get('split') == 'val'])

    summary = {
        'output_dir': str(out_dir),
        'config': {
            'include_history': bool(include_history),
            'include_web': bool(include_web),
            'max_negatives': int(max_negatives),
            'max_history_positives': int(max_history_positives),
            'max_seed_variants': int(max_seed_variants),
            'val_ratio': float(val_ratio),
        },
        'counts': {
            'corpus': len(corpus_rows),
            'queries': len(clean_queries),
            'qrels': len(qrels),
            'embedding_triplets': len(triplets),
            'reranker_pairs': len(reranker_pairs),
            'train_queries': sum(1 for row in clean_queries if row.get('split') == 'train'),
            'val_queries': sum(1 for row in clean_queries if row.get('split') == 'val'),
        },
        'corpus_by_source_kind': dict(Counter(str(row.get('source_kind', '') or '') for row in corpus_rows)),
        'corpus_by_domain': dict(Counter(str(row.get('domain', '') or '') for row in corpus_rows)),
        'queries_by_origin': dict(Counter(str(row.get('origin', '') or '') for row in clean_queries)),
        'queries_by_domain': dict(Counter(str(row.get('domain', '') or '') for row in clean_queries)),
        'queries_by_type': dict(Counter(str(row.get('query_type', '') or '') for row in clean_queries)),
        'negative_sources': dict(Counter(str(item.get('negative_source', '') or '') for item in triplets)),
        'quality': {
            'avg_query_quality': round(sum(float(row.get('quality', 0.0) or 0.0) for row in clean_queries) / max(1, len(clean_queries)), 6),
            'min_query_quality': round(min((float(row.get('quality', 0.0) or 0.0) for row in clean_queries), default=0.0), 6),
            'max_query_quality': round(max((float(row.get('quality', 0.0) or 0.0) for row in clean_queries), default=0.0), 6),
        },
        'build_stats': dict(stats),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Build unified seed/doc retriever and reranker datasets.')
    parser.add_argument('--output-name', default='retriever_unified_v1')
    parser.add_argument('--skip-history', action='store_true', help='Do not mine query-positive pairs from chat histories.')
    parser.add_argument('--include-web', action='store_true', help='Include cached web documents in the corpus.')
    parser.add_argument('--max-negatives', type=int, default=8)
    parser.add_argument('--max-history-positives', type=int, default=2)
    parser.add_argument('--max-seed-variants', type=int, default=3)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    summary = build_retriever_dataset(
        output_name=str(args.output_name),
        include_history=not bool(args.skip_history),
        include_web=bool(args.include_web),
        max_negatives=max(1, int(args.max_negatives)),
        max_history_positives=max(1, int(args.max_history_positives)),
        max_seed_variants=max(0, int(args.max_seed_variants)),
        val_ratio=float(args.val_ratio),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
