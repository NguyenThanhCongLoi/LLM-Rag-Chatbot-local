import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from .metadata import build_chunk_metadata, unique_strings
from .config import DATA_DIR
from .normalize import norm_text_ascii


RUNTIME_STATE_PATH = DATA_DIR / 'runtime_state_perfect_v1.json'
DOMAIN_DEFAULTS = {
    'phong_ban_va_chuc_nang': {
        'title': 'Phòng ban và chức năng',
        'category': 'unit_info',
        'routing_triggers': ['phong', 'ctsv', 'hanh chinh mot cua', 'phong dao tao'],
        'qa': [],
        'chunks': [],
    },
    'khoa_chuyen_mon': {
        'title': 'Khoa chuyên môn',
        'category': 'faculty_info',
        'routing_triggers': ['khoa', 'truong khoa', 'co khi', 'thuong mai'],
        'qa': [],
        'chunks': [],
    },
}
FILLER_PREFIXES = [
    'giup minh', 'giup em', 'giup toi', 'cho em hoi', 'cho minh hoi', 'ad oi', 'admin oi',
    'uneti oi', 'ban oi', 'anh chi oi', 'cho hoi',
]
FILLER_SUFFIXES = [' nhe', ' nha', ' a', ' ah', ' voi', ' dum', ' dum em']
RAG_CATEGORY_TO_DOMAIN = {
    'history': 'lich_su_hinh_thanh',
    'leadership': 'ban_giam_hieu',
    'campus': 'co_so_vat_chat',
    'student_portal': 'portal_howto',
    'training_office': 'phong_ban_va_chuc_nang',
    'ctsv': 'phong_ban_va_chuc_nang',
    'one_stop': 'phong_ban_va_chuc_nang',
    'student_support': 'phong_ban_va_chuc_nang',
    'faculty_mechanical': 'khoa_chuyen_mon',
    'faculty_commerce': 'khoa_chuyen_mon',
    'faculty_tourism': 'khoa_chuyen_mon',
    'faculty_foodtech': 'khoa_chuyen_mon',
    'faculty_it': 'khoa_chuyen_mon',
    'faculty_electrical': 'khoa_chuyen_mon',
    'faculty_electronics': 'khoa_chuyen_mon',
    'faculty_textile': 'khoa_chuyen_mon',
    'faculty_qtm': 'khoa_chuyen_mon',
    'faculty_pe': 'khoa_chuyen_mon',
    'faculty_accounting': 'khoa_chuyen_mon',
    'faculty_finance': 'khoa_chuyen_mon',
    'faculty_applied_science': 'khoa_chuyen_mon',
    'faculty_languages': 'khoa_chuyen_mon',
    'faculty_political_theory': 'khoa_chuyen_mon',
}
PRIORITY_BY_CHUNK_TYPE = {
    'overview': 100,
    'role': 98,
    'contact': 96,
    'location': 95,
    'howto': 94,
    'history': 90,
    'fact': 88,
    'structure': 84,
    'function': 84,
    'duty': 83,
    'qa': 92,
    'record': 92,
    'generic': 70,
}
CONTACT_MARKERS = ['email', 'website', 'http://', 'https://', 'so dien thoai', 'dien thoai:', 'sdt', 'lien he']
LOCATION_MARKERS = ['dia chi', 'co so', 'ha noi', 'nam dinh', 'minh khai', 'linh nam', 'tran hung dao', 'my xa']
ROLE_MARKERS = ['hieu truong', 'pho hieu truong', 'truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'chu tich']
HOWTO_MARKERS = ['dang nhap', 'doi mat khau', 'dang ky', 'tra cuu', 'xem ', 'vao ', 'chon ', 'dashboard']


def _dedupe_strings(values: List[str]) -> List[str]:
    return unique_strings(values, limit=999)


def _sanitize_variant_text(text: str) -> str:
    q = norm_text_ascii(text)
    q = re.sub(r'\s+', ' ', q).strip()
    for prefix in FILLER_PREFIXES:
        if q.startswith(prefix + ' '):
            q = q[len(prefix):].strip()
    for suffix in FILLER_SUFFIXES:
        if q.endswith(suffix):
            q = q[: -len(suffix)].strip()
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def _runtime_domain(answer: Dict[str, Any]) -> str:
    group = str(answer.get('intent_group', '') or '')
    topic = norm_text_ascii(answer.get('topic', ''))
    if group == 'portal_howto':
        return 'portal_howto'
    if group == 'unit_info':
        return 'phong_ban_va_chuc_nang'
    if group == 'faculty_info':
        return 'khoa_chuyen_mon'
    if group == 'school_info':
        if 'ban giam hieu' in topic:
            return 'ban_giam_hieu'
        if 'co so' in topic:
            return 'co_so_vat_chat'
        return 'lich_su_hinh_thanh'
    return ''


def _collect_runtime_variants(runtime: Dict[str, Any]) -> Dict[str, List[str]]:
    by_answer: Dict[str, List[str]] = {}
    for item in runtime.get('faq_variants', []) or []:
        answer_id = str(item.get('answer_id', '') or '').strip()
        if not answer_id:
            continue
        variant = _sanitize_variant_text(str(item.get('variant_text', '') or ''))
        if len(variant) < 4:
            continue
        by_answer.setdefault(answer_id, []).append(variant)
    return {k: _dedupe_strings(v)[:12] for k, v in by_answer.items()}


def _merge_runtime_qas(seed: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(seed)
    merged.setdefault('domains', {})
    for domain, body in DOMAIN_DEFAULTS.items():
        merged['domains'].setdefault(domain, deepcopy(body))
    variants_by_answer = _collect_runtime_variants(runtime)

    existing_questions: Dict[str, Dict[str, int]] = {}
    for domain, body in merged.get('domains', {}).items():
        body.setdefault('qa', [])
        existing_questions[domain] = {
            norm_text_ascii(item.get('question', '')): idx
            for idx, item in enumerate(body.get('qa', []))
            if norm_text_ascii(item.get('question', ''))
        }

    for item in runtime.get('faq_answers', []) or []:
        domain = _runtime_domain(item)
        if not domain:
            continue
        question = str(item.get('question', '') or '').strip()
        answer = str(item.get('answer', '') or '').strip()
        if not question or not answer:
            continue
        q_norm = norm_text_ascii(question)

        answer_id = str(item.get('answer_id', '') or '').strip()
        keywords = [str(x).strip() for x in item.get('keywords', []) if str(x).strip()]
        keywords.extend(variants_by_answer.get(answer_id, []))
        qa = {
            'question': question,
            'answer': answer,
            'keywords': _dedupe_strings(keywords)[:24],
        }
        q_index = existing_questions.setdefault(domain, {}).get(q_norm)
        if q_index is not None:
            existing = merged['domains'][domain]['qa'][q_index]
            merged_keywords = _dedupe_strings([
                *existing.get('keywords', []),
                *qa.get('keywords', []),
            ])[:24]
            merged['domains'][domain]['qa'][q_index] = {
                'question': question,
                'answer': answer,
                'keywords': merged_keywords,
            }
        else:
            merged['domains'][domain].setdefault('qa', []).append(qa)
            existing_questions[domain][q_norm] = len(merged['domains'][domain]['qa']) - 1
    return merged


def _merge_runtime_rag(seed: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(seed)
    merged.setdefault('domains', {})
    for domain, body in DOMAIN_DEFAULTS.items():
        merged['domains'].setdefault(domain, deepcopy(body))

    existing_facts: Dict[str, set] = {}
    for domain, body in merged.get('domains', {}).items():
        body.setdefault('facts', [])
        existing_facts[domain] = {norm_text_ascii(text) for text in body.get('facts', [])}

    for item in runtime.get('rag_records', []) or []:
        domain = RAG_CATEGORY_TO_DOMAIN.get(str(item.get('category', '') or '').strip())
        if not domain:
            continue
        chunk_text = str(item.get('chunk_text', '') or '').strip()
        if not chunk_text:
            continue
        text_norm = norm_text_ascii(chunk_text)
        if not text_norm or text_norm in existing_facts.setdefault(domain, set()):
            continue
        merged['domains'][domain].setdefault('facts', []).append(chunk_text)
        existing_facts[domain].add(text_norm)
    return merged


def _runtime_qa_meta(runtime: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    variants_by_answer = _collect_runtime_variants(runtime)
    out: Dict[str, Dict[str, Any]] = {}
    for item in runtime.get('faq_answers', []) or []:
        question = str(item.get('question', '') or '').strip()
        q_norm = norm_text_ascii(question)
        if not q_norm:
            continue
        answer_id = str(item.get('answer_id', '') or '').strip()
        keywords = [str(x).strip() for x in item.get('keywords', []) if str(x).strip()]
        keywords.extend(variants_by_answer.get(answer_id, []))
        out[q_norm] = {
            'answer_id': answer_id,
            'topic': str(item.get('topic', '') or '').strip(),
            'unit_name': str(item.get('unit_name', '') or '').strip(),
            'intent_name': str(item.get('intent_name', '') or '').strip(),
            'intent_group': str(item.get('intent_group', '') or '').strip(),
            'action_type': str(item.get('action_type', '') or '').strip(),
            'source_url': str(item.get('source_url', '') or '').strip(),
            'keywords': _dedupe_strings(keywords)[:24],
        }
    return out


def _runtime_fact_meta(runtime: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in runtime.get('rag_records', []) or []:
        text = str(item.get('chunk_text', '') or '').strip()
        text_norm = norm_text_ascii(text)
        if not text_norm:
            continue
        out[text_norm] = {
            'runtime_doc_id': str(item.get('doc_id', '') or '').strip(),
            'runtime_chunk_id': str(item.get('chunk_id', '') or '').strip(),
            'runtime_title': str(item.get('title', '') or '').strip(),
            'category': str(item.get('category', '') or '').strip(),
            'authoritative': str(item.get('authoritative', '') or '').strip().lower() in {'1', 'true', 'yes', 'y'},
            'doc_text': str(item.get('doc_text', '') or '').strip(),
        }
    return out


def _stable_hash(*parts: str) -> str:
    sha = hashlib.sha1()
    for part in parts:
        sha.update(str(part or '').encode('utf-8', errors='ignore'))
        sha.update(b'\0')
    return sha.hexdigest()[:12]


def _safe_slug(text: str, fallback: str = 'root') -> str:
    q = re.sub(r'[^a-z0-9]+', '-', norm_text_ascii(text or '')).strip('-')
    return q[:48] or fallback


def _extract_summary(text: str, max_words: int = 28) -> str:
    clean = ' '.join(str(text or '').split()).strip()
    if not clean:
        return ''
    words = clean.split()
    if len(words) <= max_words:
        return clean
    return ' '.join(words[:max_words]).rstrip(',;:. ') + '...'


def _chunk_id(domain: str, chunk_type: str, entity_name: str, text: str) -> str:
    label = _safe_slug(entity_name or chunk_type)
    return f"seed-{domain}--{chunk_type}--{label}--{_stable_hash(domain, chunk_type, entity_name, norm_text_ascii(text))}"


def _dedupe_key(chunk_type: str, entity_name: str, text: str) -> str:
    return _stable_hash(chunk_type, entity_name, norm_text_ascii(text))


def _domain_entity_type(domain: str) -> str:
    if domain == 'ban_giam_hieu':
        return 'person'
    if domain == 'co_so_vat_chat':
        return 'campus'
    if domain == 'portal_howto':
        return 'portal'
    if domain == 'khoa_chuyen_mon':
        return 'faculty'
    if domain == 'phong_ban_va_chuc_nang':
        return 'unit'
    if domain == 'hoi_dong_truong':
        return 'council'
    if domain == 'lich_su_hinh_thanh':
        return 'school'
    return 'document'


def _fact_chunk_type(domain: str, text: str) -> str:
    q = norm_text_ascii(text)
    if any(marker in q for marker in CONTACT_MARKERS):
        return 'contact'
    if any(marker in q for marker in ROLE_MARKERS):
        return 'role'
    if domain == 'co_so_vat_chat' and any(marker in q for marker in LOCATION_MARKERS):
        return 'location'
    if domain == 'portal_howto' and any(marker in q for marker in HOWTO_MARKERS):
        return 'howto'
    if domain == 'lich_su_hinh_thanh' and any(marker in q for marker in ['thanh lap', 'tien than', 'lich su', 'giai doan']):
        return 'history'
    if domain in {'phong_ban_va_chuc_nang', 'khoa_chuyen_mon'}:
        if 'chuc nang' in q:
            return 'function'
        if 'nhiem vu' in q:
            return 'duty'
    if domain in {'ban_giam_hieu', 'hoi_dong_truong'}:
        return 'role'
    if domain == 'co_so_vat_chat':
        return 'overview'
    return 'fact'


def _qa_chunk_type(domain: str, question: str, answer: str, action_type: str) -> str:
    q = norm_text_ascii(f'{question} {answer}')
    if any(marker in q for marker in ROLE_MARKERS):
        return 'role'
    if domain == 'co_so_vat_chat' and any(marker in q for marker in LOCATION_MARKERS):
        return 'location'
    if action_type == 'contact':
        return 'contact'
    if domain == 'portal_howto' or action_type in {'howto', 'guide'} or any(marker in q for marker in HOWTO_MARKERS):
        return 'howto'
    if any(marker in q for marker in CONTACT_MARKERS):
        return 'contact'
    if domain == 'lich_su_hinh_thanh' and any(marker in q for marker in ['thanh lap', 'tien than', 'lich su', 'giai doan']):
        return 'history'
    return 'overview'


def _priority(chunk_type: str) -> int:
    return PRIORITY_BY_CHUNK_TYPE.get(chunk_type, PRIORITY_BY_CHUNK_TYPE['generic'])


def _make_seed_chunk(
    *,
    domain: str,
    domain_title: str,
    kind: str,
    text: str,
    title: str,
    chunk_type: str,
    entity_name: str,
    section: str = '',
    keywords: List[str] | None = None,
    source_url: str = '',
    is_authoritative: bool = True,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    clean_text = ' '.join(str(text or '').split()).strip()
    entity_name = str(entity_name or domain_title).strip() or domain_title
    chunk_keywords = _dedupe_strings([str(item).strip() for item in (keywords or []) if str(item).strip()])[:24]
    raw_metadata = dict(metadata or {})
    enriched_metadata = build_chunk_metadata(
        doc_id=f'seed-{domain}',
        title=title,
        text=clean_text,
        section=section,
        chunk_type=chunk_type,
        entity_type=_domain_entity_type(domain),
        entity_name=entity_name,
        source_kind='seed',
        parsed_metadata={
            'category': str(raw_metadata.get('category', '') or ''),
            'source_url': source_url,
            'authoritative': bool(is_authoritative),
        },
        source_path=source_url,
        extra=raw_metadata | {
            'domain': domain,
            'domain_title': domain_title,
            'keywords': chunk_keywords,
            'keyword_ids': [f"kw-{_safe_slug(item)}-{_stable_hash(item)}" for item in chunk_keywords],
            'source_url': source_url,
        },
    )
    return {
        'kind': kind,
        'chunk_id': _chunk_id(domain, chunk_type, entity_name, clean_text),
        'doc_id': f'seed-{domain}',
        'title': title,
        'text': clean_text,
        'section': section,
        'chunk_type': chunk_type,
        'entity_type': _domain_entity_type(domain),
        'entity_name': entity_name,
        'priority': _priority(chunk_type if kind != 'qa' else 'qa'),
        'summary': _extract_summary(clean_text),
        'dedupe_key': _dedupe_key(chunk_type, entity_name, clean_text),
        'parent_chunk_id': '',
        'source_kind': 'seed',
        'is_authoritative': bool(is_authoritative),
        'source_url': source_url,
        'keywords': chunk_keywords,
        'keyword_ids': [f"kw-{_safe_slug(item)}-{_stable_hash(item)}" for item in chunk_keywords],
        'metadata': enriched_metadata,
    }


def _append_domain_chunk(body: Dict[str, Any], chunk: Dict[str, Any], seen: set[str]) -> None:
    key = str(chunk.get('dedupe_key', '') or '')
    if not key or key in seen:
        return
    seen.add(key)
    body.setdefault('chunks', []).append(chunk)


def _build_seed_chunks(data: Dict[str, Any], runtime: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(data)
    qa_meta = _runtime_qa_meta(runtime)
    fact_meta = _runtime_fact_meta(runtime)

    for domain, body in merged.get('domains', {}).items():
        body.setdefault('records', [])
        body.setdefault('facts', [])
        body.setdefault('qa', [])
        body.setdefault('locations', [])
        body['chunks'] = []
        seen: set[str] = set()
        domain_title = str(body.get('title', domain) or domain)

        for idx, rec in enumerate(body.get('records', [])):
            text = ' | '.join(str(v) for v in rec.values() if str(v).strip())
            entity_name = str(rec.get('name', '') or rec.get('role', '') or domain_title).strip()
            chunk_type = 'role' if rec.get('role') else 'record'
            chunk = _make_seed_chunk(
                domain=domain,
                domain_title=domain_title,
                kind='record',
                text=text,
                title=domain_title,
                chunk_type=chunk_type,
                entity_name=entity_name,
                keywords=[entity_name, str(rec.get('role', '') or '').strip()],
                metadata=rec | {'record_index': idx},
            )
            _append_domain_chunk(body, chunk, seen)

        for idx, location in enumerate(body.get('locations', [])):
            text = ' | '.join(str(v) for v in location.values() if str(v).strip())
            entity_name = str(location.get('city', '') or location.get('address', '') or domain_title).strip()
            chunk = _make_seed_chunk(
                domain=domain,
                domain_title=domain_title,
                kind='location',
                text=text,
                title=domain_title,
                chunk_type='location',
                entity_name=entity_name,
                keywords=[str(location.get('city', '') or '').strip(), str(location.get('address', '') or '').strip()],
                metadata=location | {'location_index': idx},
            )
            _append_domain_chunk(body, chunk, seen)

        for idx, fact in enumerate(body.get('facts', [])):
            text = str(fact or '').strip()
            text_norm = norm_text_ascii(text)
            meta = fact_meta.get(text_norm, {})
            chunk = _make_seed_chunk(
                domain=domain,
                domain_title=domain_title,
                kind='fact',
                text=text,
                title=domain_title,
                chunk_type=_fact_chunk_type(domain, text),
                entity_name=str(meta.get('runtime_title', '') or domain_title).strip(),
                source_url='',
                is_authoritative=bool(meta.get('authoritative', True)),
                keywords=[str(meta.get('runtime_title', '') or '').strip()],
                metadata=meta | {'fact_index': idx},
            )
            _append_domain_chunk(body, chunk, seen)

        for idx, qa in enumerate(body.get('qa', [])):
            question = str(qa.get('question', '') or '').strip()
            answer = str(qa.get('answer', '') or '').strip()
            q_norm = norm_text_ascii(question)
            meta = qa_meta.get(q_norm, {})
            keywords = _dedupe_strings([
                *[str(item).strip() for item in qa.get('keywords', []) if str(item).strip()],
                *[str(item).strip() for item in meta.get('keywords', []) if str(item).strip()],
            ])[:24]
            entity_name = str(meta.get('unit_name', '') or meta.get('topic', '') or domain_title).strip() or domain_title
            chunk = _make_seed_chunk(
                domain=domain,
                domain_title=domain_title,
                kind='qa',
                text=answer,
                title=question or domain_title,
                chunk_type=_qa_chunk_type(domain, question, answer, str(meta.get('action_type', '') or '')),
                entity_name=entity_name,
                section=str(meta.get('intent_name', '') or ''),
                source_url=str(meta.get('source_url', '') or ''),
                is_authoritative=True,
                keywords=keywords,
                metadata={
                    'question': question,
                    'answer': answer,
                    'keywords': keywords,
                    'answer_id': str(meta.get('answer_id', '') or ''),
                    'topic': str(meta.get('topic', '') or ''),
                    'unit_name': str(meta.get('unit_name', '') or ''),
                    'intent_name': str(meta.get('intent_name', '') or ''),
                    'intent_group': str(meta.get('intent_group', '') or ''),
                    'action_type': str(meta.get('action_type', '') or ''),
                    'qa_index': idx,
                },
            )
            _append_domain_chunk(body, chunk, seen)

        body['chunks'].sort(
            key=lambda item: (
                int(item.get('priority', 0) or 0),
                str(item.get('chunk_type', '') or ''),
                str(item.get('entity_name', '') or ''),
                str(item.get('chunk_id', '') or ''),
            ),
            reverse=True,
        )

    merged['seed_chunk_schema'] = 'v2'
    merged['seed_chunk_count'] = sum(len(body.get('chunks', [])) for body in merged.get('domains', {}).values())
    return merged


def load_seed_knowledge() -> dict:
    base_path = DATA_DIR / 'knowledge_seed.json'
    data = json.loads(base_path.read_text(encoding='utf-8'))
    runtime: Dict[str, Any] = {}
    if RUNTIME_STATE_PATH.exists():
        runtime = json.loads(RUNTIME_STATE_PATH.read_text(encoding='utf-8'))
        data = _merge_runtime_qas(data, runtime)
        data = _merge_runtime_rag(data, runtime)
    data = _build_seed_chunks(data, runtime)
    return data


def load_seed_chunks() -> List[Dict[str, Any]]:
    data = load_seed_knowledge()
    chunks: List[Dict[str, Any]] = []
    for body in data.get('domains', {}).values():
        chunks.extend(body.get('chunks', []) or [])
    return chunks
