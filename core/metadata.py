from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .normalize import norm_text_ascii


CONTACT_MARKERS = ['email', 'website', 'web:', 'http://', 'https://', 'so dien thoai', 'dien thoai:', 'sdt', 'lien he']
LOCATION_MARKERS = ['dia chi', 'co so', 'ha noi', 'nam dinh', 'minh khai', 'linh nam', 'tran hung dao', 'my xa']
ROLE_MARKERS = ['hieu truong', 'pho hieu truong', 'truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'chu tich']
HOWTO_MARKERS = ['dang nhap', 'doi mat khau', 'dang ky', 'tra cuu', 'xem ', 'chon ', 'nhan vao', 'vao muc', 'dashboard']
TEACHING_MARKERS = ['giang day', 'day hoc', 'hoc phan ve', 'cac hoc phan', 'mon hoc']
TRAINING_MARKERS = ['chuong trinh dao tao', 'nganh dao tao', 'dao tao nganh', 'ma nganh', 'tin chi', 'co hoi viec lam']

TOPIC_HINTS = {
    'admission': ['tuyen sinh', 'xet tuyen', 'dkxt', 'diem trung tuyen', 'diem san'],
    'announcement': ['thong bao', 'ke hoach', 'lich trinh', 'lich cong tac'],
    'scholarship': ['hoc bong'],
    'training': ['dao tao', 'hoc phan', 'tot nghiep', 'hoc ky', 'chuong trinh'],
    'student_affairs': ['sinh vien', 'cong tac sinh vien', 'ren luyen', 'hoc phi', 'cong no'],
    'portal': ['cong sinh vien', 'dang nhap', 'dashboard', 'dang ky hoc phan', 'lich hoc', 'lich thi'],
    'organization': ['co cau', 'phong ban', 'khoa chuyen mon', 'chuc nang', 'nhiem vu'],
    'faculty': ['khoa ', 'truong khoa', 'pho truong khoa'],
    'department': ['phong ', 'truong phong', 'pho truong phong'],
    'leadership': ['hieu truong', 'pho hieu truong', 'ban giam hieu', 'hoi dong truong'],
    'campus': ['co so', 'dia chi', 'minh khai', 'linh nam', 'tran hung dao', 'my xa'],
    'history': ['lich su', 'thanh lap', 'tien than', 'giai doan'],
    'contact': CONTACT_MARKERS,
    'location': LOCATION_MARKERS,
    'role': ROLE_MARKERS,
    'howto': HOWTO_MARKERS,
    'teaching': TEACHING_MARKERS,
    'training_program': TRAINING_MARKERS,
    'health': ['dich benh', 'virus', 'corona', 'covid', 'bo y te'],
    'quality_assurance': ['danh gia ngoai', 'kiem dinh', 'khao sat chinh thuc'],
    'research': ['nghien cuu khoa hoc', 'hoi thao', 'de tai'],
}

INTENT_TAGS_BY_CHUNK_TYPE = {
    'contact': ['contact', 'lookup'],
    'location': ['location', 'lookup'],
    'role': ['person_lookup', 'role_lookup'],
    'howto': ['howto', 'instruction'],
    'teaching': ['teaching_lookup', 'factoid'],
    'training': ['training_lookup', 'factoid'],
    'history': ['summary', 'timeline'],
    'function': ['function_lookup'],
    'duty': ['function_lookup'],
    'function_duty': ['function_lookup'],
    'overview': ['summary'],
    'fact': ['factoid'],
    'qa': ['qa'],
    'record': ['record_lookup'],
}

KEYWORD_STOPWORDS = {
    'ban', 'bao', 'cac', 'cho', 'co', 'cua', 'duoc', 'gia', 'hoc', 'hoi', 'khi', 'khong',
    'lam', 'mot', 'nay', 'neu', 'nhung', 'noi', 'phan', 'sinh', 'tai', 'the', 'thi',
    'tin', 'toi', 'tren', 'trong', 'truong', 'tu', 'uneti', 'va', 've', 'vien', 'voi',
}


def unique_strings(values: Iterable[Any], limit: int = 24) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or '').strip()
        key = norm_text_ascii(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _contains_any(text_ascii: str, markers: Iterable[str]) -> bool:
    return any(norm_text_ascii(marker) in text_ascii for marker in markers)


def extract_dates(text: str, limit: int = 12) -> List[str]:
    raw = str(text or '')
    values = []
    values.extend(re.findall(r'\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b', raw))
    values.extend(re.findall(r'\b(?:19|20)\d{2}\b', raw))
    return unique_strings(values, limit=limit)


def extract_emails(text: str, limit: int = 12) -> List[str]:
    return unique_strings(re.findall(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', str(text or '')), limit=limit)


def extract_urls(text: str, limit: int = 12) -> List[str]:
    return unique_strings(re.findall(r'https?://[^\s,;)]+' , str(text or '')), limit=limit)


def extract_search_terms(*parts: str, limit: int = 28) -> List[str]:
    tokens: List[str] = []
    for part in parts:
        for raw_tok in norm_text_ascii(part).split():
            tok = raw_tok.strip(".,:;!?()[]{}<>\"'`+-=*/\\|")
            if len(tok) < 3 or tok in KEYWORD_STOPWORDS or tok.isdigit():
                continue
            if sum(1 for ch in tok if ch.isalnum()) < max(2, len(tok) - 2):
                continue
            tokens.append(tok)
    return unique_strings(tokens, limit=limit)


def infer_topic_tags(
    *,
    doc_id: str,
    title: str,
    section: str,
    text: str,
    chunk_type: str,
    entity_type: str,
    entity_name: str,
    parsed_metadata: Dict[str, Any] | None = None,
    limit: int = 12,
) -> List[str]:
    metadata = parsed_metadata or {}
    haystack = norm_text_ascii(' '.join([
        str(doc_id or ''),
        str(title or ''),
        str(section or ''),
        str(text or ''),
        str(chunk_type or ''),
        str(entity_type or ''),
        str(entity_name or ''),
        str(metadata.get('category', '') or ''),
        str(metadata.get('channel', '') or ''),
    ]))
    tags: List[str] = []
    if str(doc_id or '').startswith('web-') or str(metadata.get('channel', '') or '').lower() == 'web':
        tags.append('web')
    for topic, hints in TOPIC_HINTS.items():
        if _contains_any(haystack, hints):
            tags.append(topic)
    if chunk_type in {'contact', 'location', 'role', 'howto', 'history'}:
        tags.append(chunk_type)
    if entity_type:
        tags.append(entity_type)
    return unique_strings(tags, limit=limit)


def infer_intent_tags(chunk_type: str, text: str, limit: int = 10) -> List[str]:
    q = norm_text_ascii(text)
    tags = list(INTENT_TAGS_BY_CHUNK_TYPE.get(str(chunk_type or ''), []))
    if _contains_any(q, CONTACT_MARKERS):
        tags.append('contact')
    if _contains_any(q, LOCATION_MARKERS):
        tags.append('location')
    if _contains_any(q, ROLE_MARKERS):
        tags.append('person_lookup')
    if _contains_any(q, HOWTO_MARKERS):
        tags.append('howto')
    return unique_strings(tags, limit=limit)


def build_chunk_metadata(
    *,
    doc_id: str,
    title: str,
    text: str,
    section: str,
    chunk_type: str,
    entity_type: str,
    entity_name: str,
    source_kind: str,
    parsed_metadata: Dict[str, Any] | None = None,
    source_path: str = '',
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    parsed_metadata = dict(parsed_metadata or {})
    extra = dict(extra or {})
    clean_text = ' '.join(str(text or '').split()).strip()
    text_ascii = norm_text_ascii(clean_text)
    source_url = (
        str(extra.get('source_url', '') or '').strip()
        or str(parsed_metadata.get('source_url', '') or parsed_metadata.get('url', '') or '').strip()
        or (str(source_path or '').strip() if str(source_path or '').lower().startswith('http') else '')
    )
    source_file = '' if source_url else str(source_path or '').strip()
    emails = extract_emails(clean_text)
    urls = unique_strings([*extract_urls(clean_text), source_url], limit=12)
    dates = extract_dates(clean_text)
    topic_tags = infer_topic_tags(
        doc_id=doc_id,
        title=title,
        section=section,
        text=clean_text,
        chunk_type=chunk_type,
        entity_type=entity_type,
        entity_name=entity_name,
        parsed_metadata=parsed_metadata,
    )
    intent_tags = infer_intent_tags(chunk_type, clean_text)
    search_terms = extract_search_terms(title, section, entity_name, chunk_type, ' '.join(topic_tags), clean_text)
    metadata = {
        'doc_id': doc_id,
        'doc_title': title,
        'section': section,
        'section_norm': norm_text_ascii(section),
        'chunk_type': chunk_type,
        'chunk_type_norm': norm_text_ascii(chunk_type),
        'entity_type': entity_type,
        'entity_name': entity_name,
        'entity_norm': norm_text_ascii(entity_name),
        'source_kind': source_kind,
        'source_url': source_url,
        'source_file': source_file,
        'topic_tags': topic_tags,
        'intent_tags': intent_tags,
        'search_terms': search_terms,
        'word_count': len(clean_text.split()),
        'char_count': len(clean_text),
        'line_count': max(1, len([line for line in str(text or '').splitlines() if line.strip()])),
        'sentence_count': max(1, len([s for s in re.split(r'(?<=[.!?;:])\s+', clean_text) if s.strip()])),
        'has_contact': _contains_any(text_ascii, CONTACT_MARKERS),
        'has_location': _contains_any(text_ascii, LOCATION_MARKERS),
        'has_role': _contains_any(text_ascii, ROLE_MARKERS),
        'has_howto': _contains_any(text_ascii, HOWTO_MARKERS),
        'has_email': bool(emails),
        'has_url': bool(urls),
        'emails': emails,
        'urls': urls,
        'dates': dates,
        'years': [value for value in dates if re.fullmatch(r'(?:19|20)\d{2}', value)],
    }
    for key in ['category', 'channel', 'published_at', 'authoritative', 'source']:
        if key in parsed_metadata and key not in metadata:
            metadata[key] = parsed_metadata[key]
    metadata.update(extra)
    return metadata


def metadata_search_text(metadata: Dict[str, Any] | None) -> str:
    meta = metadata or {}
    values: List[str] = []
    for key in [
        'doc_title', 'section', 'chunk_type', 'entity_type', 'entity_name', 'category',
        'source_url', 'source_file', 'published_at',
    ]:
        if meta.get(key):
            values.append(str(meta.get(key)))
    for key in ['topic_tags', 'intent_tags', 'search_terms', 'keywords', 'emails', 'urls', 'dates', 'years']:
        value = meta.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if str(item).strip())
        elif value:
            values.append(str(value))
    return ' '.join(values)
