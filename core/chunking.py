from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from .metadata import build_chunk_metadata
from .models import Chunk
from .normalize import norm_text_ascii, slugify


SECTION_HINTS = {
    'overview': ['tong quan', 'gioi thieu', 'thong tin chung', 'overview'],
    'history': ['lich su', 'qua trinh hinh thanh', 'hinh thanh', 'giai doan'],
    'achievement': ['thanh tich', 'ket qua dat duoc'],
    'structure': ['co cau to chuc', 'co cau'],
    'function_duty': ['chuc nang, nhiem vu'],
    'function': ['chuc nang'],
    'duty': ['nhiem vu'],
    'teaching': ['giang day', 'day hoc', 'hoc phan ve', 'cac hoc phan'],
    'training': ['chuong trinh dao tao', 'nganh dao tao', 'dao tao nganh', 'vi tri viec lam', 'co hoi viec lam'],
    'capacity': ['quy mo', 'nang luc'],
    'development': ['dinh huong'],
    'contact': ['dia chi lien he', 'thong tin lien he', 'lien he'],
}

TITLE_ONLY_HEADINGS = {
    'lich-su-hinh-thanh',
    'khoa-chuyen-mon',
    'phong-ban-va-chuc-nang',
    'co-so-vat-chat',
}

ROLE_PATTERNS = [
    r'\bhieu truong\b',
    r'\bpho hieu truong\b',
    r'\btruong khoa\b',
    r'\bpho truong khoa\b',
    r'\btruong phong\b',
    r'\bpho truong phong\b',
    r'\btro ly khoa\b',
    r'\bchu tich hoi dong truong\b',
]

CONTACT_MARKERS = ['email', 'website', 'web:', 'http://', 'https://', 'so dien thoai', 'dien thoai:', 'sdt', 'lien he']
LOCATION_MARKERS = ['dia chi', 'co so', 'minh khai', 'linh nam', 'tran hung dao', 'my xa', 'ha noi', 'nam dinh']
HOWTO_MARKERS = ['dang nhap', 'doi mat khau', 'dang ky', 'tra cuu', 'xem ', 'chon ', 'nhan vao', 'vao muc', 'dashboard']
TEACHING_MARKERS = ['giang day', 'day hoc', 'hoc phan ve', 'cac hoc phan', 'mon hoc']
TRAINING_MARKERS = ['chuong trinh dao tao', 'nganh dao tao', 'dao tao nganh', 'ma nganh', 'tin chi', 'co hoi viec lam']


CHUNK_PRIORITY = {
    'overview': 100,
    'role': 98,
    'contact': 96,
    'location': 95,
    'howto': 94,
    'teaching': 93,
    'training': 89,
    'history': 90,
    'fact': 88,
    'structure': 84,
    'function': 84,
    'duty': 83,
    'function_duty': 83,
    'achievement': 80,
    'capacity': 78,
    'development': 77,
    'generic': 70,
    'summary': 62,
    'section_header': 40,
}


def _safe_slug(text: str, fallback: str = 'root') -> str:
    value = slugify(text or '')[:48]
    return value or fallback


def _stable_hash(*parts: str) -> str:
    sha = hashlib.sha1()
    for part in parts:
        sha.update(str(part or '').encode('utf-8', errors='ignore'))
        sha.update(b'\0')
    return sha.hexdigest()[:12]


def _dedupe_key(chunk_type: str, entity_type: str, entity_name: str, text: str) -> str:
    norm_text = norm_text_ascii(text)
    return _stable_hash(chunk_type, entity_type, entity_name, norm_text)


def _chunk_id(doc_id: str, chunk_type: str, entity_name: str, section: str, text: str) -> str:
    label = _safe_slug(entity_name or section or chunk_type)
    return f"{doc_id}--{chunk_type}--{label}--{_stable_hash(doc_id, chunk_type, entity_name, section, norm_text_ascii(text))}"


def _extract_summary(text: str, max_words: int = 28) -> str:
    clean = ' '.join(str(text or '').split()).strip()
    if not clean:
        return ''
    sentences = re.split(r'(?<=[.!?;:])\s+', clean)
    first = sentences[0].strip() if sentences else clean
    words = first.split()
    if len(words) <= max_words:
        return first
    return ' '.join(words[:max_words]).rstrip(',;:. ') + '...'


def _priority(chunk_type: str) -> int:
    return CHUNK_PRIORITY.get(chunk_type, CHUNK_PRIORITY['generic'])


def _source_kind(parsed: Dict[str, Any]) -> str:
    metadata = parsed.get('metadata') or {}
    if str(metadata.get('channel', '') or '').strip().lower() == 'web':
        return 'web'
    source_path = str(parsed.get('source_path', '') or '')
    if source_path.lower().startswith('http'):
        return 'web'
    return 'local'


def _is_authoritative(parsed: Dict[str, Any]) -> bool:
    metadata = parsed.get('metadata') or {}
    flag = metadata.get('authoritative')
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str) and flag.strip():
        return flag.strip().lower() in {'1', 'true', 'yes', 'y'}
    return True


def _entity_type_for_doc(doc_id: str) -> str:
    if doc_id == 'ban-giam-hieu':
        return 'person'
    if doc_id == 'co-so-vat-chat':
        return 'campus'
    if doc_id == 'danh-sach-cac-thanh-vien-hoi-dong-truong':
        return 'council'
    if doc_id == 'phong-ban-va-chuc-nang':
        return 'unit'
    if doc_id == 'khoa-chuyen-mon':
        return 'faculty'
    if doc_id == 'huong-dan-chuc-nang-cong-thong-tin-sv':
        return 'portal'
    if doc_id.startswith('web-'):
        return 'web_article'
    return 'document'


def _normalize_blocks(parsed: Dict[str, Any]) -> List[Tuple[str, Optional[int]]]:
    blocks = parsed.get('blocks') or []
    out: List[Tuple[str, Optional[int]]] = []
    title_norm = norm_text_ascii(parsed.get('title', ''))
    for block in blocks:
        if isinstance(block, dict):
            text = str(block.get('text', '') or '').strip()
            page = block.get('page')
        else:
            text = str(block or '').strip()
            page = None
        q = norm_text_ascii(text)
        if text and not q.startswith('posted on ') and q != title_norm:
            out.append((text, page))
    return out


def _looks_like_entity_heading(text: str, doc_id: str) -> bool:
    q = norm_text_ascii(text)
    if doc_id == 'khoa-chuyen-mon':
        if q.startswith('khoa hoc ') and not q.startswith('khoa khoa hoc '):
            return False
        return q.startswith('khoa ') and 'khoa chuyen mon' not in q and len(q.split()) <= 8
    if doc_id == 'phong-ban-va-chuc-nang':
        return q.startswith('phong ') and 'phong ban va chuc nang' not in q and len(q.split()) <= 10
    return False


def _looks_like_heading(text: str, doc_id: str) -> bool:
    q = norm_text_ascii(text)
    if not q:
        return False
    if doc_id in TITLE_ONLY_HEADINGS and q in {norm_text_ascii(parsed) for parsed in ['Lịch sử hình thành', 'Khoa chuyên môn', 'Phòng ban và chức năng', 'Cơ sở vật chất']}:
        return True
    if _looks_like_entity_heading(text, doc_id):
        return True
    if re.match(r'^\d+(?:\.\d+)?\s*[.)]?\s+', q):
        return True
    if q.startswith('giai doan '):
        return True
    if len(q.split()) <= 10 and any(hint in q for hints in SECTION_HINTS.values() for hint in hints):
        return True
    if text.isupper() and 2 <= len(q.split()) <= 12:
        return True
    return False


def _section_type(section: str, text: str, doc_id: str) -> str:
    sec = norm_text_ascii(section)
    body = norm_text_ascii(text)
    if doc_id in {'lich-su-hinh-thanh', 'web-lich-su-hinh-thanh'} and sec not in {'', 'overview'}:
        if not any(marker in body for marker in CONTACT_MARKERS):
            return 'history'
    if doc_id == 'huong-dan-chuc-nang-cong-thong-tin-sv':
        if any(marker in body for marker in HOWTO_MARKERS):
            return 'howto'
        return 'overview'
    if any(hint in sec for hint in SECTION_HINTS['contact']):
        return 'contact'
    if any(hint in sec for hint in SECTION_HINTS['structure']):
        return 'structure'
    if any(hint in sec for hint in SECTION_HINTS['function_duty']):
        return 'function_duty'
    if any(hint in sec for hint in SECTION_HINTS['function']):
        return 'function'
    if any(hint in sec for hint in SECTION_HINTS['duty']):
        return 'duty'
    if any(hint in sec for hint in SECTION_HINTS['teaching']):
        return 'teaching'
    if any(hint in sec for hint in SECTION_HINTS['training']):
        return 'training'
    if any(hint in sec for hint in SECTION_HINTS['history']):
        return 'history'
    if any(hint in sec for hint in SECTION_HINTS['achievement']):
        return 'achievement'
    if any(hint in sec for hint in SECTION_HINTS['capacity']):
        return 'capacity'
    if any(hint in sec for hint in SECTION_HINTS['development']):
        return 'development'
    if re.search(r'\bgiai doan\s+\d+\b', sec):
        return 'history'
    if any(marker in body for marker in TRAINING_MARKERS):
        return 'training'
    if any(marker in body for marker in TEACHING_MARKERS):
        return 'teaching'
    if doc_id not in {'khoa-chuyen-mon', 'phong-ban-va-chuc-nang'} and any(marker in body for marker in HOWTO_MARKERS):
        return 'howto'
    if _text_has_contact_marker(text):
        return 'contact'
    return 'overview' if sec in {'overview', ''} else 'generic'


def _line_is_contact(text: str) -> bool:
    return _text_has_contact_marker(text)


def _text_has_contact_marker(text: str) -> bool:
    q = norm_text_ascii(text)
    if any(marker in q for marker in ['email', 'website', 'web:', 'http://', 'https://', 'sdt', 'lien he']):
        return True
    return bool(re.search(r'\b(?:so\s+)?dien\s+thoai\s*:', q))


def _line_is_teaching(text: str) -> bool:
    q = norm_text_ascii(text)
    if any(marker in q for marker in ['hoc phan ve', 'cac hoc phan']):
        return True
    if 'giang day' in q and 'nghien cuu' not in q and 'giang day bang' not in q:
        return True
    return False


def _line_is_training(text: str) -> bool:
    q = norm_text_ascii(text)
    return any(marker in q for marker in TRAINING_MARKERS)


def _split_sentences(text: str) -> List[str]:
    clean = ' '.join(str(text or '').replace('\r', '\n').split())
    return [piece.strip() for piece in re.split(r'(?<=[.!?;])\s+', clean) if piece.strip()]


def _line_is_location(text: str) -> bool:
    q = norm_text_ascii(text)
    if 'email' in q or 'dien thoai' in q:
        return False
    if re.search(r'\bso\s+\d{1,4}\b', q) and any(marker in q for marker in ['dia chi', 'ha noi', 'nam dinh', 'minh khai', 'linh nam', 'tran hung dao', 'my xa']):
        return True
    address_hits = sum(1 for marker in ['minh khai', 'linh nam', 'tran hung dao', 'my xa', 'ha noi', 'nam dinh', 'dia chi'] if marker in q)
    return address_hits >= 2 or ('co so' in q and ('ha noi' in q or 'nam dinh' in q))


def _line_is_role(text: str) -> bool:
    q = norm_text_ascii(text)
    if len(q.split()) > 28:
        return False
    if ':' not in text and ' la ' not in q and not re.match(r'^(ts|ths|pgs|pgs.ts|ts\.)\b', q):
        return False
    return any(re.search(pattern, q) for pattern in ROLE_PATTERNS)


def _split_bullets(text: str) -> List[str]:
    cleaned = str(text or '').replace('\r', '\n')
    pieces: List[str] = []
    for part in re.split(r'\n+', cleaned):
        line = part.strip(' -+•–\t')
        if not line:
            continue
        if len(line.split()) >= 4:
            pieces.append(line)
    return pieces


def _text_windows(text: str, window_words: int = 170, overlap: int = 35) -> List[str]:
    words = text.split()
    if len(words) <= 220:
        return [text.strip()]
    windows: List[str] = []
    start = 0
    while start < len(words):
        piece = ' '.join(words[start:start + window_words]).strip()
        if piece:
            windows.append(piece)
        if start + window_words >= len(words):
            break
        start += max(1, window_words - overlap)
    return windows


def _make_chunk(
    *,
    parsed: Dict[str, Any],
    text: str,
    section: str,
    page: Optional[int],
    chunk_type: str,
    entity_type: str,
    entity_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    parent_chunk_id: str = '',
) -> Chunk:
    doc_id = str(parsed['doc_id'])
    clean_text = ' '.join(str(text or '').split()).strip()
    dedupe = _dedupe_key(chunk_type, entity_type, entity_name, clean_text)
    summary = _extract_summary(clean_text)
    source_kind = _source_kind(parsed)
    enriched_metadata = build_chunk_metadata(
        doc_id=doc_id,
        title=str(parsed.get('title') or doc_id),
        text=clean_text,
        section=section,
        chunk_type=chunk_type,
        entity_type=entity_type,
        entity_name=entity_name,
        source_kind=source_kind,
        parsed_metadata=parsed.get('metadata') or {},
        source_path=str(parsed.get('source_path', '') or ''),
        extra=metadata or {},
    )
    return Chunk(
        chunk_id=_chunk_id(doc_id, chunk_type, entity_name, section, clean_text),
        doc_id=doc_id,
        title=str(parsed.get('title') or doc_id),
        text=clean_text,
        page=page,
        section=section,
        chunk_type=chunk_type,
        entity_type=entity_type,
        entity_name=entity_name,
        priority=_priority(chunk_type),
        summary=summary,
        dedupe_key=dedupe,
        parent_chunk_id=parent_chunk_id,
        source_kind=source_kind,
        is_authoritative=_is_authoritative(parsed),
        metadata=enriched_metadata,
    )


def _append_chunk(chunks: List[Chunk], chunk: Chunk):
    if not chunk.text:
        return
    chunks.append(chunk)


def _chunk_record_docs(parsed: Dict[str, Any], blocks: List[Tuple[str, Optional[int]]]) -> List[Chunk]:
    doc_id = str(parsed['doc_id'])
    chunks: List[Chunk] = []
    if doc_id == 'ban-giam-hieu':
        triplets = [text for text, _ in blocks]
        for idx in range(0, len(triplets), 3):
            group = triplets[idx:idx + 3]
            if not group:
                continue
            name = group[0].strip()
            role = group[1].strip() if len(group) > 1 else ''
            email = group[2].strip() if len(group) > 2 else ''
            text = ' | '.join(part for part in [name, role, email] if part)
            chunk = _make_chunk(
                parsed=parsed,
                text=text,
                section='record',
                page=None,
                chunk_type='role',
                entity_type='person',
                entity_name=name,
                metadata={'name': name, 'role': role, 'email': email},
            )
            _append_chunk(chunks, chunk)
        return chunks

    if doc_id == 'co-so-vat-chat':
        for text, page in blocks:
            sentences = [piece.strip() for piece in re.split(r'(?<=[.!?])\s+', text) if piece.strip()]
            for sentence in sentences:
                chunk_type = 'overview'
                sentence_ascii = norm_text_ascii(sentence)
                if _line_is_contact(sentence):
                    chunk_type = 'contact'
                elif '4 dia diem dao tao' in sentence_ascii or '4 co so' in sentence_ascii:
                    chunk_type = 'overview'
                elif _line_is_location(sentence):
                    chunk_type = 'location'
                chunk = _make_chunk(
                    parsed=parsed,
                    text=sentence,
                    section='record',
                    page=page,
                    chunk_type=chunk_type,
                    entity_type='campus',
                    entity_name='UNETI',
                )
                _append_chunk(chunks, chunk)
        return chunks
    return chunks


def _emit_primary_and_children(
    parsed: Dict[str, Any],
    section: str,
    text: str,
    page: Optional[int],
    entity_type: str,
    entity_name: str,
    chunks: List[Chunk],
):
    base_type = _section_type(section, text, str(parsed['doc_id']))
    windows = _text_windows(text)
    primary_chunks: List[Chunk] = []
    for piece in windows:
        chunk = _make_chunk(
            parsed=parsed,
            text=piece,
            section=section,
            page=page,
            chunk_type=base_type,
            entity_type=entity_type,
            entity_name=entity_name,
        )
        _append_chunk(chunks, chunk)
        primary_chunks.append(chunk)

    parent_id = primary_chunks[0].chunk_id if primary_chunks else ''

    sentences = _split_sentences(text)
    for idx, sentence in enumerate(sentences):
        child_type = ''
        if _line_is_teaching(sentence):
            child_type = 'teaching'
        elif _line_is_training(sentence) and base_type not in {'contact', 'location'}:
            child_type = 'training'
        if not child_type:
            continue

        child_text = sentence
        if child_type == 'teaching' and idx > 0 and 'bo mon' in norm_text_ascii(sentences[idx - 1]):
            child_text = f'{sentences[idx - 1]} {sentence}'
        child = _make_chunk(
            parsed=parsed,
            text=child_text,
            section=section,
            page=page,
            chunk_type=child_type,
            entity_type=entity_type,
            entity_name=entity_name,
            metadata={'granularity': 'sentence', 'sentence_index': idx},
            parent_chunk_id=parent_id,
        )
        _append_chunk(chunks, child)

    bullet_lines = _split_bullets(text)
    if len(bullet_lines) >= 2 and base_type in {'function', 'duty', 'function_duty', 'howto'}:
        for bullet in bullet_lines:
            child = _make_chunk(
                parsed=parsed,
                text=bullet,
                section=section,
                page=page,
                chunk_type='howto' if base_type == 'howto' else 'fact',
                entity_type=entity_type,
                entity_name=entity_name,
                parent_chunk_id=parent_id,
            )
            _append_chunk(chunks, child)

    short_lines = _split_bullets(text)
    for line in short_lines:
        if len(line.split()) < 3:
            continue
        if _line_is_role(line):
            child = _make_chunk(
                parsed=parsed,
                text=line,
                section=section,
                page=page,
                chunk_type='role',
                entity_type=entity_type or 'person',
                entity_name=entity_name,
                parent_chunk_id=parent_id,
            )
            _append_chunk(chunks, child)
        elif _line_is_contact(line):
            child = _make_chunk(
                parsed=parsed,
                text=line,
                section=section,
                page=page,
                chunk_type='contact',
                entity_type=entity_type,
                entity_name=entity_name,
                parent_chunk_id=parent_id,
            )
            _append_chunk(chunks, child)
        elif _line_is_location(line) and base_type != 'history':
            child = _make_chunk(
                parsed=parsed,
                text=line,
                section=section,
                page=page,
                chunk_type='location',
                entity_type=entity_type,
                entity_name=entity_name,
                parent_chunk_id=parent_id,
            )
            _append_chunk(chunks, child)


def adaptive_chunk(parsed: Dict[str, Any]) -> List[Chunk]:
    doc_id = str(parsed['doc_id'])
    blocks = _normalize_blocks(parsed)
    if not blocks:
        return []

    if doc_id in {'ban-giam-hieu', 'co-so-vat-chat'}:
        return _chunk_record_docs(parsed, blocks)

    chunks: List[Chunk] = []
    entity_type = _entity_type_for_doc(doc_id)
    current_entity = ''
    current_section = 'overview'
    buffer: List[str] = []
    buffer_pages: List[int] = []

    def flush():
        nonlocal buffer, buffer_pages
        text = '\n'.join(part for part in buffer if part).strip()
        if text:
            _emit_primary_and_children(
                parsed=parsed,
                section=current_section,
                text=text,
                page=buffer_pages[0] if buffer_pages else None,
                entity_type=entity_type,
                entity_name=current_entity or parsed.get('title', ''),
                chunks=chunks,
            )
        buffer = []
        buffer_pages = []

    for text, page in blocks:
        if _looks_like_entity_heading(text, doc_id):
            flush()
            current_entity = text.strip()
            current_section = 'overview'
            continue
        if _looks_like_heading(text, doc_id):
            flush()
            current_section = text.strip()
            continue
        buffer.append(text)
        if page is not None:
            buffer_pages.append(int(page))

    flush()

    # Overview chunk for entity roots is intentionally retained even when there are
    # more detailed child chunks, because it is useful for concise answers.
    return chunks
