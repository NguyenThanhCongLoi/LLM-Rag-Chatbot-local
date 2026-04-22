from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .indexer import HybridIndex, _keyword_id, _keyword_terms
from .metadata import metadata_search_text
from .models import RetrievedItem
from .normalize import norm_text_ascii, norm_text_vn
from .seed_loader import load_seed_knowledge


def _normalize_scores(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    max_v = float(arr.max())
    min_v = float(arr.min())
    if max_v - min_v < 1e-9:
        return np.ones_like(arr) if max_v > 0 else np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def _query_prefers(query_ascii: str, *markers: str) -> bool:
    return any(marker in query_ascii for marker in markers)


def _chunk_type_boost(query_ascii: str, chunk_type: str) -> float:
    boosts = {
        'contact': _query_prefers(query_ascii, 'email', 'mail', 'website', 'web', 'dien thoai', 'sdt', 'lien he'),
        'location': _query_prefers(query_ascii, 'dia chi', 'o dau', 'co so', 'ha noi', 'nam dinh', 'minh khai', 'linh nam'),
        'role': _query_prefers(query_ascii, 'la ai', 'hieu truong', 'pho hieu truong', 'truong khoa', 'truong phong', 'chu tich'),
        'howto': _query_prefers(query_ascii, 'cach', 'nhu the nao', 'dang nhap', 'dang ky', 'tra cuu', 'xem ', 'huong dan'),
        'teaching': _query_prefers(query_ascii, 'day mon', 'mon gi', 'mon nao', 'giang day', 'hoc phan', 'day hoc'),
        'training': _query_prefers(query_ascii, 'dao tao gi', 'nganh nao', 'nganh gi', 'chuong trinh dao tao', 'ma nganh', 'tin chi'),
        'history': _query_prefers(query_ascii, 'lich su', 'thanh lap', 'tien than', 'giai doan'),
        'overview': _query_prefers(query_ascii, 'tong quan', 'gioi thieu', 'uneti', 'truong co'),
        'function': _query_prefers(query_ascii, 'lam gi', 'chuc nang', 'nhiem vu'),
        'duty': _query_prefers(query_ascii, 'lam gi', 'nhiem vu', 'phu trach'),
        'function_duty': _query_prefers(query_ascii, 'lam gi', 'chuc nang', 'nhiem vu'),
    }
    if boosts.get(chunk_type):
        return 0.12
    if chunk_type in {'overview', 'fact'}:
        return 0.03
    return 0.0


def _priority_boost(priority: int) -> float:
    clipped = max(0, min(100, int(priority or 0)))
    return clipped / 1000.0


def _keyword_rerank_score(query_ascii: str, keyword_weights: Dict[str, float], keyword_idf: Dict[str, float]) -> float:
    query_terms = []
    seen = set()
    for tok in _keyword_terms(query_ascii):
        if tok in seen:
            continue
        seen.add(tok)
        query_terms.append(tok)
    if not query_terms or not keyword_weights:
        return 0.0
    total_query_weight = sum(max(0.05, float(keyword_idf.get(term, 0.0))) for term in query_terms)
    if total_query_weight <= 0.0:
        return 0.0
    matched = sum(float(keyword_weights.get(term, 0.0)) for term in query_terms)
    return matched / total_query_weight


def _metadata_terms(metadata: Dict[str, object]) -> List[str]:
    terms: List[str] = []
    for key in ['topic_tags', 'intent_tags', 'search_terms', 'keywords', 'emails', 'urls', 'dates', 'years']:
        value = metadata.get(key)
        if isinstance(value, list):
            terms.extend(norm_text_ascii(item) for item in value if str(item).strip())
        elif value:
            terms.append(norm_text_ascii(value))
    for key in ['entity_norm', 'section_norm', 'chunk_type_norm', 'category']:
        if metadata.get(key):
            terms.append(norm_text_ascii(metadata.get(key)))
    return [term for term in terms if term]


def _metadata_boost(query_ascii: str, metadata: Dict[str, object]) -> float:
    if not query_ascii or not metadata:
        return 0.0
    score = 0.0
    exact_terms = 0
    partial_terms = 0
    for term in _metadata_terms(metadata):
        if not term or len(term) < 3:
            continue
        if ' ' in term and term in query_ascii:
            exact_terms += 1
        else:
            tokens = [tok for tok in term.split() if len(tok) >= 3]
            if tokens and all(tok in query_ascii for tok in tokens[:4]):
                exact_terms += 1
            elif any(tok in query_ascii for tok in tokens):
                partial_terms += 1
    score += min(0.18, exact_terms * 0.06)
    score += min(0.10, partial_terms * 0.025)

    topic_tags = {norm_text_ascii(tag) for tag in metadata.get('topic_tags', []) or []}
    intent_tags = {norm_text_ascii(tag) for tag in metadata.get('intent_tags', []) or []}
    if 'contact' in intent_tags and _query_prefers(query_ascii, 'email', 'mail', 'website', 'web', 'dien thoai', 'sdt', 'lien he'):
        score += 0.08
    if 'location' in intent_tags and _query_prefers(query_ascii, 'dia chi', 'o dau', 'co so', 'ha noi', 'nam dinh'):
        score += 0.08
    if 'howto' in intent_tags and _query_prefers(query_ascii, 'cach', 'nhu the nao', 'huong dan', 'dang ky', 'tra cuu', 'xem '):
        score += 0.08
    if 'teaching_lookup' in intent_tags and _query_prefers(query_ascii, 'day mon', 'mon gi', 'mon nao', 'giang day', 'hoc phan', 'day hoc'):
        score += 0.10
    if 'training_lookup' in intent_tags and _query_prefers(query_ascii, 'dao tao gi', 'nganh nao', 'nganh gi', 'chuong trinh dao tao', 'ma nganh'):
        score += 0.10
    if 'person_lookup' in intent_tags and _query_prefers(query_ascii, 'la ai', 'truong khoa', 'truong phong', 'hieu truong'):
        score += 0.08
    if 'faculty' in topic_tags and 'khoa' in query_ascii:
        score += 0.04
    if 'department' in topic_tags and 'phong' in query_ascii:
        score += 0.04
    return score


def _seed_keyword_index(domain_chunks: List[Dict[str, object]]) -> Dict[str, object]:
    keyword_terms_by_chunk: List[List[str]] = []
    df_counter = Counter()
    for chunk in domain_chunks:
        text = ' '.join([
            str(chunk.get('text', '') or '').strip(),
            str(chunk.get('summary', '') or '').strip(),
            str(chunk.get('section', '') or '').strip(),
            str(chunk.get('entity_name', '') or '').strip(),
            str(chunk.get('title', '') or '').strip(),
            ' '.join(str(item).strip() for item in chunk.get('keywords', []) or [] if str(item).strip()),
            metadata_search_text(chunk.get('metadata', {}) if isinstance(chunk.get('metadata', {}), dict) else {}),
        ]).strip()
        terms = _keyword_terms(text)
        keyword_terms_by_chunk.append(terms)
        df_counter.update(set(terms))

    num_docs = max(1, len(keyword_terms_by_chunk))
    keyword_idf: Dict[str, float] = {}
    keyword_dictionary: Dict[str, Dict[str, object]] = {}
    for term, df in sorted(df_counter.items()):
        idf = math.log(num_docs / (1.0 + float(df))) + 1.0
        if idf < 0.0:
            idf = 0.0
        keyword_idf[term] = float(idf)
        keyword_dictionary[term] = {
            'id': _keyword_id(term),
            'term': term,
            'df': int(df),
            'idf': float(round(idf, 6)),
        }

    chunk_keyword_weights: List[Dict[str, float]] = []
    chunk_keyword_terms: List[List[str]] = []
    chunk_keyword_ids: List[List[str]] = []
    for terms in keyword_terms_by_chunk:
        tf_counter = Counter(terms)
        weights: Dict[str, float] = {}
        for term, tf in tf_counter.items():
            idf = keyword_idf.get(term, 0.0)
            if idf <= 0.0:
                continue
            weights[term] = (1.0 + math.log(float(tf))) * idf
        ranked_terms = [
            term for term, _ in sorted(
                weights.items(),
                key=lambda item: (item[1], keyword_idf.get(item[0], 0.0), item[0]),
                reverse=True,
            )[:12]
        ]
        chunk_keyword_weights.append({term: float(weights[term]) for term in ranked_terms})
        chunk_keyword_terms.append(ranked_terms)
        chunk_keyword_ids.append([str(keyword_dictionary[term]['id']) for term in ranked_terms if term in keyword_dictionary])

    return {
        'keyword_idf': keyword_idf,
        'keyword_dictionary': keyword_dictionary,
        'chunk_keyword_weights': chunk_keyword_weights,
        'chunk_keyword_terms': chunk_keyword_terms,
        'chunk_keyword_ids': chunk_keyword_ids,
    }


def _entity_boost(query_ascii: str, entity_ascii: str) -> float:
    entity_ascii = str(entity_ascii or '').strip()
    if not entity_ascii:
        return 0.0
    if entity_ascii in query_ascii:
        return 0.30
    short_entity = entity_ascii
    for prefix in ['phong ', 'khoa ', 'ban ', 'truong ']:
        if short_entity.startswith(prefix):
            short_entity = short_entity[len(prefix):].strip()
            break
    if short_entity and short_entity in query_ascii:
        return 0.20
    entity_tokens = [tok for tok in entity_ascii.split() if len(tok) >= 3]
    if not entity_tokens:
        return 0.0
    hits = sum(1 for tok in entity_tokens if tok in query_ascii)
    if hits == len(entity_tokens):
        return 0.18
    if hits >= max(2, len(entity_tokens) - 1):
        return 0.12
    return 0.0


def _role_specific_boost(query_ascii: str, text_ascii: str) -> float:
    boost = 0.0
    if 'pho hieu truong' in query_ascii:
        if 'pho hieu truong' in text_ascii:
            boost += 0.22
        elif 'hieu truong' in text_ascii:
            boost -= 0.10
    elif 'hieu truong' in query_ascii:
        if 'hieu truong' in text_ascii and 'pho hieu truong' not in text_ascii:
            boost += 0.24
        elif 'pho hieu truong' in text_ascii:
            boost -= 0.12
    return boost


def _campus_specific_boost(query_ascii: str, text_ascii: str) -> float:
    boost = 0.0
    if 'ha noi' in query_ascii:
        if 'ha noi' in text_ascii:
            boost += 0.22
        if 'nam dinh' in text_ascii and 'ha noi' not in text_ascii:
            boost -= 0.08
    if 'nam dinh' in query_ascii:
        if 'nam dinh' in text_ascii:
            boost += 0.22
        if 'ha noi' in text_ascii and 'nam dinh' not in text_ascii:
            boost -= 0.08
    return boost


def _teaching_specific_boost(query_ascii: str, chunk_type: str, text_ascii: str) -> float:
    if not _query_prefers(query_ascii, 'day mon', 'mon gi', 'mon nao', 'giang day', 'hoc phan', 'day hoc'):
        return 0.0
    boost = 0.0
    if chunk_type == 'teaching':
        boost += 0.12
    if any(marker in text_ascii for marker in ['cac hoc phan', 'hoc phan ve']):
        boost += 0.30
    if any(marker in text_ascii for marker in ['toan hoc', 'logic hoc', 'vat ly', 'hoa hoc']):
        boost += 0.14
    if any(marker in text_ascii for marker in ['dia chi', 'email', 'website', 'linh nam', 'tran hung dao']):
        boost -= 0.35
    if 'nghien cuu' in text_ascii and not any(marker in text_ascii for marker in ['cac hoc phan', 'hoc phan ve']):
        boost -= 0.18
    return boost


def _training_specific_boost(query_ascii: str, chunk_type: str, text_ascii: str) -> float:
    if not _query_prefers(query_ascii, 'dao tao gi', 'nganh nao', 'nganh gi', 'chuong trinh dao tao', 'ma nganh'):
        return 0.0
    boost = 0.0
    if chunk_type == 'training':
        boost += 0.10
    if any(marker in text_ascii for marker in ['nganh dao tao', 'dao tao khdl', 'ma nganh', 'chuong trinh cu nhan']):
        boost += 0.22
    if 'khoa hoc du lieu' in text_ascii:
        boost += 0.10
    if any(marker in text_ascii for marker in ['dieu chinh chuong trinh', 'quy che hoat dong']):
        boost -= 0.18
    return boost


def _establishment_boost(query_ascii: str, chunk_type: str, text_ascii: str) -> float:
    if 'thanh lap' not in query_ascii:
        return 0.0
    boost = 0.0
    if 'duoc thanh lap' in text_ascii:
        boost += 0.12
    if 'truong dai hoc kinh te ky thuat cong nghiep duoc thanh lap' in text_ascii:
        boost += 0.22
    if 'quyet dinh' in text_ascii:
        boost += 0.10
    if chunk_type == 'overview':
        boost += 0.08
    if 'tien than' in text_ascii and 'tien than' not in query_ascii:
        boost -= 0.05
    if 'truong trung cap' in text_ascii and 'tien than' not in query_ascii:
        boost -= 0.06
    return boost


def _rerank_items(items: List[RetrievedItem], top_k: int) -> List[RetrievedItem]:
    if not items:
        return []
    items = sorted(items, key=lambda x: x.score, reverse=True)
    pool = items[:max(top_k * 4, 12)]
    chosen: List[RetrievedItem] = []
    while pool and len(chosen) < top_k:
        best_idx = 0
        best_score = None
        for idx, item in enumerate(pool):
            score = float(item.score)
            meta = item.metadata or {}
            for selected in chosen:
                sel_meta = selected.metadata or {}
                if meta.get('dedupe_key') and meta.get('dedupe_key') == sel_meta.get('dedupe_key'):
                    score -= 0.35
                if meta.get('parent_chunk_id') and meta.get('parent_chunk_id') == sel_meta.get('parent_chunk_id'):
                    score -= 0.08
                if meta.get('entity_name') and meta.get('entity_name') == sel_meta.get('entity_name'):
                    score -= 0.04
                if meta.get('chunk_type') and meta.get('chunk_type') == sel_meta.get('chunk_type'):
                    score -= 0.02
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(pool.pop(best_idx))
    return chosen


class KnowledgeStore:
    def __init__(self):
        self.seed = load_seed_knowledge()['domains']
        self.indexes: Dict[str, HybridIndex] = {}
        self.doc_chunks = {}
        self.seed_keyword_indexes = {
            domain: _seed_keyword_index(list(body.get('chunks', []) or []))
            for domain, body in self.seed.items()
        }
        self._seed_retrieve_cache = {}
        self._chunk_retrieve_cache = {}

    def add_chunks(self, doc_id: str, chunks):
        self.doc_chunks[doc_id] = chunks
        self.indexes[doc_id] = HybridIndex(chunks)
        self._chunk_retrieve_cache = {}

    def _token_overlap_score(self, query_tokens: Iterable[str], doc_tokens: Iterable[str]) -> float:
        query_set = {tok for tok in query_tokens if tok}
        doc_set = {tok for tok in doc_tokens if tok}
        if not query_set or not doc_set:
            return 0.0
        inter = len(query_set & doc_set)
        return inter / max(1, len(query_set))

    def _seed_score(self, query_ascii: str, text_ascii: str, query_tokens: List[str], extra_boost: float = 0.0) -> float:
        overlap = self._token_overlap_score(query_tokens, text_ascii.split())
        exact_phrase = 0.25 if query_ascii and query_ascii in text_ascii else 0.0
        token_hits = 0.06 * sum(1 for tok in query_tokens if len(tok) >= 4 and tok in text_ascii)
        return overlap + exact_phrase + token_hits + extra_boost

    def seed_retrieve(self, query: str, domain: str, top_k: int = 5) -> List[RetrievedItem]:
        cache_key = (norm_text_ascii(query), domain, int(top_k))
        cached = getattr(self, '_seed_retrieve_cache', {}).get(cache_key)
        if cached is not None:
            return list(cached)
        q_ascii = norm_text_ascii(query)
        q_tokens = q_ascii.split()
        out: List[RetrievedItem] = []
        data = self.seed.get(domain, {})
        domain_chunks = data.get('chunks', []) or []
        keyword_index = self.seed_keyword_indexes.get(domain, {})
        keyword_idf = keyword_index.get('keyword_idf', {}) if isinstance(keyword_index, dict) else {}
        chunk_keyword_weights = keyword_index.get('chunk_keyword_weights', []) if isinstance(keyword_index, dict) else []
        chunk_keyword_terms = keyword_index.get('chunk_keyword_terms', []) if isinstance(keyword_index, dict) else []
        chunk_keyword_ids = keyword_index.get('chunk_keyword_ids', []) if isinstance(keyword_index, dict) else []

        if domain_chunks:
            for i, chunk in enumerate(domain_chunks):
                text = str(chunk.get('text', '') or '').strip()
                if not text:
                    continue
                keywords = ' '.join(str(item).strip() for item in chunk.get('keywords', []) if str(item).strip())
                section = str(chunk.get('section', '') or '').strip()
                entity_name = str(chunk.get('entity_name', '') or '').strip()
                chunk_type = str(chunk.get('chunk_type', '') or '').strip()
                title = str(chunk.get('title', '') or data.get('title', domain))
                metadata = dict(chunk.get('metadata', {}) or {})
                search_text = ' '.join(part for part in [text, keywords, section, entity_name, title, chunk_type, metadata_search_text(metadata)] if part)
                score = self._seed_score(q_ascii, norm_text_ascii(search_text), q_tokens, extra_boost=0.04)
                score += _chunk_type_boost(q_ascii, norm_text_ascii(chunk_type))
                score += _entity_boost(q_ascii, norm_text_ascii(entity_name))
                score += _metadata_boost(q_ascii, metadata)
                score += _priority_boost(int(chunk.get('priority', 0) or 0))
                text_ascii = norm_text_ascii(text)
                if chunk_type:
                    score += _establishment_boost(q_ascii, norm_text_ascii(chunk_type), text_ascii)
                score += _teaching_specific_boost(q_ascii, norm_text_ascii(chunk_type), text_ascii)
                score += _training_specific_boost(q_ascii, norm_text_ascii(chunk_type), text_ascii)
                score += _role_specific_boost(q_ascii, text_ascii)
                score += _campus_specific_boost(q_ascii, text_ascii)
                keyword_weights = chunk_keyword_weights[i] if i < len(chunk_keyword_weights) else {}
                keyword_terms = chunk_keyword_terms[i] if i < len(chunk_keyword_terms) else list(chunk.get('keywords', []) or [])
                keyword_ids = chunk_keyword_ids[i] if i < len(chunk_keyword_ids) else list(chunk.get('keyword_ids', []) or [])
                keyword_score = _keyword_rerank_score(q_ascii, keyword_weights, keyword_idf)
                score += min(0.22, keyword_score * 0.22)
                metadata.update({
                    'doc_id': chunk.get('doc_id', f'seed-{domain}'),
                    'section': chunk.get('section', ''),
                    'chunk_type': chunk.get('chunk_type', ''),
                    'entity_type': chunk.get('entity_type', ''),
                    'entity_name': chunk.get('entity_name', ''),
                    'priority': chunk.get('priority', 0),
                    'summary': chunk.get('summary', ''),
                    'dedupe_key': chunk.get('dedupe_key', ''),
                    'parent_chunk_id': chunk.get('parent_chunk_id', ''),
                    'source_kind': chunk.get('source_kind', 'seed'),
                    'is_authoritative': chunk.get('is_authoritative', True),
                    'keyword_score': round(float(keyword_score), 6),
                    'keywords': keyword_terms,
                    'keyword_ids': keyword_ids,
                    'keyword_weights': keyword_weights,
                    'source_url': chunk.get('source_url', ''),
                })
                out.append(
                    RetrievedItem(
                        kind=str(chunk.get('kind', 'chunk') or 'chunk'),
                        score=float(score),
                        text=text,
                        source_id=str(chunk.get('chunk_id', f'{domain}:chunk:{i}') or f'{domain}:chunk:{i}'),
                        title=title,
                        metadata=metadata,
                    )
                )

            result = _rerank_items(out, top_k)
            self._remember_seed_retrieve(cache_key, result)
            return result

        for i, rec in enumerate(data.get('records', [])):
            text = ' | '.join(str(v) for v in rec.values())
            score = self._seed_score(q_ascii, norm_text_ascii(text), q_tokens, extra_boost=0.2)
            out.append(
                RetrievedItem(
                    kind='record',
                    score=score,
                    text=text,
                    source_id=f'{domain}:record:{i}',
                    title=data.get('title', domain),
                    metadata=rec | {
                        'chunk_type': 'record',
                        'priority': 92,
                        'dedupe_key': norm_text_ascii(text),
                        'is_authoritative': True,
                        'source_kind': 'seed',
                    },
                )
            )

        for i, fact in enumerate(data.get('facts', [])):
            score = self._seed_score(q_ascii, norm_text_ascii(fact), q_tokens, extra_boost=0.1)
            out.append(
                RetrievedItem(
                    kind='fact',
                    score=score,
                    text=fact,
                    source_id=f'{domain}:fact:{i}',
                    title=data.get('title', domain),
                    metadata={
                        'chunk_type': 'fact',
                        'priority': 88,
                        'dedupe_key': norm_text_ascii(fact),
                        'is_authoritative': True,
                        'source_kind': 'seed',
                    },
                )
            )

        for i, qa in enumerate(data.get('qa', [])):
            keywords = ' '.join(qa.get('keywords', []))
            qa_text = f"{qa.get('question', '')} {qa.get('answer', '')} {keywords}"
            score = self._seed_score(q_ascii, norm_text_ascii(qa_text), q_tokens, extra_boost=0.12)
            out.append(
                RetrievedItem(
                    kind='qa',
                    score=score,
                    text=qa['answer'],
                    source_id=f'{domain}:qa:{i}',
                    title=qa['question'],
                    metadata=qa | {
                        'chunk_type': 'qa',
                        'priority': 94,
                        'dedupe_key': norm_text_ascii(qa['answer']),
                        'is_authoritative': True,
                        'source_kind': 'seed',
                    },
                )
            )

        for i, location in enumerate(data.get('locations', [])):
            text = ' | '.join(str(v) for v in location.values())
            score = self._seed_score(q_ascii, norm_text_ascii(text), q_tokens, extra_boost=0.18)
            out.append(
                RetrievedItem(
                    kind='location',
                    score=score,
                    text=text,
                    source_id=f'{domain}:location:{i}',
                    title=data.get('title', domain),
                    metadata=location | {
                        'chunk_type': 'location',
                        'priority': 95,
                        'dedupe_key': norm_text_ascii(text),
                        'is_authoritative': True,
                        'source_kind': 'seed',
                    },
                )
            )

        out.sort(key=lambda x: x.score, reverse=True)
        result = out[:top_k]
        self._remember_seed_retrieve(cache_key, result)
        return result

    def chunk_retrieve(self, query: str, domain_doc_ids: List[str], top_k: int = 6) -> List[RetrievedItem]:
        cache_key = (norm_text_ascii(query), tuple(domain_doc_ids), int(top_k))
        cached = getattr(self, '_chunk_retrieve_cache', {}).get(cache_key)
        if cached is not None:
            return list(cached)
        q_vn = norm_text_vn(query)
        q_ascii = norm_text_ascii(query)
        q_tokens = q_ascii.split()
        items: List[RetrievedItem] = []

        for doc_id in domain_doc_ids:
            idx = self.indexes.get(doc_id)
            if not idx or idx.mat_vn is None or idx.mat_ascii is None:
                continue

            score_parts: Dict[str, np.ndarray] = {
                'tfidf_vn': _normalize_scores(cosine_similarity(idx.vec_vn.transform([q_vn]), idx.mat_vn).ravel()),
                'tfidf_ascii': _normalize_scores(cosine_similarity(idx.vec_ascii.transform([q_ascii]), idx.mat_ascii).ravel()),
                'overlap': _normalize_scores([self._token_overlap_score(q_tokens, toks) for toks in idx.tokenized_ascii]),
            }
            if idx.bm25 is not None:
                score_parts['bm25'] = _normalize_scores(idx.bm25.get_scores(q_tokens))
            semantic_scores = idx.semantic_scores(q_vn, q_ascii)
            if semantic_scores is not None:
                score_parts['semantic'] = _normalize_scores(semantic_scores)

            weights = {
                'tfidf_vn': 0.28,
                'tfidf_ascii': 0.16,
                'overlap': 0.18,
                'bm25': 0.18,
                'semantic': 0.20,
            }
            used = [name for name in score_parts if name in weights]
            if not used:
                continue
            total_weight = sum(weights[name] for name in used)
            scores = sum(weights[name] * score_parts[name] for name in used) / total_weight

            for i, (c, score) in enumerate(zip(idx.chunks, scores)):
                boost = 0.0
                title_ascii = idx.title_ascii[i]
                section_ascii = idx.section_ascii[i]
                chunk_type_ascii = idx.chunk_type_ascii[i]
                entity_ascii = idx.entity_ascii[i]
                text_ascii = idx.texts_ascii[i]
                keyword_score = _keyword_rerank_score(q_ascii, idx.chunk_keyword_weights[i], idx.keyword_idf)
                if any(tok in title_ascii for tok in q_tokens[:6]):
                    boost += 0.08
                if any(tok in section_ascii for tok in q_tokens[:6]):
                    boost += 0.08
                if any(tok in entity_ascii for tok in q_tokens[:6]):
                    boost += 0.10
                boost += _entity_boost(q_ascii, entity_ascii)
                boost += _chunk_type_boost(q_ascii, chunk_type_ascii)
                boost += _metadata_boost(q_ascii, c.metadata or {})
                boost += _priority_boost(getattr(c, 'priority', 0))
                boost += _role_specific_boost(q_ascii, text_ascii)
                boost += _campus_specific_boost(q_ascii, text_ascii)
                boost += _establishment_boost(q_ascii, chunk_type_ascii, text_ascii)
                boost += _teaching_specific_boost(q_ascii, chunk_type_ascii, text_ascii)
                boost += _training_specific_boost(q_ascii, chunk_type_ascii, text_ascii)
                boost += min(0.22, keyword_score * 0.22)
                exact_hits = sum(1 for tok in q_tokens if len(tok) >= 5 and tok in text_ascii)
                boost += min(0.16, exact_hits * 0.03)

                item_metadata = dict(c.metadata or {})
                item_metadata.update(
                    {
                        'doc_id': c.doc_id,
                        'section': c.section,
                        'page': c.page,
                        'chunk_type': c.chunk_type,
                        'entity_type': c.entity_type,
                        'entity_name': c.entity_name,
                        'priority': c.priority,
                        'summary': c.summary,
                        'dedupe_key': c.dedupe_key,
                        'parent_chunk_id': c.parent_chunk_id,
                        'source_kind': c.source_kind,
                        'is_authoritative': c.is_authoritative,
                        'keyword_score': round(float(keyword_score), 6),
                        'keyword_ids': idx.chunk_keyword_ids[i],
                        'keywords': idx.chunk_keyword_terms[i],
                        'keyword_weights': idx.chunk_keyword_weights[i],
                        'semantic_source': idx.semantic_source,
                        'vector_backend': idx.vector_backend,
                    }
                )
                items.append(
                    RetrievedItem(
                        kind='chunk',
                        score=float(score + boost),
                        text=c.text,
                        source_id=c.chunk_id,
                        title=c.title,
                        metadata=item_metadata,
                    )
                )

        result = _rerank_items(items, top_k)
        self._remember_chunk_retrieve(cache_key, result)
        return result

    def _remember_seed_retrieve(self, cache_key, result: List[RetrievedItem]) -> None:
        cache = getattr(self, '_seed_retrieve_cache', None)
        if cache is None:
            self._seed_retrieve_cache = {}
            cache = self._seed_retrieve_cache
        if len(cache) >= 256:
            cache.pop(next(iter(cache)))
        cache[cache_key] = list(result)

    def _remember_chunk_retrieve(self, cache_key, result: List[RetrievedItem]) -> None:
        cache = getattr(self, '_chunk_retrieve_cache', None)
        if cache is None:
            self._chunk_retrieve_cache = {}
            cache = self._chunk_retrieve_cache
        if len(cache) >= 256:
            cache.pop(next(iter(cache)))
        cache[cache_key] = list(result)
