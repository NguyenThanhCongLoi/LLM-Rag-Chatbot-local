from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .config import INDEX_DIR
from .metadata import metadata_search_text
from .models import Chunk
from .normalize import norm_text_ascii, norm_text_vn
from .vector_store import LocalVectorStore

DEFAULT_OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
DEFAULT_OLLAMA_EMBED_MODEL = os.getenv('UNETI_OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest').strip()
SENTENCE_MODEL_REF = os.getenv('UNETI_EMBED_MODEL', '').strip()
EMBED_LOCAL_ONLY = os.getenv('UNETI_EMBED_LOCAL_ONLY', '1').strip().lower() in {'1', 'true', 'yes'}
USE_OLLAMA_EMBED = os.getenv('UNETI_USE_OLLAMA_EMBED', '0').strip().lower() in {'1', 'true', 'yes', 'y'}

KEYWORD_STOPWORDS = {
    'va', 'voi', 'cua', 'cho', 'la', 'cac', 'nhung', 'mot', 'trong', 'the', 'nao', 'lam', 'gi',
    'tai', 'theo', 'duoc', 'den', 'tren', 'tu', 'khi', 'neu', 'thi', 'day', 'nay', 'do', 've',
    'phan', 'muc', 'noi', 'co', 'so', 'truong', 'uneti', 'sinh', 'vien', 'phong', 'khoa',
}


def _safe_name(value: str) -> str:
    return re.sub(r'[^a-z0-9._-]+', '_', value.lower()).strip('_') or 'default'


def _keyword_terms(text: str) -> List[str]:
    tokens = []
    for raw_tok in norm_text_ascii(text).split():
        tok = raw_tok.strip(".,:;!?()[]{}<>\"'`+-=*/\\|")
        if len(tok) < 3 or tok in KEYWORD_STOPWORDS or tok.isdigit():
            continue
        if sum(1 for ch in tok if ch.isalnum()) < max(2, len(tok) - 2):
            continue
        tokens.append(tok)
    return tokens


def _keyword_id(term: str) -> str:
    digest = hashlib.sha1(term.encode('utf-8', errors='ignore')).hexdigest()[:10]
    return f'kw-{_safe_name(term)[:24]}-{digest}'


@dataclass
class SemanticConfig:
    prefer_ollama: bool = USE_OLLAMA_EMBED
    ollama_base_url: str = DEFAULT_OLLAMA_URL
    ollama_embed_model: str = DEFAULT_OLLAMA_EMBED_MODEL
    ollama_timeout: int = 20
    prefer_sentence_transformers: bool = True
    use_local_only: bool = EMBED_LOCAL_ONLY
    sentence_model_ref: str = SENTENCE_MODEL_REF
    svd_rank: int = 64

    @property
    def ollama_enabled(self) -> bool:
        model = str(self.ollama_embed_model or '').strip().lower()
        return self.prefer_ollama and bool(model) and model not in {'0', 'false', 'none', 'off', 'disabled'}

    @property
    def sentence_enabled(self) -> bool:
        return bool(self.sentence_model_ref) and self.prefer_sentence_transformers


class _OllamaEncoder:
    @classmethod
    def encode(cls, texts: List[str], base_url: str, model: str, timeout: int) -> np.ndarray | None:
        if not texts or not model:
            return None
        embeddings: List[List[float]] = []
        url = base_url.rstrip('/') + '/api/embed'
        batch_size = 24
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            try:
                resp = requests.post(url, json={'model': model, 'input': batch}, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                batch_vectors = data.get('embeddings') or []
                if len(batch_vectors) != len(batch):
                    return None
                embeddings.extend(batch_vectors)
            except Exception:
                return None
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except Exception:
            return None
        if matrix.ndim != 2 or len(matrix) != len(texts):
            return None
        return normalize(matrix)


class _SentenceEncoder:
    _model = None
    _loaded_ref = None
    _attempted_ref = None
    _import_failed = False

    @classmethod
    def _load_library(cls):
        if cls._import_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            cls._import_failed = True
            return None
        return SentenceTransformer

    @classmethod
    def get_model(cls, ref: str, local_only: bool):
        if not ref:
            return None
        if cls._loaded_ref == ref and cls._model is not None:
            return cls._model
        if cls._attempted_ref == ref and cls._model is None:
            return None
        cls._attempted_ref = ref
        SentenceTransformer = cls._load_library()
        if SentenceTransformer is None:
            return None
        try:
            cls._model = SentenceTransformer(ref, local_files_only=local_only)
            cls._loaded_ref = ref
            return cls._model
        except Exception:
            cls._model = None
            return None

    @classmethod
    def encode(cls, texts: List[str], ref: str, local_only: bool) -> np.ndarray | None:
        model = cls.get_model(ref, local_only)
        if model is None:
            return None
        try:
            return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)
        except Exception:
            return None


class HybridIndex:
    def __init__(self, chunks: List[Chunk], semantic_cfg: SemanticConfig | None = None, vector_store: LocalVectorStore | None = None):
        self.chunks = chunks
        self.semantic_cfg = semantic_cfg or SemanticConfig()
        self.vector_store = vector_store or LocalVectorStore(INDEX_DIR / 'vectors')
        self._semantic_query_cache: Dict[str, np.ndarray] = {}

        self.texts_vn = [
            norm_text_vn(' '.join([
                c.text,
                c.summary,
                c.section,
                c.title,
                c.chunk_type,
                c.entity_type,
                c.entity_name,
                metadata_search_text(c.metadata),
            ]))
            for c in chunks
        ]
        self.texts_ascii = [
            norm_text_ascii(' '.join([
                c.text,
                c.summary,
                c.section,
                c.title,
                c.chunk_type,
                c.entity_type,
                c.entity_name,
                metadata_search_text(c.metadata),
            ]))
            for c in chunks
        ]
        self.title_ascii = [norm_text_ascii(c.title) for c in chunks]
        self.section_ascii = [norm_text_ascii(c.section) for c in chunks]
        self.chunk_type_ascii = [norm_text_ascii(c.chunk_type) for c in chunks]
        self.entity_ascii = [norm_text_ascii(c.entity_name) for c in chunks]
        self.metadata_ascii = [norm_text_ascii(metadata_search_text(c.metadata)) for c in chunks]
        self.tokenized_ascii = [t.split() for t in self.texts_ascii]
        self.keyword_terms = [
            _keyword_terms(' '.join([c.text, c.summary, c.section, c.entity_name, c.title, metadata_search_text(c.metadata)]))
            for c in chunks
        ]
        self.keyword_dictionary: Dict[str, Dict[str, float | int | str]] = {}
        self.keyword_idf: Dict[str, float] = {}
        self.chunk_keyword_weights: List[Dict[str, float]] = []
        self.chunk_keyword_ids: List[List[str]] = []
        self.chunk_keyword_terms: List[List[str]] = []
        self._fit_keyword_dictionary()

        self.vec_vn = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.vec_ascii = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.mat_vn = self.vec_vn.fit_transform(self.texts_vn) if self.texts_vn else None
        self.mat_ascii = self.vec_ascii.fit_transform(self.texts_ascii) if self.texts_ascii else None

        self.bm25 = BM25Okapi(self.tokenized_ascii) if self.tokenized_ascii else None
        self.semantic_matrix = None
        self.semantic_source = 'none'
        self.vector_backend = 'none'
        self._svd = None
        self._fit_semantic_matrix()

    def _fit_keyword_dictionary(self):
        if not self.keyword_terms:
            return
        num_docs = max(1, len(self.keyword_terms))
        df_counter = Counter()
        for terms in self.keyword_terms:
            df_counter.update(set(terms))

        self.keyword_dictionary = {}
        self.keyword_idf = {}
        for term, df in sorted(df_counter.items()):
            idf = math.log(num_docs / (1.0 + float(df))) + 1.0
            if idf < 0.0:
                idf = 0.0
            keyword_id = _keyword_id(term)
            self.keyword_idf[term] = float(idf)
            self.keyword_dictionary[term] = {
                'id': keyword_id,
                'term': term,
                'df': int(df),
                'idf': float(round(idf, 6)),
            }

        self.chunk_keyword_weights = []
        self.chunk_keyword_ids = []
        self.chunk_keyword_terms = []
        for terms in self.keyword_terms:
            tf_counter = Counter(terms)
            weights: Dict[str, float] = {}
            for term, tf in tf_counter.items():
                idf = self.keyword_idf.get(term, 0.0)
                if idf <= 0.0:
                    continue
                weights[term] = (1.0 + math.log(float(tf))) * idf
            ranked_terms = [
                term for term, _ in sorted(
                    weights.items(),
                    key=lambda item: (item[1], self.keyword_idf.get(item[0], 0.0), item[0]),
                    reverse=True,
                )[:12]
            ]
            self.chunk_keyword_weights.append({term: float(weights[term]) for term in ranked_terms})
            self.chunk_keyword_terms.append(ranked_terms)
            self.chunk_keyword_ids.append([str(self.keyword_dictionary[term]['id']) for term in ranked_terms if term in self.keyword_dictionary])

    def _doc_namespace(self) -> str:
        if not self.chunks:
            return 'empty'
        return self.chunks[0].doc_id

    def _text_signature(self) -> str:
        sha = hashlib.sha1()
        for chunk, text in zip(self.chunks, self.texts_vn):
            sha.update(chunk.chunk_id.encode('utf-8', errors='ignore'))
            sha.update(b'\0')
            sha.update(text.encode('utf-8', errors='ignore'))
            sha.update(b'\0')
        return sha.hexdigest()

    def _vector_signature(self, backend: str) -> str:
        base = self._text_signature()
        if backend == 'ollama':
            return f'{backend}:{self.semantic_cfg.ollama_embed_model}:{base}'
        if backend == 'sentence-transformers':
            return f'{backend}:{self.semantic_cfg.sentence_model_ref}:{base}'
        if backend == 'tfidf-dense':
            return f'{backend}:{base}'
        return f'{backend}:{self.semantic_cfg.svd_rank}:{base}'

    def _load_cached_vectors(self, filename: str, signature: str) -> np.ndarray | None:
        cached = self.vector_store.load(self._doc_namespace(), filename, signature)
        if cached is not None and len(cached) == len(self.chunks):
            return cached
        return None

    def _fit_semantic_matrix(self):
        if not self.chunks:
            return

        if self.semantic_cfg.ollama_enabled:
            signature = self._vector_signature('ollama')
            cache_name = f'ollama_{_safe_name(self.semantic_cfg.ollama_embed_model)}'
            cached = self._load_cached_vectors(cache_name, signature)
            if cached is not None:
                self.semantic_matrix = cached
                self.semantic_source = 'ollama'
                self.vector_backend = 'ollama'
                return
            ollama_embeddings = _OllamaEncoder.encode(
                self.texts_vn,
                base_url=self.semantic_cfg.ollama_base_url,
                model=self.semantic_cfg.ollama_embed_model,
                timeout=self.semantic_cfg.ollama_timeout,
            )
            if ollama_embeddings is not None:
                self.semantic_matrix = ollama_embeddings
                self.semantic_source = 'ollama'
                self.vector_backend = 'ollama'
                self.vector_store.save(
                    self._doc_namespace(),
                    cache_name,
                    signature,
                    ollama_embeddings,
                    metadata={
                        'backend': 'ollama',
                        'model_ref': self.semantic_cfg.ollama_embed_model,
                    },
                )
                return

        if self.semantic_cfg.sentence_enabled:
            signature = self._vector_signature('sentence-transformers')
            cached = self._load_cached_vectors('sentence_transformers', signature)
            if cached is not None:
                self.semantic_matrix = cached
                self.semantic_source = 'sentence-transformers'
                self.vector_backend = 'sentence-transformers'
                return
            sentence_embeddings = _SentenceEncoder.encode(
                self.texts_vn,
                ref=self.semantic_cfg.sentence_model_ref,
                local_only=self.semantic_cfg.use_local_only,
            )
            if sentence_embeddings is not None:
                self.semantic_matrix = sentence_embeddings
                self.semantic_source = 'sentence-transformers'
                self.vector_backend = 'sentence-transformers'
                self.vector_store.save(
                    self._doc_namespace(),
                    'sentence_transformers',
                    signature,
                    sentence_embeddings,
                    metadata={
                        'backend': 'sentence-transformers',
                        'model_ref': self.semantic_cfg.sentence_model_ref,
                    },
                )
                return

        if self.mat_ascii is None:
            return

        max_rank = min(self.mat_ascii.shape[0] - 1, self.mat_ascii.shape[1] - 1, self.semantic_cfg.svd_rank)
        if max_rank < 2:
            signature = self._vector_signature('tfidf-dense')
            cached = self._load_cached_vectors('tfidf_dense', signature)
            if cached is not None:
                self.semantic_matrix = cached
                self.semantic_source = 'tfidf-dense'
                self.vector_backend = 'tfidf-dense'
                return

            dense = normalize(self.mat_ascii.toarray()).astype(np.float32)
            self.semantic_matrix = dense
            self.semantic_source = 'tfidf-dense'
            self.vector_backend = 'tfidf-dense'
            self.vector_store.save(
                self._doc_namespace(),
                'tfidf_dense',
                signature,
                dense,
                metadata={'backend': 'tfidf-dense'},
            )
            return

        signature = self._vector_signature('svd')
        cached = self._load_cached_vectors('svd_dense', signature)
        if cached is not None:
            self.semantic_matrix = cached
            self.semantic_source = 'svd'
            self.vector_backend = 'svd'
            self._svd = TruncatedSVD(n_components=cached.shape[1], random_state=42)
            self._svd.fit(self.mat_ascii)
            return

        self._svd = TruncatedSVD(n_components=max_rank, random_state=42)
        dense = self._svd.fit_transform(self.mat_ascii)
        dense = normalize(dense).astype(np.float32)
        self.semantic_matrix = dense
        self.semantic_source = 'svd'
        self.vector_backend = 'svd'
        self.vector_store.save(
            self._doc_namespace(),
            'svd_dense',
            signature,
            dense,
            metadata={
                'backend': 'svd',
                'rank': int(dense.shape[1]),
            },
        )

    def semantic_scores(self, query_vn: str, query_ascii: str) -> np.ndarray | None:
        if not self.chunks or self.semantic_matrix is None:
            return None
        cache_key = f'{self.vector_backend}:{query_vn}\0{query_ascii}'
        cached = self._semantic_query_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.vector_backend == 'ollama':
            query_vec = _OllamaEncoder.encode(
                [query_vn],
                base_url=self.semantic_cfg.ollama_base_url,
                model=self.semantic_cfg.ollama_embed_model,
                timeout=self.semantic_cfg.ollama_timeout,
            )
            if query_vec is None:
                return None
            scores = np.matmul(self.semantic_matrix, query_vec[0])
            self._cache_semantic_scores(cache_key, scores)
            return scores

        if self.vector_backend == 'sentence-transformers':
            query_vec = _SentenceEncoder.encode(
                [query_vn],
                ref=self.semantic_cfg.sentence_model_ref,
                local_only=self.semantic_cfg.use_local_only,
            )
            if query_vec is None:
                return None
            scores = np.matmul(self.semantic_matrix, query_vec[0])
            self._cache_semantic_scores(cache_key, scores)
            return scores

        if self.vector_backend == 'svd' and self._svd is not None and self.mat_ascii is not None:
            q_mat = self.vec_ascii.transform([query_ascii])
            dense = self._svd.transform(q_mat)
            dense = normalize(dense).astype(np.float32)
            scores = np.matmul(self.semantic_matrix, dense[0])
            self._cache_semantic_scores(cache_key, scores)
            return scores

        if self.vector_backend == 'tfidf-dense' and self.mat_ascii is not None:
            dense = self.vec_ascii.transform([query_ascii]).toarray()
            dense = normalize(dense).astype(np.float32)
            scores = np.matmul(self.semantic_matrix, dense[0])
            self._cache_semantic_scores(cache_key, scores)
            return scores
        return None

    def _cache_semantic_scores(self, cache_key: str, scores: np.ndarray) -> None:
        if len(self._semantic_query_cache) >= 128:
            self._semantic_query_cache.pop(next(iter(self._semantic_query_cache)))
        self._semantic_query_cache[cache_key] = scores
