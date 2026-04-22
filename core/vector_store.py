from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class LocalVectorStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _doc_dir(self, namespace: str) -> Path:
        path = self.root / namespace
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load(self, namespace: str, name: str, signature: str) -> np.ndarray | None:
        doc_dir = self._doc_dir(namespace)
        meta_path = doc_dir / f'{name}.json'
        vec_path = doc_dir / f'{name}.npy'
        if not meta_path.exists() or not vec_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            return None
        if meta.get('signature') != signature:
            return None
        try:
            matrix = np.load(vec_path)
        except Exception:
            return None
        if meta.get('rows') != int(matrix.shape[0]):
            return None
        return matrix

    def save(self, namespace: str, name: str, signature: str, matrix: np.ndarray, metadata: Dict[str, Any] | None = None):
        doc_dir = self._doc_dir(namespace)
        meta_path = doc_dir / f'{name}.json'
        vec_path = doc_dir / f'{name}.npy'
        np.save(vec_path, matrix)
        meta = {
            'signature': signature,
            'rows': int(matrix.shape[0]),
            'dim': int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        }
        if metadata:
            meta.update(metadata)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    def describe(self, namespace: str) -> List[Dict[str, Any]]:
        doc_dir = self._doc_dir(namespace)
        out: List[Dict[str, Any]] = []
        for meta_path in sorted(doc_dir.glob('*.json')):
            try:
                out.append(json.loads(meta_path.read_text(encoding='utf-8')))
            except Exception:
                continue
        return out
