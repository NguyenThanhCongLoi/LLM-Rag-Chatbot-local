from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import fitz
from docx import Document
from .normalize import slugify

def parse_docx(path: Path) -> Dict[str, Any]:
    doc = Document(path)
    blocks = []
    for para in doc.paragraphs:
        txt = (para.text or '').strip()
        if txt:
            blocks.append(txt)
    return {
        'doc_id': slugify(path.stem),
        'title': path.stem,
        'blocks': blocks,
        'source_path': str(path),
        'metadata': {'channel': 'local', 'file_type': 'docx'},
    }

def parse_pdf(path: Path) -> Dict[str, Any]:
    pdf = fitz.open(path)
    blocks = []
    for i, page in enumerate(pdf):
        txt = page.get_text('text')
        for part in txt.split('\n'):
            part = part.strip()
            if part:
                blocks.append({'page': i + 1, 'text': part})
    return {
        'doc_id': slugify(path.stem),
        'title': path.stem,
        'blocks': blocks,
        'source_path': str(path),
        'metadata': {'channel': 'local', 'file_type': 'pdf'},
    }

def ingest_file(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    if ext == '.docx':
        return parse_docx(path)
    if ext == '.pdf':
        return parse_pdf(path)
    raise ValueError(f'Unsupported file: {path}')
