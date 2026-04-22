from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    page: Optional[int] = None
    section: str = ""
    chunk_type: str = "generic"
    entity_type: str = ""
    entity_name: str = ""
    priority: int = 50
    summary: str = ""
    dedupe_key: str = ""
    parent_chunk_id: str = ""
    source_kind: str = "local"
    is_authoritative: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievedItem:
    kind: str
    score: float
    text: str
    source_id: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)
