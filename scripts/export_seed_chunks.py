from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import PARSED_DIR
from core.seed_loader import load_seed_chunks, load_seed_knowledge


def export_seed_chunks(output_name: str = 'seed_chunks_v2.json') -> dict:
    chunks = load_seed_chunks()
    data = load_seed_knowledge()
    out_path = PARSED_DIR / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': str(data.get('seed_chunk_schema', 'v2') or 'v2'),
        'count': len(chunks),
        'chunks': chunks,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return {
        'schema': payload['schema'],
        'count': len(chunks),
        'path': str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Export normalized seed/runtime chunks to storage/parsed.')
    parser.add_argument('--output-name', default='seed_chunks_v2.json', help='Output filename inside storage/parsed')
    args = parser.parse_args()
    summary = export_seed_chunks(output_name=args.output_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
