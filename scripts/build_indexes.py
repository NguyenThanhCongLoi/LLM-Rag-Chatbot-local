import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.pipeline_v4 import UnetiDocumentAgentV4Max


def main():
    parser = argparse.ArgumentParser(description='Build/load chunk and vector indexes for local UNETI documents.')
    parser.add_argument('--include-web', action='store_true', help='Also build cached web document indexes.')
    args = parser.parse_args()

    agent = UnetiDocumentAgentV4Max()
    doc_ids = agent.local_docs()
    if args.include_web:
        doc_ids += agent.web_docs()
    agent._ensure_docs_loaded(doc_ids)
    print('Loaded docs:', agent.available_docs())
    for doc_id, idx in sorted(agent.store.indexes.items()):
        print(f'- {doc_id}: semantic_source={idx.semantic_source}, vector_backend={idx.vector_backend}, chunks={len(idx.chunks)}')


if __name__ == '__main__':
    main()
