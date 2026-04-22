from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.web_corpus import UnetiSiteCorpusBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None, help='Giới hạn số URL sitemap cần crawl')
    parser.add_argument('--force', action='store_true', help='Tải lại các trang đã cache')
    args = parser.parse_args()

    builder = UnetiSiteCorpusBuilder()
    manifest = builder.build(limit=args.limit, force=args.force)
    print(manifest)


if __name__ == '__main__':
    main()
