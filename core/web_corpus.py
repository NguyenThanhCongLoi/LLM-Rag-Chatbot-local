from __future__ import annotations

import hashlib
import json
import re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List
import xml.etree.ElementTree as ET

import requests

from .config import WEB_PARSED_DIR
from .normalize import norm_text_ascii, slugify


def _safe_filename(value: str) -> str:
    slug = slugify(value)[:80]
    digest = hashlib.sha1(value.encode('utf-8', errors='ignore')).hexdigest()[:12]
    return f'{slug}-{digest}.json'


NAVIGATION_BLOCKS = {
    'skip to content',
    'lich su hinh thanh',
    'su mang tam nhin phat trien va gia tri cot loi',
    'triet ly giao duc',
    'co cau to chuc',
    'so do co cau to chuc',
    'bch dang uy',
    'hoi dong truong',
    'ban giam hieu',
    'doan tn hoi sv',
}


def _clean_blocks(blocks: List[str], title: str) -> List[str]:
    cleaned: List[str] = []
    title_norm = norm_text_ascii(title)
    for block in blocks:
        text = ' '.join(block.replace('\xa0', ' ').split())
        norm = norm_text_ascii(text)
        if not norm:
            continue
        if norm in NAVIGATION_BLOCKS:
            continue
        if 'thong tin tuyen sinh' in norm or 'dong gop y kien' in norm or 'thu vien so' in norm:
            continue
        if norm == title_norm:
            continue
        if len(norm.split()) <= 2:
            continue
        cleaned.append(text)
    return cleaned


class _BlockTextParser(HTMLParser):
    BLOCK_TAGS = {'h1', 'h2', 'h3', 'h4', 'p', 'li'}
    SKIP_TAGS = {'script', 'style', 'noscript'}
    TARGET_TAGS = {'article', 'main'}
    TARGET_CLASSES = {'entry-content', 'entry-summary', 'inside-article', 'article-content'}

    def __init__(self):
        super().__init__()
        self.blocks: List[str] = []
        self._current: List[str] = []
        self._skip_depth = 0
        self.title = ''
        self._in_title = False
        self._target_depth = 0
        self._global_depth = 0
        self._force_capture = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self._global_depth += 1
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        attrs_dict = {k.lower(): (v or '') for k, v in attrs}
        class_tokens = set((attrs_dict.get('class', '') or '').lower().split())
        if tag in self.TARGET_TAGS or (class_tokens & self.TARGET_CLASSES):
            if self._target_depth == 0:
                self._target_depth = self._global_depth
        if self._skip_depth:
            return
        if tag == 'title':
            self._in_title = True
            return
        if tag in self.BLOCK_TAGS and self._current:
            self._flush()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == 'title':
            self._in_title = False
        elif not self._skip_depth and tag in self.BLOCK_TAGS:
            self._flush()
        if self._target_depth and self._global_depth == self._target_depth:
            self._target_depth = 0
        self._global_depth = max(0, self._global_depth - 1)

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = ' '.join((data or '').split())
        if not text:
            return
        if self._in_title:
            self.title += (' ' + text if self.title else text)
        else:
            if self._target_depth == 0 and not self._force_capture:
                return
            self._current.append(text)

    def _flush(self):
        text = ' '.join(self._current).strip()
        self._current = []
        if text and len(text) >= 3:
            self.blocks.append(unescape(text))


class UnetiSiteCorpusBuilder:
    def __init__(self, base_url: str = 'https://uneti.edu.vn', timeout: int = 15, verify_ssl: bool = False):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.headers = {'User-Agent': 'UNETI-Document-Agent/4.2 (+crawler)'}

    def _get(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=self.timeout, verify=self.verify_ssl)
        resp.raise_for_status()
        return resp.text

    def _parse_sitemap(self, xml_text: str) -> List[str]:
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []
        urls: List[str] = []
        for loc in root.findall('.//{*}loc'):
            value = (loc.text or '').strip()
            if value.startswith(self.base_url):
                urls.append(value)
        return urls

    def discover_urls(self, limit: int | None = None) -> List[str]:
        sitemap_urls = [
            f'{self.base_url}/post-sitemap.xml',
            f'{self.base_url}/page-sitemap.xml',
        ]
        discovered: List[str] = []
        seen = set()
        for sitemap in sitemap_urls:
            try:
                xml_text = self._get(sitemap)
            except Exception:
                continue
            for url in self._parse_sitemap(xml_text):
                if url in seen:
                    continue
                seen.add(url)
                discovered.append(url)
                if limit and len(discovered) >= limit:
                    return discovered
        return discovered

    def fetch_page(self, url: str) -> Dict[str, object] | None:
        try:
            html_text = self._get(url)
        except Exception:
            return None
        parser = _BlockTextParser()
        try:
            parser.feed(html_text)
        except Exception:
            return None
        title = parser.title.strip() or url
        blocks = parser.blocks
        if not blocks:
            parser = _BlockTextParser()
            parser._force_capture = True
            try:
                parser.feed(html_text)
            except Exception:
                return None
            title = parser.title.strip() or title
            blocks = parser.blocks
        blocks = _clean_blocks(blocks, title)
        if not blocks:
            return None
        title = re.sub(r'\s*-\s*TRƯỜNG ĐẠI HỌC.*$', '', title, flags=re.I).strip() or title
        doc_id = f"web-{slugify(title)}"
        return {
            'doc_id': doc_id,
            'title': title,
            'blocks': blocks,
            'source_url': url,
            'source_path': url,
            'metadata': {'channel': 'web', 'url': url},
        }

    def save_page(self, page: Dict[str, object], output_dir: Path = WEB_PARSED_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
        url = str(page.get('source_url') or page.get('source_path') or page.get('title') or '')
        path = output_dir / _safe_filename(url)
        path.write_text(json.dumps(page, ensure_ascii=False, indent=2), encoding='utf-8')
        return path

    def build(self, limit: int | None = None, force: bool = False, output_dir: Path = WEB_PARSED_DIR) -> Dict[str, int]:
        output_dir.mkdir(parents=True, exist_ok=True)
        urls = self.discover_urls(limit=limit)
        saved = 0
        skipped = 0
        for url in urls:
            out_path = output_dir / _safe_filename(url)
            if out_path.exists() and not force:
                skipped += 1
                continue
            page = self.fetch_page(url)
            if not page:
                skipped += 1
                continue
            self.save_page(page, output_dir=output_dir)
            saved += 1
        manifest = {
            'base_url': self.base_url,
            'count_urls': len(urls),
            'saved': saved,
            'skipped': skipped,
        }
        (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        return manifest
