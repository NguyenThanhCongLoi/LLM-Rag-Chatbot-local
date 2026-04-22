from __future__ import annotations

import os
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from html import unescape
from typing import List
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests

DEFAULT_BASE_URL = os.getenv('UNETI_WEB_BASE_URL', 'https://uneti.edu.vn').rstrip('/')
DEFAULT_VERIFY_SSL = os.getenv('UNETI_WEB_VERIFY_SSL', '0').strip().lower() in {'1', 'true', 'yes'}
DEFAULT_TIMEOUT = int(os.getenv('UNETI_WEB_TIMEOUT', '12'))


@dataclass
class WebNotice:
    title: str
    link: str
    published_at: str = ''
    summary: str = ''


class UnetiWebClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT, verify_ssl: bool = DEFAULT_VERIFY_SSL):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.headers = {'User-Agent': 'UNETI-Document-Agent/4.2 (+local)'}

    def _get(self, url: str) -> str:
        resp = requests.get(url, headers=self.headers, timeout=self.timeout, verify=self.verify_ssl)
        resp.raise_for_status()
        return resp.text

    def _parse_feed(self, xml_text: str, limit: int) -> List[WebNotice]:
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return []
        items = []
        for item in root.findall('.//item')[:limit]:
            title = (item.findtext('title') or '').strip()
            link = (item.findtext('link') or '').strip()
            summary = unescape((item.findtext('description') or '').strip())
            pub_date = (item.findtext('pubDate') or '').strip()
            if pub_date:
                try:
                    pub_date = parsedate_to_datetime(pub_date).strftime('%Y-%m-%d')
                except Exception:
                    pass
            if not title or not link or 'uneti.edu.vn' not in link:
                continue
            items.append(WebNotice(title=title, link=link, published_at=pub_date, summary=summary))
        return items

    def latest_notices(self, limit: int = 5) -> List[WebNotice]:
        xml_text = self._get(f'{self.base_url}/category/thong-bao/feed/')
        return self._parse_feed(xml_text, limit)

    def search(self, query: str, limit: int = 5) -> List[WebNotice]:
        encoded = quote_plus(query)
        xml_text = self._get(f'{self.base_url}/?s={encoded}&feed=rss2')
        return self._parse_feed(xml_text, limit)
