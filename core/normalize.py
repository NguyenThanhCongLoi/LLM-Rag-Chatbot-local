import re
import unicodedata

def norm_text_vn(text: str) -> str:
    s = unicodedata.normalize('NFC', str(text or '').strip().lower())
    s = re.sub(r'[^\w\s@./:+-]', ' ', s, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', s).strip()

def norm_text_ascii(text: str) -> str:
    s = norm_text_vn(text).replace('đ', 'd')
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    return re.sub(r'\s+', ' ', s).strip()

def slugify(text: str) -> str:
    s = norm_text_ascii(text)
    s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
    return s or 'item'
