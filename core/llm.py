from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import requests

DEFAULT_OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://127.0.0.1:11434')
DEFAULT_ANALYZER_MODEL = os.getenv('OLLAMA_ANALYZER_MODEL', 'qwen2.5:1.5b')
DEFAULT_ANSWER_MODEL = os.getenv('OLLAMA_ANSWER_MODEL', 'qwen2.5:1.5b')


@dataclass
class LLMConfig:
    enabled: bool = False
    analyzer_model: str = DEFAULT_ANALYZER_MODEL
    answer_model: str = DEFAULT_ANSWER_MODEL
    base_url: str = DEFAULT_OLLAMA_URL
    timeout: int = 20


def _format_contexts(contexts: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for i, ctx in enumerate(contexts, start=1):
        meta = ctx.get('metadata') or {}
        section = meta.get('section') or ''
        chunk_type = meta.get('chunk_type') or ''
        entity_name = meta.get('entity_name') or ''
        summary = meta.get('summary') or ''
        page = meta.get('page')
        score = ctx.get('score')
        header = [f'[{i}] {ctx.get("title", "")}'.strip()]
        if chunk_type:
            header.append(f'type={chunk_type}')
        if entity_name:
            header.append(f'entity={entity_name}')
        if section:
            header.append(f'section={section}')
        if page not in [None, '']:
            header.append(f'page={page}')
        topic_tags = meta.get('topic_tags') or []
        intent_tags = meta.get('intent_tags') or []
        if topic_tags:
            header.append('topics=' + ','.join(str(tag) for tag in topic_tags[:4]))
        if intent_tags:
            header.append('intents=' + ','.join(str(tag) for tag in intent_tags[:3]))
        if score not in [None, '']:
            header.append(f'score={float(score):.3f}')
        lines.append(' | '.join(header))
        body = str(ctx.get('text', '') or '').strip()
        limit = 420 if i <= 2 else 220
        if len(body) > limit:
            body = body[:limit].rstrip() + '...'
        if i >= 3 and summary:
            body = f'Tóm tắt: {summary}\nChi tiết: {body}'
        lines.append(body)
    return '\n'.join(lines)


class OllamaClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def _generate(self, model: str, prompt: str, temperature: float = 0.0) -> str:
        url = self.cfg.base_url.rstrip('/') + '/api/generate'
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': temperature},
        }
        resp = requests.post(url, json=payload, timeout=self.cfg.timeout)
        resp.raise_for_status()
        return (resp.json().get('response') or '').strip()

    def ping(self) -> tuple[bool, str]:
        try:
            out = self._generate(self.cfg.analyzer_model, 'Reply exactly with OK', 0.0)
            return ('OK' in out.upper(), out[:200])
        except Exception as e:
            return (False, str(e))

    def analyze(self, question: str, memory: Dict[str, Any], rule_domain: str | None, rule_qtype: str) -> Dict[str, Any]:
        prompt = f"""
Bạn là bộ phân tích truy vấn cho chatbot UNETI.

Mục tiêu:
- Chỉ phân tích câu hỏi.
- Không trả lời nội dung.
- Ưu tiên rule có sẵn hơn suy diễn.

Trả đúng 1 JSON theo schema:
{{
  "intent": "factoid|summary|compare|contact|howto|list",
  "domain": "ban_giam_hieu|co_so_vat_chat|lich_su_hinh_thanh|hoi_dong_truong|portal_howto|khoa_chuyen_mon|phong_ban_va_chuc_nang|general_docs",
  "use_history": false,
  "retrieval_query": "...",
  "top_k": 4,
  "answer_style": "direct|structured|detailed",
  "task_for_llm2": "..."
}}

Quy tắc cứng:
- Hiệu trưởng, phó hiệu trưởng, ban lãnh đạo -> ban_giam_hieu
- Cơ sở, địa chỉ, Minh Khai, Lĩnh Nam, Trần Hưng Đạo, Mỹ Xá -> co_so_vat_chat
- Cổng sinh viên, lịch học, lịch thi, đăng ký học phần, công nợ -> portal_howto
- Khi rule domain đã rõ thì giữ nguyên domain đó
- Retrieval query ngắn, giàu tín hiệu, không lan man
- Không dùng history nếu câu hiện tại đã đủ rõ

Memory: {json.dumps(memory, ensure_ascii=False)}
Rule domain: {rule_domain or ''}
Rule question type: {rule_qtype}
Question: {question}
""".strip()
        raw = self._generate(self.cfg.analyzer_model, prompt, 0.0)
        return _safe_json(raw)

    def answer(self, question: str, plan: Dict[str, Any], contexts: List[Dict[str, Any]]) -> str:
        prompt = f"""
Bạn là bộ trả lời dựa trên tài liệu UNETI.

Quy tắc bắt buộc:
- Chỉ dùng thông tin có trong CONTEXT.
- Không được bịa, không được suy ra ngoài evidence.
- Nếu CONTEXT không đủ thì trả đúng câu: Không tìm thấy đủ thông tin trong tài liệu đã cung cấp.
- Không đổi tên người, email, địa chỉ, số điện thoại, URL, ngày tháng.
- Không tiết lộ thông tin cá nhân không có trong CONTEXT.
- Trả lời bằng tiếng Việt, ngắn gọn, đúng trọng tâm.

Quy tắc trình bày:
- direct: trả lời trực tiếp 1-2 câu.
- structured: câu đầu trả lời trực tiếp, sau đó gạch đầu dòng các ý cần thiết.
- detailed: đầy đủ hơn nhưng vẫn bám CONTEXT.
- howto: ưu tiên các bước rõ ràng.
- list: chỉ liệt kê đúng các mục có evidence.

Intent: {plan.get('intent', 'factoid')}
Answer style: {plan.get('answer_style', 'direct')}
Chỉ dẫn thêm: {plan.get('task_for_llm2', '')}
Question: {question}
CONTEXT:
{_format_contexts(contexts)}
""".strip()
        return self._generate(self.cfg.answer_model, prompt, 0.0)


def _safe_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}
