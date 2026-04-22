from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .chunking import adaptive_chunk
from .config import DATA_DIR, WEB_PARSED_DIR
from .guards import answer_guard
from .ingest import ingest_file
from .llm import LLMConfig, OllamaClient
from .normalize import norm_text_ascii, slugify
from .retrieval import KnowledgeStore
from .reward_model import load_reward_model
from .review_store import lookup_policy_review, lookup_review, record_review_usage
from .routing import (
    classify_policy_query,
    classify_sensitive_query,
    clarification_reply,
    detect_campus_filter,
    detect_domain,
    detect_meta_query,
    detect_portal_topic,
    detect_query_family,
    detect_question_type,
    expand_query,
    is_control,
    is_out_of_scope_query,
    is_low_signal_query,
    is_policy_query,
    is_sensitive_query,
    is_web_notice_query,
    meta_reply,
    out_of_scope_reply,
    policy_query_reason,
    sensitive_query_reason,
    should_use_history,
)
from .seed_loader import load_seed_knowledge
from .web_fallback import UnetiWebClient


DOC_HIERARCHY = {
    'overview': ['lich-su-hinh-thanh', 'co-so-vat-chat', 'ban-giam-hieu', 'danh-sach-cac-thanh-vien-hoi-dong-truong'],
    'organization': ['phong-ban-va-chuc-nang', 'khoa-chuyen-mon'],
}


class UnetiDocumentAgentV4Max:
    def __init__(self, llm_config: LLMConfig | None = None):
        self.cfg = llm_config or LLMConfig(enabled=False)
        self.llm = OllamaClient(self.cfg) if self.cfg.enabled else None
        self.store = KnowledgeStore()
        self.seed = load_seed_knowledge()['domains']
        self.web = UnetiWebClient()
        self.doc_sources: Dict[str, Any] = {}
        self._register_seed_docs()

    def set_llm_config(self, cfg: LLMConfig):
        self.cfg = cfg
        self.llm = OllamaClient(cfg) if cfg.enabled else None

    def _register_seed_docs(self):
        for path in sorted(DATA_DIR.iterdir()):
            if path.suffix.lower() not in {'.docx', '.pdf'}:
                continue
            self.doc_sources[slugify(path.stem)] = path
        for path in sorted(WEB_PARSED_DIR.glob('*.json')):
            if path.name == 'manifest.json':
                continue
            try:
                data = _load_parsed_json(path)
                self.doc_sources[str(data['doc_id'])] = path
            except Exception:
                continue

    def available_docs(self) -> List[str]:
        return sorted(self.doc_sources.keys())

    def local_docs(self) -> List[str]:
        return sorted(doc_id for doc_id in self.doc_sources if not str(doc_id).startswith('web-'))

    def web_docs(self) -> List[str]:
        return sorted(doc_id for doc_id in self.doc_sources if str(doc_id).startswith('web-'))

    def _ensure_docs_loaded(self, doc_ids: List[str]):
        for doc_id in doc_ids:
            if doc_id in self.store.doc_chunks:
                continue
            path = self.doc_sources.get(doc_id)
            if not path:
                continue
            try:
                parsed = _load_parsed_json(path) if path.suffix.lower() == '.json' else ingest_file(path)
                chunks = adaptive_chunk(parsed)
                self.store.add_chunks(parsed['doc_id'], chunks)
            except Exception:
                continue

    def _tiered_chunk_retrieve(self, query: str, doc_tiers: List[List[str]], top_k: int) -> Tuple[List[Any], List[Dict[str, Any]]]:
        collected: List[Any] = []
        tier_debug: List[Dict[str, Any]] = []
        for idx, tier_docs in enumerate(doc_tiers, start=1):
            if not tier_docs:
                continue
            self._ensure_docs_loaded(tier_docs)
            tier_items = self.store.chunk_retrieve(query, tier_docs, top_k=top_k)
            top_score = float(tier_items[0].score) if tier_items else 0.0
            tier_debug.append({
                'tier': idx,
                'docs': tier_docs,
                'count': len(tier_items),
                'top_score': top_score,
            })
            collected.extend(tier_items)
            if _is_sufficient_tier_evidence(tier_items):
                break
        return collected, tier_debug

    def answer(self, question: str, memory: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        memory = dict(memory or {})
        debug: Dict[str, Any] = {'question': question}
        debug['query_family'] = detect_query_family(question, memory)

        if is_control(question):
            return 'Đã xóa ngữ cảnh trước. Bạn có thể hỏi một câu mới.', debug | {'route': 'control'}, _empty_memory(question)

        meta_intent = detect_meta_query(question)
        if meta_intent:
            answer = meta_reply(meta_intent)
            return answer, debug | {'route': 'meta', 'meta_intent': meta_intent}, _memory_with_channel(memory, question, channel='meta', answer=answer)

        if is_low_signal_query(question):
            answer = clarification_reply()
            return answer, debug | {'route': 'clarification'}, _memory_with_channel(memory, question, channel='meta', answer=answer)

        if is_policy_query(question):
            policy = classify_policy_query(question)
            policy_code = str(policy.get('policy_code', '') or '')
            answer = str(policy.get('message', '') or policy_query_reason(question))
            policy_review = lookup_policy_review(policy_code)
            if policy_review:
                record_review_usage(policy_review, question)
            if policy_review and str(policy_review.get('approved_answer', '') or '').strip():
                answer = str(policy_review.get('approved_answer', '') or '').strip()
            debug['route'] = 'policy_block'
            debug['policy_code'] = policy_code
            if policy_review:
                debug['policy_review_match'] = {
                    'turn_id': policy_review.get('turn_id', ''),
                    'policy_code': policy_review.get('policy_code', ''),
                }
            return answer, debug, _memory_with_channel(memory, question, channel='policy', answer=answer)

        if is_sensitive_query(question):
            policy = classify_sensitive_query(question)
            policy_code = str(policy.get('policy_code', '') or '')
            answer = str(policy.get('message', '') or sensitive_query_reason(question))
            policy_review = lookup_policy_review(policy_code)
            if policy_review:
                record_review_usage(policy_review, question)
            if policy_review and str(policy_review.get('approved_answer', '') or '').strip():
                answer = str(policy_review.get('approved_answer', '') or '').strip()
            debug['route'] = 'sensitive_block'
            debug['policy_code'] = policy_code
            if policy_review:
                debug['policy_review_match'] = {
                    'turn_id': policy_review.get('turn_id', ''),
                    'policy_code': policy_review.get('policy_code', ''),
                }
            return answer, debug, _memory_with_channel(memory, question, channel='policy', answer=answer)

        review_match = lookup_review(question)
        if review_match:
            debug['review_match'] = {
                'turn_id': review_match.get('turn_id', ''),
                'verdict': review_match.get('verdict', ''),
                'match_score': review_match.get('match_score', 0.0),
            }
            approved_answer = str(review_match.get('approved_answer', '') or '').strip()
            if approved_answer and float(review_match.get('match_score', 0.0) or 0.0) >= 10.0:
                debug['route'] = 'admin_review_override'
                record_review_usage(review_match, question)
                return approved_answer, debug, _memory_with_channel(memory, question, channel='review', answer=approved_answer)

        if is_out_of_scope_query(question, memory):
            answer = out_of_scope_reply()
            return answer, debug | {'route': 'out_of_scope'}, _memory_with_channel(memory, question, channel='meta', answer=answer)

        if is_web_notice_query(question):
            answer, web_debug = _answer_from_uneti_web(question, self.web)
            debug.update(web_debug)
            if answer:
                return answer, debug, _memory_with_channel(memory, question, channel='web', answer=answer)
            return (
                'Tôi chưa lấy được thông báo mới từ uneti.edu.vn ở thời điểm này. '
                'Vui lòng thử lại sau hoặc truy cập trực tiếp website UNETI để xem thông báo mới nhất.'
            ), debug | {'route': 'web_notice_only'}, _memory_with_channel(memory, question, channel='web')

        rule_domain = detect_domain(question)
        rule_qtype = detect_question_type(question)
        inferred_topic = detect_portal_topic(question) if rule_domain == 'portal_howto' or 'lich' in norm_text_ascii(question) else ''
        plan = {
            'intent': rule_qtype,
            'domain': rule_domain or (str(review_match.get('domain_hint', '') or '') if float(review_match.get('match_score', 0.0) or 0.0) >= 0.75 else '') or 'general_docs',
            'use_history': False,
            'retrieval_query': _review_augmented_query(question, review_match),
            'top_k': 6 if rule_qtype in {'summary', 'list', 'howto', 'compare'} else 4,
            'answer_style': 'structured' if rule_qtype in {'summary', 'list', 'howto', 'compare'} else 'direct',
            'task_for_llm2': 'Trả lời trực tiếp, đúng trọng tâm, chỉ dựa trên evidence. Thiếu evidence thì nói thiếu.',
            'topic': inferred_topic or memory.get('last_topic', ''),
        }

        if self.llm:
            try:
                llm_plan = self.llm.analyze(question, memory, rule_domain, rule_qtype)
            except Exception as e:
                llm_plan = {}
                debug['llm_analyze_error'] = str(e)
            if llm_plan:
                plan.update({k: v for k, v in llm_plan.items() if v not in [None, '']})

        plan = _sanitize_plan(plan, question, memory, rule_domain, rule_qtype)
        debug['plan'] = plan

        domain = plan.get('domain', 'general_docs')
        domain_data = self.seed.get(domain, {})
        seed_items = (
            self.store.seed_retrieve(plan.get('retrieval_query', question), domain, top_k=max(3, int(plan.get('top_k', 4))))
            if domain in self.seed
            else []
        )
        doc_tiers = _domain_doc_tiers(domain, question, plan, self.local_docs())
        chunk_docs = [doc_id for tier in doc_tiers for doc_id in tier]
        chunk_items, tier_debug = self._tiered_chunk_retrieve(
            plan.get('retrieval_query', question),
            doc_tiers,
            top_k=max(5, int(plan.get('top_k', 4)) + 2),
        )
        merged = _merge_items(seed_items, chunk_items, top_k=max(4, int(plan.get('top_k', 4))))
        contexts = [
            {
                'kind': x.kind,
                'text': x.text,
                'title': x.title,
                'source_id': x.source_id,
                'metadata': x.metadata,
                'score': x.score,
            }
            for x in merged
        ]
        debug['doc_tiers'] = tier_debug
        debug['contexts'] = contexts
        debug['keyword_dictionary'] = _context_keyword_dictionary(contexts)

        answer = ''
        rule_first_answer = _rule_answer(
            question,
            contexts,
            plan,
            domain_data,
            self.store.doc_chunks,
        )
        if rule_first_answer and _can_skip_llm_for_rule_answer(rule_first_answer, contexts, plan):
            answer = rule_first_answer
            debug['route'] = 'rule_first'
        elif self.llm:
            try:
                answer = self.llm.answer(question, plan, _compact_contexts_for_llm(contexts, plan))
            except Exception as e:
                debug['llm_answer_error'] = str(e)
                answer = ''

        if not answer or not answer_guard(answer, contexts):
            answer = rule_first_answer
            if not answer and _has_strong_evidence(contexts):
                answer = _fallback_answer(question, contexts, plan)
            if self.llm:
                debug['guard_fallback'] = True

        if not answer:
            web_doc_ids = self.web_docs()
            if web_doc_ids:
                self._ensure_docs_loaded(web_doc_ids)
                web_items = self.store.chunk_retrieve(question, web_doc_ids, top_k=5)
                web_contexts = [
                    {
                        'kind': x.kind,
                        'text': x.text,
                        'title': x.title,
                        'source_id': x.source_id,
                        'metadata': x.metadata,
                        'score': x.score,
                    }
                    for x in web_items
                ]
                debug['cached_web_contexts'] = web_contexts
                if _has_strong_evidence(web_contexts):
                    answer = _fallback_answer(question, web_contexts, {'intent': 'summary'})

        if not answer:
            answer, web_debug = _answer_from_uneti_web(question, self.web)
            debug.update({f'web_{k}': v for k, v in web_debug.items()})

        if not answer:
            answer = (
                'Không tìm thấy đủ thông tin trong tài liệu đã cung cấp. '
                'Nếu bạn cần thông báo hoặc tin mới, tôi sẽ ưu tiên tra trên uneti.edu.vn.'
            )
        if answer and chunk_docs:
            verified, verify_debug = _verify_answer_with_retrieval(
                question,
                answer,
                plan,
                self.store,
                chunk_docs,
                domain_data,
                self.store.doc_chunks,
            )
            if verify_debug:
                debug['answer_verification'] = verify_debug
            if verified and verified != answer:
                debug['answer_verification_applied'] = True
                answer = verified
        verified_answer = _post_verify_answer(
            question,
            answer,
            plan,
            contexts,
            domain_data,
            self.store.doc_chunks,
        )
        if verified_answer and verified_answer != answer:
            debug['post_verified'] = True
            answer = verified_answer
        limited_answer, limited_detail = _maybe_compose_limited_detail_answer(
            question,
            answer,
            plan,
            contexts,
            domain_data,
            self.store.doc_chunks,
        )
        if limited_answer != answer:
            debug['route'] = 'partial_scope'
            debug['partial_detail'] = limited_detail
            debug['partial_core_question'] = _base_role_question(question, plan)
            answer = limited_answer

        new_memory = {
            'active_domain': domain,
            'active_doc': chunk_docs[0] if chunk_docs else domain,
            'last_entity': _infer_entity_from_answer(domain, question, answer, contexts, domain_data),
            'last_named_unit': _infer_named_unit(domain, question, plan, self.store.doc_chunks, contexts),
            'last_question_type': plan.get('intent', rule_qtype),
            'last_retrieved_ids': [c['source_id'] for c in contexts[:6]],
            'last_user_query': question,
            'last_assistant_answer': answer,
            'last_topic': plan.get('topic', ''),
            'last_channel': 'docs' if answer and not debug.get('web_used') else 'web',
            'context_turns': _next_context_turns(question, plan, memory),
        }
        return answer, debug, new_memory


def _load_parsed_json(path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding='utf-8'))
    blocks = data.get('blocks') or []
    if not isinstance(blocks, list):
        blocks = []
    return {
        'doc_id': data.get('doc_id') or slugify(data.get('title') or path.stem),
        'title': data.get('title') or path.stem,
        'blocks': blocks,
        'source_path': str(path),
        'metadata': data.get('metadata') or {},
    }


def _empty_memory(question: str) -> Dict[str, Any]:
    return {
        'active_domain': '',
        'active_doc': '',
        'last_entity': '',
        'last_named_unit': '',
        'last_question_type': 'factoid',
        'last_retrieved_ids': [],
        'last_user_query': question,
        'last_assistant_answer': '',
        'last_topic': '',
        'last_channel': 'docs',
        'context_turns': 0,
    }


def _memory_with_channel(memory: Dict[str, Any], question: str, channel: str, answer: str = '') -> Dict[str, Any]:
    new_memory = dict(memory or {})
    turns_left = max(0, int(new_memory.get('context_turns', 0) or 0) - 1)
    new_memory.update({'last_user_query': question, 'last_channel': channel, 'context_turns': turns_left})
    if answer:
        new_memory['last_assistant_answer'] = answer
    return new_memory


def _review_augmented_query(question: str, review_match: Dict[str, Any]) -> str:
    if float(review_match.get('match_score', 0.0) or 0.0) < 0.75:
        return question
    hint = str(review_match.get('retrieval_hint', '') or '').strip()
    if not hint:
        return question
    return f'{question} {hint}'.strip()


def _next_context_turns(question: str, plan: Dict[str, Any], memory: Dict[str, Any]) -> int:
    if bool(plan.get('use_history')):
        return max(0, int(memory.get('context_turns', 0) or 0) - 1)
    q = norm_text_ascii(question)
    if any(tok in q for tok in ['khoa ', 'phong ', 'hieu truong', 'pho hieu truong', 'ban giam hieu', 'co so', 'cong sinh vien']):
        return 2
    if str(plan.get('domain', '') or '') in {'khoa_chuyen_mon', 'phong_ban_va_chuc_nang', 'ban_giam_hieu', 'co_so_vat_chat', 'portal_howto'}:
        return 1
    return 0


def _sanitize_plan(plan: Dict[str, Any], question: str, memory: Dict[str, Any], rule_domain: str | None, rule_qtype: str) -> Dict[str, Any]:
    use_history = bool(plan.get('use_history')) or should_use_history(question, memory)
    domain = plan.get('domain') or rule_domain or (memory.get('active_domain') if use_history else '') or 'general_docs'
    if rule_domain:
        domain = rule_domain

    intent = plan.get('intent') or rule_qtype or 'factoid'
    if rule_qtype in {'contact', 'howto', 'list', 'summary'}:
        intent = rule_qtype

    try:
        top_k = int(plan.get('top_k', 4))
    except Exception:
        top_k = 4
    top_k = max(3, min(8, top_k))

    style = str(plan.get('answer_style', '') or '').strip().lower()
    if style == 'concise':
        style = 'direct'
    if style not in {'direct', 'structured', 'detailed'}:
        style = 'structured' if intent in {'summary', 'list', 'howto', 'compare'} else 'direct'

    retrieval_query = expand_query(str(plan.get('retrieval_query', '') or question), domain=domain, memory=memory)
    topic = ''
    if domain == 'portal_howto':
        topic = detect_portal_topic(question) or str(plan.get('topic', '') or '')

    return {
        'intent': intent,
        'domain': domain,
        'use_history': use_history,
        'retrieval_query': retrieval_query,
        'top_k': top_k,
        'answer_style': style,
        'task_for_llm2': plan.get('task_for_llm2') or 'Trả lời trực tiếp, đúng trọng tâm, chỉ dựa trên evidence.',
        'topic': topic,
    }


def _domain_doc_tiers(domain: str, question: str, plan: Dict[str, Any], local_docs: List[str]) -> List[List[str]]:
    q = norm_text_ascii(question)
    intent = str(plan.get('intent', '') or '')
    if domain == 'ban_giam_hieu':
        return [['ban-giam-hieu'], ['danh-sach-cac-thanh-vien-hoi-dong-truong']]
    if domain == 'hoi_dong_truong':
        return [['danh-sach-cac-thanh-vien-hoi-dong-truong'], ['ban-giam-hieu']]
    if domain == 'co_so_vat_chat':
        return [['co-so-vat-chat']]
    if domain == 'lich_su_hinh_thanh':
        return [['lich-su-hinh-thanh']]
    if domain == 'phong_ban_va_chuc_nang':
        return [['phong-ban-va-chuc-nang'], DOC_HIERARCHY['overview']]
    if domain == 'khoa_chuyen_mon':
        return [['khoa-chuyen-mon'], DOC_HIERARCHY['overview']]
    if domain == 'portal_howto':
        return [['huong-dan-chuc-nang-cong-thong-tin-sv']]
    if domain == 'general_docs':
        if any(tok in q for tok in ['phong ', 'khoa ', 'truong khoa', 'truong phong', 'email', 'website', 'dia chi', 'o dau', 'van phong']):
            return [DOC_HIERARCHY['organization'], DOC_HIERARCHY['overview']]
        if intent in {'summary', 'list'} or any(tok in q for tok in ['truong co', 'uneti', 'lich su', 'co so', 'ban giam hieu', 'hoi dong truong']):
            return [DOC_HIERARCHY['overview'], DOC_HIERARCHY['organization']]
        return [DOC_HIERARCHY['organization'], DOC_HIERARCHY['overview'], [doc for doc in local_docs if doc not in set(DOC_HIERARCHY['overview'] + DOC_HIERARCHY['organization'])]]
    return [local_docs]


def _merge_items(seed_items, chunk_items, top_k: int):
    merged = sorted(seed_items + chunk_items, key=lambda x: x.score, reverse=True)
    out = []
    seen = set()
    for item in merged:
        metadata = getattr(item, 'metadata', {}) or {}
        dedupe_key = str(metadata.get('dedupe_key', '') or '').strip()
        key = ('dedupe', dedupe_key) if dedupe_key else (item.source_id, item.text.strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


def _has_strong_evidence(contexts: List[Dict[str, Any]]) -> bool:
    if not contexts:
        return False
    top_score = float(contexts[0].get('score', 0.0) or 0.0)
    return top_score >= 0.32


def _is_sufficient_tier_evidence(items: List[Any]) -> bool:
    if not items:
        return False
    top_score = float(items[0].score or 0.0)
    if top_score >= 0.72:
        return True
    if len(items) >= 2 and top_score >= 0.56 and float(items[1].score or 0.0) >= 0.46:
        return True
    return False


def _format_record(meta: Dict[str, Any]) -> str:
    parts = []
    if meta.get('name'):
        parts.append(str(meta['name']))
    if meta.get('role'):
        parts.append(str(meta['role']))
    if meta.get('email'):
        parts.append(f"Email: {meta['email']}")
    if meta.get('address'):
        parts.append(str(meta['address']))
    if meta.get('city'):
        parts.append(f"Thành phố: {meta['city']}")
    return ' | '.join(parts)


def _context_text(ctx: Dict[str, Any]) -> str:
    meta = ctx.get('metadata') or {}
    structured = _format_record(meta)
    return structured or ctx.get('text', '')


def _context_keyword_dictionary(contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for ctx in contexts:
        meta = ctx.get('metadata') or {}
        keyword_ids = meta.get('keyword_ids') or []
        keywords = meta.get('keywords') or []
        weights = meta.get('keyword_weights') or {}
        for keyword_id, keyword in zip(keyword_ids, keywords):
            row = rows.setdefault(str(keyword_id), {
                'id': str(keyword_id),
                'term': str(keyword),
                'max_weight': 0.0,
                'hits': 0,
            })
            row['hits'] += 1
            row['max_weight'] = max(float(row['max_weight']), float(weights.get(keyword, 0.0)))
    return sorted(rows.values(), key=lambda item: (item['max_weight'], item['hits'], item['term']), reverse=True)[:24]


def _can_skip_llm_for_rule_answer(answer: str, contexts: List[Dict[str, Any]], plan: Dict[str, Any]) -> bool:
    if not answer or not contexts:
        return False
    intent = str(plan.get('intent', '') or 'factoid')
    if intent not in {'factoid', 'contact', 'howto', 'list'}:
        return False
    if len(answer) > 1400:
        return False
    if answer_guard(answer, contexts):
        return True
    domain = str(plan.get('domain', '') or '')
    return domain in {'ban_giam_hieu', 'co_so_vat_chat', 'portal_howto', 'khoa_chuyen_mon', 'phong_ban_va_chuc_nang'} and _has_strong_evidence(contexts)


def _compact_contexts_for_llm(contexts: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not contexts:
        return []
    top_k = max(3, min(5, int(plan.get('top_k', 4) or 4)))
    compacted: List[Dict[str, Any]] = []
    for idx, ctx in enumerate(contexts[:top_k]):
        item = dict(ctx)
        meta = dict(item.get('metadata') or {})
        text = str(item.get('text', '') or '')
        limit = 520 if idx == 0 else 300
        if len(text) > limit:
            summary = str(meta.get('summary', '') or '').strip()
            text = (summary + '\n' if summary else '') + text[:limit].rstrip() + '...'
        item['text'] = text
        keep_meta = {
            key: meta.get(key)
            for key in [
                'doc_id', 'section', 'page', 'chunk_type', 'entity_type', 'entity_name',
                'summary', 'source_kind', 'source_url', 'topic_tags', 'intent_tags',
                'emails', 'urls', 'dates', 'keywords',
            ]
            if meta.get(key) not in [None, '', []]
        }
        item['metadata'] = keep_meta
        compacted.append(item)
    return compacted


def _answer_leadership(question: str, domain_data: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    records = domain_data.get('records', [])
    if not records:
        return ''

    if 'pho hieu truong' in q or 'hieu pho' in q:
        for rec in records:
            role = norm_text_ascii(rec.get('role', ''))
            if 'pho hieu truong' in role:
                return _format_record(rec)

    if 'hieu truong' in q or 'nguoi dung dau' in q or 'lanh dao cao nhat' in q:
        for rec in records:
            role = norm_text_ascii(rec.get('role', ''))
            if 'hieu truong' in role and 'pho hieu truong' not in role:
                return _format_record(rec)

    if 'ban giam hieu' in q or 'lanh dao' in q or 'gom nhung ai' in q or 'danh sach' in q:
        return '\n'.join(f"- {_format_record(rec)}" for rec in records)

    return _format_record(records[0])


def _answer_campus(question: str, domain_data: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    locations = domain_data.get('locations', [])
    if not locations:
        return ''

    filters = detect_campus_filter(question)
    city = filters.get('city')
    address = filters.get('address')

    filtered = locations
    if city == 'ha noi':
        filtered = [loc for loc in locations if 'ha noi' in norm_text_ascii(loc.get('city', ''))]
    elif city == 'nam dinh':
        filtered = [loc for loc in locations if 'nam dinh' in norm_text_ascii(loc.get('city', ''))]

    if address:
        filtered = [loc for loc in filtered if address in norm_text_ascii(loc.get('address', ''))]

    if address and filtered:
        return filtered[0]['address']

    if city and filtered:
        city_label = 'Hà Nội' if city == 'ha noi' else 'Nam Định'
        lines = [f"- {loc['address']}" for loc in filtered]
        return f"Cơ sở {city_label} gồm {len(filtered)} địa điểm:\n" + '\n'.join(lines)

    if any(x in q for x in ['cac co so', 'dia diem dao tao', 'co nhung co so nao', 'bao nhieu co so', 'cac dia diem']):
        hanoi = [loc for loc in locations if 'ha noi' in norm_text_ascii(loc.get('city', ''))]
        namdinh = [loc for loc in locations if 'nam dinh' in norm_text_ascii(loc.get('city', ''))]
        lines = ['Trường có 4 địa điểm đào tạo:']
        if hanoi:
            lines.append('- Hà Nội: ' + '; '.join(loc['address'] for loc in hanoi))
        if namdinh:
            lines.append('- Nam Định: ' + '; '.join(loc['address'] for loc in namdinh))
        return '\n'.join(lines)

    return ''


def _is_school_overview_query(question: str) -> bool:
    q = norm_text_ascii(question)
    return any(
        hint in q for hint in [
            'gioi thieu ve truong', 'gioi thieu ve uneti', 'noi ve truong', 'noi ve uneti',
            'thong tin ve truong', 'tong quan ve truong', 'truong uneti la truong nao',
        ]
    )


def _answer_school_overview(seed: Dict[str, Any]) -> str:
    history = seed.get('lich_su_hinh_thanh', {})
    campus = seed.get('co_so_vat_chat', {})
    facts = list(history.get('facts', []) or [])
    locations = list(campus.get('locations', []) or [])
    if not facts:
        return ''

    intro = facts[0]
    if 'UNETI' not in intro:
        intro = 'UNETI: ' + intro
    extra_parts: List[str] = []
    if len(facts) > 1:
        extra_parts.append(facts[1])
    if locations:
        extra_parts.append(f'Trường hiện có {len(locations)} địa điểm đào tạo tại Hà Nội và Nam Định.')

    answer = intro
    if extra_parts:
        answer += ' ' + ' '.join(extra_parts[:2])
    return answer.strip()


def _qa_topic(qa: Dict[str, Any]) -> str:
    q = norm_text_ascii(qa.get('question', ''))
    a = norm_text_ascii(qa.get('answer', ''))
    text = f'{q} {a}'
    if 'trang chu' in text or 'sinhvien.uneti.edu.vn' in text:
        return 'homepage'
    if 'dang nhap' in text:
        return 'login'
    if 'doi mat khau' in text:
        return 'change_password'
    if 'thong tin sinh vien' in text:
        return 'student_info'
    if 'lich theo tuan' in text or 'lich hoc' in text or 'lich thi' in text:
        return 'schedule_week'
    if 'lich theo tien do' in text:
        return 'schedule_progress'
    if 'lich toan truong' in text:
        return 'schedule_global'
    if 'nhac nho' in text:
        return 'reminders'
    if 'ket qua hoc tap' in text:
        return 'study_results'
    if 'ket qua ren luyen' in text or 'diem ren luyen' in text:
        return 'conduct_results'
    if 'diem danh' in text:
        return 'attendance'
    if 'chuong trinh khung' in text:
        return 'curriculum'
    if 'huy' in text and 'dang ky hoc phan' in text:
        return 'cancel_registration'
    if 'mon hoc dieu kien' in text:
        return 'conditional_course'
    if 'thi lai' in text:
        return 'retake_registration'
    if 'phieu thu' in text:
        return 'receipt'
    if 'cong no' in text or 'hoc phi' in text:
        return 'debt_lookup'
    if 'dang ky hoc phan' in text:
        return 'course_registration'
    return 'generic'


def _answer_portal(question: str, domain_data: Dict[str, Any], plan: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    qas = domain_data.get('qa', [])
    if not qas:
        return ''

    topic = str(plan.get('topic', '') or detect_portal_topic(question) or '')
    is_howto_query = _is_howto_like_question(question, str(plan.get('intent', '') or ''))
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    for qa in qas:
        qa_q = norm_text_ascii(qa.get('question', ''))
        qa_a = norm_text_ascii(qa.get('answer', ''))
        qa_text = f'{qa_q} {qa_a}'
        score = 0.0
        qa_topic = _qa_topic(qa)
        if topic and qa_topic == topic:
            score += 3.0
        for kw in qa.get('keywords', []):
            if norm_text_ascii(kw) in q:
                score += 1.5
        for tok in q.split():
            if len(tok) >= 4 and tok in qa_text:
                score += 0.25
        if 'lich hoc' in q or 'lich thi' in q:
            if qa_topic == 'schedule_week':
                score += 2.0
            if qa_topic == 'course_registration':
                score -= 1.0
        if is_howto_query:
            score += _instruction_specificity(qa.get('answer', '')) * 0.9
            if qa_topic in {'course_registration', 'conditional_course', 'retake_registration', 'login', 'change_password'}:
                score += 0.25
        ranked.append((score, qa))

    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best = ranked[0]
    if best_score <= 0:
        return ''
    if topic == 'debt_lookup' and any(x in q for x in ['hoc phi', 'nop hoc phi', 'dong hoc phi', 'thanh toan hoc phi']):
        debt_answer = ''
        receipt_answer = ''
        for _, qa in ranked:
            qa_topic = _qa_topic(qa)
            if qa_topic == 'debt_lookup' and not debt_answer:
                debt_answer = str(qa.get('answer', '') or '').strip()
            if qa_topic == 'receipt' and not receipt_answer:
                receipt_answer = str(qa.get('answer', '') or '').strip()
            if debt_answer and receipt_answer:
                break
        if debt_answer:
            parts = [debt_answer]
            if receipt_answer:
                parts.append(receipt_answer)
            parts.append('Tôi chưa thấy hướng dẫn thanh toán học phí trực tuyến cụ thể trong tri thức hiện có.')
            return ' '.join(parts)
    return best.get('answer', '')


HOWTO_QUERY_HINTS = [
    'cach', 'nhu the nao', 'o dau', 'dang nhap', 'dang ky', 'thi lai', 'doi mat khau',
    'xem ', 'tra cuu', 'mo ', 'vao ', 'chon ', 'nhap ',
]
ACTION_ANSWER_MARKERS = [
    'dashboard ->', '->', 'chon ', 'nhan ', 'xac nhan', 'nhap ', 'vao ', 'mo ',
    'menu', 'muc ', 'hoc ky', 'dot dang ky', 'mon hoc', 'lop hoc phan', 'username',
    'mat khau', 'hien thi', 'xem chi tiet',
]
GENERIC_NOTICE_MARKERS = [
    'duoc nha truong thong bao', 'duoc thong bao', 'theo tung hoc ky', 'theo doi thong bao',
    'huong dan cu the cua tung dot', 'thong bao cua tung dot', 'tren menu thong bao',
]


def _is_howto_like_question(question: str, intent: str = '') -> bool:
    q = norm_text_ascii(question)
    if intent == 'howto':
        return True
    return any(hint in q for hint in HOWTO_QUERY_HINTS)


def _instruction_specificity(answer: str) -> float:
    text = norm_text_ascii(answer)
    if not text:
        return 0.0
    score = 0.0
    for marker in ACTION_ANSWER_MARKERS:
        if marker in text:
            score += 0.22 if marker != '->' else 0.5
    score += min(0.8, text.count('->') * 0.28)
    if re.search(r'\b(?:buoc|b1|b2|b3)\b', text):
        score += 0.4
    if text.count(',') >= 2:
        score += 0.18
    if text.count(';') >= 1:
        score += 0.12
    for marker in GENERIC_NOTICE_MARKERS:
        if marker in text:
            score -= 0.45
    return score


UNIT_MARKER_RE = re.compile(r'((?:Khoa|Phòng)\s+(?!\d)[^|\n]+?)\s*(?:\|\s*)?Posted on', re.IGNORECASE)


def _extract_unit_markers(text: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for match in UNIT_MARKER_RE.finditer(text or ''):
        name = ' '.join(match.group(1).replace('\xa0', ' ').split())
        if name:
            out.append((match.start(1), name))
    return out


def _is_contact_chunk(section: str, text: str) -> bool:
    sec = norm_text_ascii(section)
    body = norm_text_ascii(text)
    contact_markers = [
        'co so ha noi', 'co so nam dinh', 'co so ninh binh', 'dia chi:', 'website:',
        'email:', 'dien thoai:', 'sdt', 'linh nam', 'minh khai', 'tran hung dao', 'ha noi', 'nam dinh',
    ]
    if any(h in sec for h in ['dia chi lien he', 'thong tin lien he', ' lien he']):
        return any(h in body for h in contact_markers)
    return any(h in body for h in contact_markers)


def _extract_unit_contact_index(chunks: List[Any]) -> Dict[str, List[str]]:
    current_unit = ''
    contacts: Dict[str, List[str]] = {}
    for chunk in chunks:
        section = str(getattr(chunk, 'section', '') or '').strip()
        text = str(getattr(chunk, 'text', '') or '').strip()
        chunk_type = str(getattr(chunk, 'chunk_type', '') or '').strip()
        chunk_unit = str(getattr(chunk, 'entity_name', '') or '').strip()
        if chunk_unit and norm_text_ascii(chunk_unit) not in {'khoa chuyen mon', 'phong ban va chuc nang'}:
            current_unit = chunk_unit
        combined = '\n'.join(part for part in [section, text] if part)
        markers = _extract_unit_markers(combined)
        visible = combined[:markers[0][0]].strip() if markers else combined
        if markers:
            first_unit = markers[0][1]
            if visible and _is_contact_chunk(section, visible):
                contacts.setdefault(first_unit, []).append(visible)
        elif current_unit and visible and _is_contact_chunk(section, visible):
            contacts.setdefault(current_unit, []).append(visible)
        if markers:
            current_unit = markers[-1][1]
    return contacts


def _extract_unit_profiles(chunks: List[Any]) -> Dict[str, List[Dict[str, str]]]:
    current_unit = ''
    profiles: Dict[str, List[Dict[str, str]]] = {}
    for chunk in chunks:
        section = str(getattr(chunk, 'section', '') or '').strip()
        text = str(getattr(chunk, 'text', '') or '').strip()
        chunk_type = str(getattr(chunk, 'chunk_type', '') or '').strip()
        chunk_unit = str(getattr(chunk, 'entity_name', '') or '').strip()
        if chunk_unit and norm_text_ascii(chunk_unit) not in {'khoa chuyen mon', 'phong ban va chuc nang'}:
            current_unit = chunk_unit
        combined = '\n'.join(part for part in [section, text] if part)
        markers = _extract_unit_markers(combined)
        visible = combined[:markers[0][0]].strip() if markers else combined
        sec = norm_text_ascii(section)
        is_contact_section = any(h in sec for h in ['dia chi lien he', 'thong tin lien he', ' lien he'])
        profile_units = [current_unit] if current_unit else []
        body_norm = norm_text_ascii(visible)
        if 'khoa quan tri' in body_norm and 'marketing' in body_norm:
            profile_units.append('Khoa Quan tri & Marketing')
        for profile_unit in dict.fromkeys(profile_units):
            if profile_unit and visible and not is_contact_section:
                profiles.setdefault(profile_unit, []).append({'section': section, 'text': visible, 'chunk_type': chunk_type})
        if markers:
            current_unit = markers[-1][1]
    return profiles


UNIT_MATCH_STOPWORDS = {
    'khoa', 'phong', 'ban', 'bo', 'mon', 'trung', 'tam', 'va', 'voi', 'cua',
    'o', 'dau', 'dia', 'chi', 'email', 'website', 'web', 'link', 'so', 'dien', 'thoai', 'sdt',
    'truong', 'pho', 'la', 'gi', 'ai', 'nhu', 'the', 'nao', 'chuc', 'nang', 'nhiem', 'vu',
    'lich', 'su', 'thanh', 'tich', 'quy', 'mo', 'co', 'cau', 'to', 'chuc',
    'bao', 'nhieu', 'liet', 'ke', 'gom', 'nhung', 'cac', 'day', 'giang', 'hoc', 'phan',
    'nganh',
}


def _unit_match_tokens(text: str) -> List[str]:
    return [
        tok for tok in norm_text_ascii(text).split()
        if len(tok) >= 2 and tok not in UNIT_MATCH_STOPWORDS
    ]


def _unit_match_phrases(tokens: List[str]) -> set[str]:
    phrases: set[str] = set()
    max_n = min(4, len(tokens))
    for n in range(2, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            phrases.add(' '.join(tokens[i:i + n]))
    return phrases


def _extract_unit_query_fragment(question: str) -> str:
    q = norm_text_ascii(question)
    patterns = [
        r'\b(?:truong|pho truong)\s+(?:khoa|phong)\s+([a-z0-9][a-z0-9\s&.-]*)',
        r'\b(?:khoa|phong)\s+([a-z0-9][a-z0-9\s&.-]*)',
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        fragment = match.group(1)
        fragment = re.split(
            r'\b(?:o dau|dia chi|email|website|web|link|so dien thoai|sdt|chuc nang|nhiem vu|lich su|thanh tich|quy mo|co cau|to chuc|day mon gi|day mon|giang day|hoc phan|dao tao gi|dao tao|nganh nao|nganh gi|la gi|la ai|nhu the nao)\b',
            fragment,
            maxsplit=1,
        )[0].strip()
        if fragment:
            return fragment
    return ''


def _match_named_unit(question: str, unit_names: List[str]) -> str:
    q = norm_text_ascii(question)
    query_fragment = _extract_unit_query_fragment(question)
    query_basis = query_fragment or q
    query_token_list = _unit_match_tokens(query_basis)
    q_tokens = set(query_token_list)
    q_phrases = _unit_match_phrases(query_token_list) if query_token_list else set()
    best_name = ''
    best_score = 0.0
    for name in unit_names:
        norm_name = norm_text_ascii(name)
        short_name = re.sub(r'^(khoa|phong)\s+', '', norm_name).strip()
        name_token_list = _unit_match_tokens(short_name)
        name_tokens = set(name_token_list)
        name_phrases = _unit_match_phrases(name_token_list) if name_token_list else set()
        score = 0.0
        if norm_name and norm_name in q:
            score += 4.0 + len(norm_name) / 100.0
        if short_name and short_name in q:
            score += 3.0 + len(short_name) / 100.0
        if query_fragment and query_fragment in short_name:
            score += 3.5 + len(query_fragment) / 100.0
        if name_tokens:
            token_hits = len(q_tokens & name_tokens)
            score += token_hits / len(name_tokens)
            score += token_hits * 0.6
            if q_tokens and q_tokens.issubset(name_tokens):
                score += 2.2
        phrase_hits = len(q_phrases & name_phrases)
        if phrase_hits:
            score += phrase_hits * 1.4
        enough_overlap = bool(name_tokens) and (
            len(q_tokens & name_tokens) >= max(1, min(2, len(q_tokens)))
            or phrase_hits >= 1
            or (q_tokens and q_tokens.issubset(name_tokens))
        )
        if score > best_score and (score >= 3.0 or enough_overlap):
            best_name = name
            best_score = score
    return best_name


def _split_block_lines(text: str) -> List[str]:
    cleaned = re.sub(r'\bPosted on\b.*$', '', text or '', flags=re.IGNORECASE | re.DOTALL)
    lines = []
    for piece in re.split(r'(?:\s*\|\s*|\r?\n)+', cleaned):
        line = ' '.join(piece.replace('\xa0', ' ').split()).strip(' -')
        if line:
            lines.append(line)
    return lines


def _unit_query_topic(question: str) -> str:
    q = norm_text_ascii(question)
    if any(tok in q for tok in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa', 'lanh dao', 'ban chu nhiem']):
        return 'leadership'
    if any(tok in q for tok in ['email', 'mail']):
        return 'email'
    if any(tok in q for tok in ['website', 'web', 'link']):
        return 'website'
    if any(tok in q for tok in ['dien thoai', 'so dien thoai', 'sdt']):
        return 'phone'
    if any(tok in q for tok in ['dia chi', 'o dau', 'van phong']):
        return 'contact'
    if any(tok in q for tok in ['day mon gi', 'day mon', 'giang day', 'hoc phan nao', 'hoc phan gi', 'mon gi', 'mon nao']):
        return 'teaching'
    if any(tok in q for tok in ['dao tao gi', 'dao tao nganh', 'nganh nao', 'nganh gi', 'chuong trinh dao tao']):
        return 'training'
    if 'chuc nang' in q:
        return 'function'
    if 'nhiem vu' in q:
        return 'duty'
    if any(tok in q for tok in ['lich su', 'qua trinh hinh thanh', 'thanh lap']):
        return 'history'
    if any(tok in q for tok in ['thanh tich', 'ket qua dat duoc']):
        return 'achievement'
    if any(tok in q for tok in ['quy mo', 'nang luc']):
        return 'capacity'
    if 'dinh huong' in q:
        return 'development'
    if any(tok in q for tok in ['co cau', 'to chuc']):
        return 'structure'
    return 'summary'


def _section_topic(section: str, chunk_type: str = '') -> str:
    ctype = norm_text_ascii(chunk_type)
    if ctype in {'teaching', 'training'}:
        return ctype
    sec = norm_text_ascii(section)
    if any(tok in sec for tok in ['dia chi lien he', 'thong tin lien he', ' lien he']):
        return 'contact'
    if 'co cau' in sec:
        return 'structure'
    if 'chuc nang' in sec and 'nhiem vu' in sec:
        return 'function_duty'
    if 'chuc nang' in sec:
        return 'function'
    if 'nhiem vu' in sec:
        return 'duty'
    if any(tok in sec for tok in ['lich su', 'qua trinh hinh thanh', 'thong tin chung']):
        return 'history'
    if any(tok in sec for tok in ['thanh tich', 'ket qua dat duoc']):
        return 'achievement'
    if any(tok in sec for tok in ['quy mo', 'nang luc']):
        return 'capacity'
    if 'dinh huong' in sec:
        return 'development'
    if any(tok in sec for tok in ['giang day', 'hoc phan']):
        return 'teaching'
    if any(tok in sec for tok in ['chuong trinh dao tao', 'dao tao', 'vi tri viec lam']):
        return 'training'
    return 'generic'


def _find_unit_entries(entries: List[Dict[str, str]], topics: List[str]) -> List[Dict[str, str]]:
    matched = []
    for entry in entries:
        topic = _section_topic(entry.get('section', ''), entry.get('chunk_type', ''))
        if topic in topics:
            matched.append(entry)
    return matched


def _split_contact_lines(text: str) -> List[str]:
    cleaned = re.sub(r'\bPosted on\b.*$', '', text or '', flags=re.IGNORECASE | re.DOTALL)
    lines = []
    for piece in re.split(r'(?:\s*\|\s*|\r?\n)+', cleaned):
        line = ' '.join(piece.replace('\xa0', ' ').split()).strip(' -')
        line = re.sub(r'^\s*\d+(?:\.\d+)*\.?\s*(?:Địa chỉ liên hệ|Liên hệ|Thông tin liên hệ):?\s*', '', line, flags=re.IGNORECASE)
        norm = norm_text_ascii(line)
        if not line or norm in {
            '7. lien he', '7. dia chi lien he', '6. thong tin lien he', '8. lien he',
            'co so ha noi:', 'co so nam dinh:', 'co so ninh binh:', 'co so ha noi', 'co so nam dinh', 'co so ninh binh',
        }:
            continue
        lines.append(line)
    return lines


def _infer_contact_label(line: str) -> str:
    q = norm_text_ascii(line)
    if any(tok in q for tok in ['ha noi', 'linh nam', 'minh khai']):
        return 'Cơ sở Hà Nội'
    if any(tok in q for tok in ['nam dinh', 'tran hung dao', 'my xa', 'ninh binh']):
        return 'Cơ sở Nam Định'
    return ''


def _format_unit_contact_answer(unit_name: str, blobs: List[str], question: str) -> str:
    q = norm_text_ascii(question)
    wants_address = any(tok in q for tok in ['dia chi', 'o dau', 'van phong'])
    wants_email = any(tok in q for tok in ['email', 'mail'])
    wants_website = any(tok in q for tok in ['website', 'web', 'link'])
    wants_phone = any(tok in q for tok in ['dien thoai', 'so dien thoai', 'sdt'])

    lines: List[str] = []
    seen = set()
    for blob in blobs:
        for raw in _split_contact_lines(blob):
            norm = norm_text_ascii(raw)
            keep = False
            if wants_email:
                keep = 'email:' in norm
            elif wants_website:
                keep = 'website' in norm
            elif wants_phone:
                keep = 'dien thoai' in norm or 'sdt' in norm
            else:
                keep = wants_address or any(tok in norm for tok in ['co so ', 'dia chi:', 'linh nam', 'minh khai', 'tran hung dao', 'nam dinh', 'ha noi'])
            if not keep:
                continue

            normalized_line = raw
            if wants_address or not (wants_email or wants_website or wants_phone):
                label = _infer_contact_label(raw)
                content = re.sub(r'^(?:Địa chỉ:|Dia chi:)\s*', '', normalized_line, flags=re.IGNORECASE).strip()
                if label and not norm.startswith('co so '):
                    normalized_line = f'{label}: {content}'
                else:
                    normalized_line = content if norm.startswith('dia chi:') else normalized_line
            key = norm_text_ascii(normalized_line)
            if key and key not in seen:
                seen.add(key)
                lines.append(normalized_line)

    if not lines:
        return ''
    if len(lines) == 1:
        return lines[0]
    return f'{unit_name}:\n' + '\n'.join(f'- {line}' for line in lines)


def _answer_unit_leadership(question: str, entries: List[Dict[str, str]]) -> str:
    q = norm_text_ascii(question)
    wanted: List[str] = []
    if 'pho truong khoa' in q:
        wanted = ['pho truong khoa']
    elif 'truong khoa' in q:
        wanted = ['truong khoa']
    elif 'pho truong phong' in q:
        wanted = ['pho truong phong']
    elif 'truong phong' in q:
        wanted = ['truong phong', 'phu trach phong']
    elif 'tro ly khoa' in q:
        wanted = ['tro ly khoa']

    lines: List[str] = []
    seen = set()
    for entry in entries:
        split_lines = _split_block_lines(entry.get('text', ''))
        for line_index, line in enumerate(split_lines):
            norm = norm_text_ascii(line)
            if wanted:
                if not any(tag in norm for tag in wanted):
                    continue
                if any(tag == 'truong khoa' for tag in wanted) and 'pho truong khoa' in norm and 'phu trach' not in norm:
                    continue
                if any(tag == 'truong phong' for tag in wanted) and 'pho truong phong' in norm and 'phu trach phong' not in norm:
                    continue
            else:
                if not any(tag in norm for tag in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa']):
                    continue
            if 'email' not in norm:
                for nearby in split_lines[line_index + 1:line_index + 4]:
                    nearby_norm = norm_text_ascii(nearby)
                    if 'email' in nearby_norm:
                        line = f'{line} {nearby}'
                        norm = norm_text_ascii(line)
                        break
            if norm not in seen:
                seen.add(norm)
                lines.append(line)
    if not lines:
        return ''
    if len(lines) == 1:
        return lines[0]
    return '\n'.join(f'- {line}' for line in lines[:6])


def _answer_unit_comm_field(question: str, entries: List[Dict[str, str]], field: str) -> str:
    q = norm_text_ascii(question)
    lines: List[str] = []
    seen = set()
    role_tags = [tag for tag in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa'] if tag in q]
    for entry in entries:
        split_lines = _split_block_lines(entry.get('text', ''))
        if role_tags:
            for i, line in enumerate(split_lines):
                norm = norm_text_ascii(line)
                if not any(tag in norm for tag in role_tags):
                    continue
                if 'truong khoa' in role_tags and 'pho truong khoa' in norm and 'phu trach' not in norm:
                    continue
                if 'truong phong' in role_tags and 'pho truong phong' in norm and 'phu trach phong' not in norm:
                    continue
                for nearby in split_lines[i + 1:i + 4]:
                    nearby_norm = norm_text_ascii(nearby)
                    if field == 'email' and 'email' in nearby_norm:
                        return nearby
                    if field == 'website' and ('website' in nearby_norm or 'http' in nearby_norm):
                        return nearby
                    if field == 'phone' and ('dien thoai' in nearby_norm or 'sdt' in nearby_norm):
                        return nearby
        for line in split_lines:
            norm = norm_text_ascii(line)
            if field == 'email' and 'email' not in norm:
                continue
            if field == 'website' and 'website' not in norm and 'http' not in norm:
                continue
            if field == 'phone' and 'dien thoai' not in norm and 'sdt' not in norm:
                continue
            if norm not in seen:
                seen.add(norm)
                lines.append(line)
    if not lines:
        return ''
    if len(lines) == 1:
        return lines[0]
    return '\n'.join(f'- {line}' for line in lines[:6])


def _format_section_answer(entries: List[Dict[str, str]], max_items: int = 2) -> str:
    texts = []
    seen = set()
    for entry in entries:
        text = ' '.join(entry.get('text', '').split()).strip()
        norm = norm_text_ascii(text)
        if text and norm not in seen:
            seen.add(norm)
            texts.append(text)
        if len(texts) >= max_items:
            break
    if not texts:
        return ''
    if len(texts) == 1:
        return texts[0]
    return texts[0] + '\n' + '\n'.join(f'- {text}' for text in texts[1:])


def _answer_unit_teaching(unit_name: str, entries: List[Dict[str, str]]) -> str:
    for entry in entries:
        text = ' '.join(entry.get('text', '').split()).strip()
        if not text:
            continue
        match = re.search(
            r'(?:giảng dạy|giang day).*?(?:các học phần|cac hoc phan)\s+(?:về|ve)\s+([^.;]+)',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            subjects = match.group(1).strip(' .,:;')
            if subjects:
                return f'{unit_name} đảm nhiệm giảng dạy các học phần: {subjects}.'
        if any(marker in norm_text_ascii(text) for marker in ['giang day', 'hoc phan ve', 'cac hoc phan']):
            return text
    return ''


def _answer_unit_training(unit_name: str, entries: List[Dict[str, str]]) -> str:
    best_text = ''
    best_score = -1.0
    for entry in entries:
        text = ' '.join(entry.get('text', '').split()).strip()
        if not text:
            continue
        norm = norm_text_ascii(text)
        score = 0.0
        if any(marker in norm for marker in ['nganh dao tao', 'dao tao khdl', 'ma nganh']):
            score += 3.0
        if any(marker in norm for marker in ['chuong trinh cu nhan', 'khoa hoc du lieu', 'tin chi']):
            score += 2.0
        if 'co hoi viec lam' in norm:
            score += 0.5
        if any(marker in norm for marker in ['dieu chinh chuong trinh', 'quy che hoat dong']):
            score -= 1.5
        if score > best_score:
            best_score = score
            best_text = text
    if not best_text:
        return ''

    match = re.search(
        r'(?:mở|mo)\s+ngành\s+đào tạo\s+([^.,;]+?)(?:\s+với\s+mã ngành\s+([0-9]+))?[.,;]',
        best_text,
        flags=re.IGNORECASE,
    )
    if match:
        program = match.group(1).strip()
        code = (match.group(2) or '').strip()
        suffix = f', mã ngành {code}' if code else ''
        return f'{unit_name} có thông tin đào tạo {program}{suffix}.'
    if best_score >= 1.5:
        return best_text
    return ''


def _answer_named_unit_qa(question: str, domain_data: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    topic = _unit_query_topic(question)
    if topic not in {'contact', 'email', 'website', 'phone'}:
        return ''
    if any(tag in q for tag in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa']):
        return ''

    q_tokens = set(_unit_match_tokens(q))
    if not q_tokens:
        return ''

    best_answer = ''
    best_score = 0.0
    for qa in domain_data.get('qa', []):
        qa_q = norm_text_ascii(qa.get('question', ''))
        qa_a = norm_text_ascii(qa.get('answer', ''))
        qa_text = f'{qa_q} {qa_a}'
        qa_tokens = set(_unit_match_tokens(qa_text))
        if not qa_tokens:
            continue

        score = 0.0
        token_hits = len(q_tokens & qa_tokens)
        score += token_hits * 0.8
        if q_tokens.issubset(qa_tokens):
            score += 2.0

        for kw in qa.get('keywords', []):
            if norm_text_ascii(kw) in q:
                score += 1.5

        if topic in {'contact', 'summary'} and 'lien he' in qa_q:
            score += 1.4
        if topic == 'email' and 'email' in qa_a:
            score += 1.2
        if topic == 'website' and ('website' in qa_a or 'http' in qa_a):
            score += 1.2
        if topic == 'phone' and ('dien thoai' in qa_a or 'sdt' in qa_a):
            score += 1.2
        if topic == 'contact' and any(tok in qa_a for tok in ['dia chi', 'ha noi', 'nam dinh', 'linh nam', 'minh khai', 'tran hung dao']):
            score += 1.2

        if score > best_score and score >= 2.4:
            best_score = score
            best_answer = str(qa.get('answer', '') or '').strip()

    return best_answer


def _answer_named_unit(question: str, query_hint: str, chunks: List[Any]) -> str:
    if not chunks:
        return ''

    profiles = _extract_unit_profiles(chunks)
    contacts = _extract_unit_contact_index(chunks)
    unit_names = sorted(set(profiles.keys()) | set(contacts.keys()))
    unit_name = _match_named_unit(query_hint or question, unit_names)
    if not unit_name:
        return ''

    topic = _unit_query_topic(question)
    entries = profiles.get(unit_name, [])

    if topic in {'contact', 'email', 'website', 'phone'}:
        answer = _format_unit_contact_answer(unit_name, contacts.get(unit_name, []), question)
        if answer:
            return answer
        comm_field = {'email': 'email', 'website': 'website', 'phone': 'phone'}.get(topic)
        if comm_field:
            answer = _answer_unit_comm_field(question, entries, comm_field)
            if answer:
                return answer

    if topic == 'leadership':
        answer = _answer_unit_leadership(question, _find_unit_entries(entries, ['structure', 'generic']))
        if answer:
            return answer

    if topic == 'teaching':
        answer = _answer_unit_teaching(unit_name, _find_unit_entries(entries, ['teaching']))
        if answer:
            return answer

    if topic == 'training':
        answer = _answer_unit_training(unit_name, _find_unit_entries(entries, ['training']))
        if answer:
            return answer

    topic_map = {
        'function': ['function', 'function_duty'],
        'duty': ['duty', 'function_duty'],
        'teaching': ['teaching'],
        'training': ['training'],
        'history': ['history'],
        'achievement': ['achievement'],
        'capacity': ['capacity'],
        'development': ['development'],
        'structure': ['structure'],
        'summary': ['history', 'generic', 'structure'],
    }
    if topic in topic_map:
        answer = _format_section_answer(_find_unit_entries(entries, topic_map[topic]), max_items=2 if topic == 'summary' else 1)
        if answer:
            return answer

    if topic == 'summary' and entries:
        answer = _format_section_answer(entries, max_items=1)
        if answer:
            return answer
    return ''


def _extract_faculty_names_from_text(text: str) -> List[str]:
    text = ' '.join(text.replace('\xa0', ' ').split())
    pattern = r'(Khoa\s+[A-ZÀ-ỴA-Za-zÀ-ỹ&\-/–\s]+?)\s+Posted on'
    names = []
    for match in re.finditer(pattern, text):
        name = ' '.join(match.group(1).split())
        if name.startswith('Khoa chuyên môn '):
            name = name.replace('Khoa chuyên môn ', '', 1)
        if name not in names:
            names.append(name)
    return names


def _answer_faculties(question: str, contexts: List[Dict[str, Any]], faculty_chunks: List[Any]) -> str:
    q = norm_text_ascii(question)
    names: List[str] = []
    if faculty_chunks:
        source_text = ' '.join(getattr(chunk, 'text', '') for chunk in faculty_chunks)
        for chunk in faculty_chunks:
            name = str(getattr(chunk, 'entity_name', '') or '').strip()
            name_norm = norm_text_ascii(name)
            if name_norm.startswith('khoa ') and name_norm not in {'khoa chuyen mon'} and name not in names:
                names.append(name)
    else:
        source_text = ' '.join(ctx.get('text', '') for ctx in contexts)
    for name in _extract_faculty_names_from_text(source_text):
        if name not in names:
            names.append(name)
    if not names:
        return ''
    if 'bao nhieu khoa' in q or 'co bao nhieu khoa' in q:
        lines = [f"- {name}" for name in names]
        return f"Trong tài liệu hiện có {len(names)} khoa:\n" + '\n'.join(lines)
    if 'danh sach khoa' in q or 'gom nhung khoa nao' in q:
        return '\n'.join(f"- {name}" for name in names)
    return ''


def _pick_primary_context(contexts: List[Dict[str, Any]], question: str) -> Dict[str, Any]:
    q = norm_text_ascii(question)
    q_tokens = [tok for tok in q.split() if len(tok) >= 4]
    is_contact_query = any(tok in q for tok in ['o dau', 'dia chi', 'email', 'website', 'web', 'link', 'so dien thoai', 'sdt', 'van phong'])
    is_howto_query = _is_howto_like_question(question)
    best_ctx = contexts[0]
    best_score = -1.0
    for ctx in contexts:
        meta = ctx.get('metadata') or {}
        text = norm_text_ascii(_context_text(ctx) + ' ' + ctx.get('text', ''))
        score = 0.0
        score += 0.15 * sum(1 for tok in q_tokens if tok in text)
        score += 0.8 * float(ctx.get('score', 0.0) or 0.0)
        if ctx.get('kind') == 'qa':
            score += 0.25
        if any(meta.get(key) for key in ['name', 'role', 'email', 'address', 'city']):
            score += 0.2
        if is_contact_query:
            if ctx.get('kind') == 'qa':
                score += 0.7
            if any(tok in text for tok in ['co dia chi', 'dia chi', 'email', 'website', 'linh nam', 'minh khai', 'tran hung dao', 'ha noi', 'nam dinh']):
                score += 0.8
        if is_howto_query:
            score += _instruction_specificity(ctx.get('text', '')) * 0.45
            if ctx.get('kind') == 'qa':
                score += 0.2
        if ('hieu truong' in q or 'nguoi dung dau' in q) and 'hieu truong' in text and 'pho hieu truong' not in text:
            score += 1.2
        if 'ha noi' in q and 'ha noi' in text:
            score += 1.0
        if 'nam dinh' in q and 'nam dinh' in text:
            score += 1.0
        if score > best_score:
            best_score = score
            best_ctx = ctx
    return best_ctx


def _explicit_unit_fragment(question: str) -> str:
    q = norm_text_ascii(question)
    for pattern in [
        r'\b(?:truong|pho truong)\s+(?:khoa|phong)\s+([a-z0-9][a-z0-9\s&.-]*)',
        r'\b(?:khoa|phong)\s+([a-z0-9][a-z0-9\s&.-]*)',
    ]:
        match = re.search(pattern, q)
        if not match:
            continue
        fragment = match.group(1)
        fragment = re.split(
            r'\b(?:o dau|dia chi|email|website|web|link|so dien thoai|sdt|chuc nang|nhiem vu|lich su|thanh tich|quy mo|co cau|to chuc|la gi|la ai|nhu the nao|lam viec khi nao)\b',
            fragment,
            maxsplit=1,
        )[0].strip()
        if fragment:
            return fragment
    return ''


def _prefer_top_qa_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return ''
    q_norm = norm_text_ascii(question)
    if any(tag in q_norm for tag in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa']):
        return ''
    top = contexts[0]
    if top.get('kind') != 'qa':
        return ''
    title = str(top.get('title', '') or '')
    text = str(top.get('text', '') or '')
    combined = f'{title} {text}'.strip()
    unit_fragment = _explicit_unit_fragment(question)
    if unit_fragment and unit_fragment not in norm_text_ascii(combined):
        return ''
    coverage = _query_coverage(question, combined)
    score = float(top.get('score', 0.0) or 0.0)
    if coverage >= 0.6 or (coverage >= 0.45 and score >= 1.0):
        return text.strip()
    return ''


def _rule_answer(
    question: str,
    contexts: List[Dict[str, Any]],
    plan: Dict[str, Any],
    domain_data: Dict[str, Any],
    doc_chunks: Dict[str, List[Any]],
) -> str:
    domain = plan.get('domain', '')
    if _is_school_overview_query(question):
        answer = _answer_school_overview({
            'lich_su_hinh_thanh': load_seed_knowledge()['domains'].get('lich_su_hinh_thanh', {}),
            'co_so_vat_chat': load_seed_knowledge()['domains'].get('co_so_vat_chat', {}),
        })
        if answer:
            return answer
    if domain == 'ban_giam_hieu':
        return _answer_leadership(question, domain_data)
    if domain == 'co_so_vat_chat':
        return _answer_campus(question, domain_data)
    if domain == 'lich_su_hinh_thanh' and _is_school_overview_query(question):
        answer = _answer_school_overview({
            'lich_su_hinh_thanh': domain_data,
            'co_so_vat_chat': load_seed_knowledge()['domains'].get('co_so_vat_chat', {}),
        })
        if answer:
            return answer
    if domain == 'portal_howto':
        return _answer_portal(question, domain_data, plan)
    query_hint = str(plan.get('retrieval_query', '') or question)
    if domain == 'phong_ban_va_chuc_nang':
        if _unit_query_topic(question) == 'leadership':
            role_question = _resolve_role_query(question, query_hint)
            role_answer = _compact_role_identity_answer(_role_identity_answer(role_question, plan, domain_data, doc_chunks))
            if role_answer:
                return role_answer
            return _missing_role_identity_answer(role_question)
        qa_answer = _prefer_top_qa_answer(question, contexts)
        if qa_answer:
            return qa_answer
        qa_answer = _answer_named_unit_qa(question, domain_data)
        if qa_answer:
            return qa_answer
        answer = _answer_named_unit(question, query_hint, doc_chunks.get('phong-ban-va-chuc-nang', []))
        if answer:
            return answer
    if domain == 'khoa_chuyen_mon':
        if _unit_query_topic(question) == 'leadership':
            role_question = _resolve_role_query(question, query_hint)
            role_answer = _compact_role_identity_answer(_role_identity_answer(role_question, plan, domain_data, doc_chunks))
            if role_answer:
                return role_answer
            return _missing_role_identity_answer(role_question)
        qa_answer = _prefer_top_qa_answer(question, contexts)
        if qa_answer:
            return qa_answer
        qa_answer = _answer_named_unit_qa(question, domain_data)
        if qa_answer:
            return qa_answer
        answer = _answer_named_unit(question, query_hint, doc_chunks.get('khoa-chuyen-mon', []))
        if answer:
            return answer
        return _answer_faculties(question, contexts, doc_chunks.get('khoa-chuyen-mon', []))
    return ''


QUESTION_STOPWORDS = {
    'cach', 'nhu', 'the', 'nao', 'o', 'dau', 'la', 'gi', 'giup', 'em', 'minh', 'toi', 'cho', 'hoi',
    'ban', 'vao', 'xem', 'tra', 'cuu', 'lam', 'sao', 'voi', 'nhe', 'a', 'ah',
}

ROLE_SUBJECT_HINTS = [
    'hieu truong', 'pho hieu truong', 'hieu pho',
    'truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa',
]
PERSON_DETAIL_LIMITS = [
    ('birth_date', 'ngày sinh', ['ngay sinh', 'sinh ngay', 'ngay thang nam sinh']),
    ('birth_year', 'năm sinh', ['nam sinh']),
    ('age', 'tuổi', ['tuoi', 'bao nhieu tuoi']),
    ('hometown', 'quê quán', ['que quan', 'noi sinh', 'sinh o dau']),
    ('private_phone', 'số điện thoại cá nhân', ['so dien thoai', 'dien thoai', 'sdt']),
    ('private_address', 'địa chỉ riêng', ['dia chi nha', 'dia chi rieng', 'nha o dau', 'noi o']),
    ('social_contact', 'liên hệ cá nhân', ['facebook', 'zalo']),
    ('identity_doc', 'thông tin giấy tờ cá nhân', ['cccd', 'cmnd', 'can cuoc']),
    ('family', 'thông tin gia đình', ['gia dinh', 'thong tin gia dinh', 'vo chong', 'bo me', 'con cai']),
]


def _content_tokens(text: str) -> List[str]:
    q = norm_text_ascii(text)
    return [tok for tok in q.split() if len(tok) >= 3 and tok not in QUESTION_STOPWORDS]


def _text_has_hint(text: str, hint: str) -> bool:
    q = norm_text_ascii(text)
    token = norm_text_ascii(hint)
    if not token:
        return False
    if ' ' in token:
        return token in q
    return bool(re.search(rf'\b{re.escape(token)}\b', q))


def _contains_role_subject(question: str, plan: Dict[str, Any]) -> bool:
    q = norm_text_ascii(question)
    if str(plan.get('domain', '') or '') == 'ban_giam_hieu':
        return any(hint in q for hint in ['hieu truong', 'pho hieu truong', 'hieu pho'])
    return any(hint in q for hint in ROLE_SUBJECT_HINTS)


def _detect_detail_limited_request(question: str, plan: Dict[str, Any]) -> Dict[str, str]:
    if not _contains_role_subject(question, plan):
        return {}
    q = norm_text_ascii(question)
    for kind, label, hints in PERSON_DETAIL_LIMITS:
        if any(_text_has_hint(q, hint) for hint in hints):
            return {'kind': kind, 'label': label}
    return {}


def _base_role_question(question: str, plan: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    if str(plan.get('domain', '') or '') == 'ban_giam_hieu':
        if 'pho hieu truong' in q or 'hieu pho' in q:
            return 'pho hieu truong la ai'
        if 'hieu truong' in q:
            return 'hieu truong la ai'
        return ''

    role_patterns = [
        r'(pho truong khoa\s+[a-z0-9][a-z0-9\s&./-]*)',
        r'(truong khoa\s+[a-z0-9][a-z0-9\s&./-]*)',
        r'(pho truong phong\s+[a-z0-9][a-z0-9\s&./-]*)',
        r'(truong phong\s+[a-z0-9][a-z0-9\s&./-]*)',
        r'(tro ly khoa\s+[a-z0-9][a-z0-9\s&./-]*)',
    ]
    for pattern in role_patterns:
        match = re.search(pattern, q)
        if not match:
            continue
        fragment = re.split(
            r'\b(?:ngay sinh|nam sinh|tuoi|bao nhieu tuoi|que quan|noi sinh|sinh o dau|so dien thoai|dien thoai|sdt|dia chi nha|dia chi rieng|nha o dau|noi o|facebook|zalo|cccd|cmnd|can cuoc|vo|chong|con|bo me|gia dinh|la ai|la gi|nhu the nao)\b',
            match.group(1),
            maxsplit=1,
        )[0].strip()
        if fragment:
            return f'{fragment} la ai'
    return ''


def _resolve_role_query(question: str, query_hint: str) -> str:
    if _explicit_unit_fragment(question):
        return question
    q = norm_text_ascii(question)
    hint_fragment = _explicit_unit_fragment(query_hint)
    if not hint_fragment:
        hint_norm = norm_text_ascii(query_hint)
        question_norm = norm_text_ascii(question)
        if hint_norm.startswith(question_norm):
            suffix = hint_norm[len(question_norm):].strip()
            if suffix:
                hint_fragment = suffix
    if not hint_fragment:
        return question
    if 'pho truong khoa' in q:
        return f'pho truong khoa {hint_fragment} la ai'
    if 'truong khoa' in q:
        return f'truong khoa {hint_fragment} la ai'
    if 'pho truong phong' in q:
        return f'pho truong phong {hint_fragment} la ai'
    if 'truong phong' in q:
        return f'truong phong {hint_fragment} la ai'
    if 'tro ly khoa' in q:
        return f'tro ly khoa {hint_fragment} la ai'
    return question


def _role_identity_answer(
    question: str,
    plan: Dict[str, Any],
    domain_data: Dict[str, Any],
    doc_chunks: Dict[str, List[Any]],
) -> str:
    base_question = _base_role_question(question, plan)
    if not base_question:
        return ''
    domain = str(plan.get('domain', '') or '')
    if domain == 'ban_giam_hieu':
        return str(_answer_leadership(base_question, domain_data) or '').strip()
    if domain == 'phong_ban_va_chuc_nang':
        answer = str(_answer_named_unit(base_question, base_question, doc_chunks.get('phong-ban-va-chuc-nang', [])) or '').strip()
        return answer or _role_identity_from_qas(base_question, domain_data)
    if domain == 'khoa_chuyen_mon':
        answer = str(_answer_named_unit(base_question, base_question, doc_chunks.get('khoa-chuyen-mon', [])) or '').strip()
        return answer or _role_identity_from_qas(base_question, domain_data)
    return ''


def _compact_role_identity_answer(answer: str) -> str:
    text = str(answer or '').strip()
    if not text:
        return ''
    if '|' in text:
        parts = [part.strip(' .') for part in text.split('|') if part.strip()]
        if len(parts) >= 2:
            name = parts[0]
            role = parts[1]
            return f'{role} là {name}.'
    if not text.endswith('.'):
        text += '.'
    return text


def _missing_role_identity_answer(question: str) -> str:
    q = norm_text_ascii(question)
    role_label = 'Lãnh đạo đơn vị'
    if 'pho truong khoa' in q:
        role_label = 'Phó trưởng khoa'
    elif 'truong khoa' in q:
        role_label = 'Trưởng khoa'
    elif 'pho truong phong' in q:
        role_label = 'Phó trưởng phòng'
    elif 'truong phong' in q:
        role_label = 'Trưởng phòng'
    elif 'tro ly khoa' in q:
        role_label = 'Trợ lý khoa'

    unit_fragment = _explicit_unit_fragment(question)
    if unit_fragment:
        unit_label = ' '.join(part.capitalize() for part in unit_fragment.split())
        return f'Tôi chưa thấy thông tin về {role_label} {unit_label} trong tri thức hiện có.'
    return f'Tôi chưa thấy thông tin về {role_label.lower()} trong tri thức hiện có.'


def _role_identity_from_qas(question: str, domain_data: Dict[str, Any]) -> str:
    q = norm_text_ascii(question)
    q_tokens = set(_unit_match_tokens(q))
    if not q_tokens:
        return ''
    role_tags = [tag for tag in ['truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong', 'tro ly khoa'] if tag in q]
    best_answer = ''
    best_score = 0.0
    for qa in domain_data.get('qa', []):
        qa_q = norm_text_ascii(qa.get('question', ''))
        qa_a = norm_text_ascii(qa.get('answer', ''))
        if role_tags and not any(tag in qa_q or tag in qa_a for tag in role_tags):
            continue
        qa_tokens = set(_unit_match_tokens(f'{qa_q} {qa_a}'))
        if not qa_tokens:
            continue
        score = len(q_tokens & qa_tokens)
        if q_tokens.issubset(qa_tokens):
            score += 2
        if score > best_score:
            best_score = score
            best_answer = str(qa.get('answer', '') or '').strip()
    return best_answer if best_score >= 2 else ''


def _detail_limit_notice(detail: Dict[str, str]) -> str:
    kind = str(detail.get('kind', '') or '')
    label = str(detail.get('label', 'thông tin này') or 'thông tin này')
    if kind in {'private_phone', 'private_address', 'social_contact', 'identity_doc', 'family'}:
        return f'Còn thông tin về {label} là nhóm thông tin cá nhân riêng tư nên chatbot không cung cấp; trong dữ liệu hiện có cũng không có thông tin này.'
    return f'Còn thông tin về {label} hiện không có trong tri thức và vượt quá mức chi tiết chatbot được phép xác nhận.'


def _maybe_compose_limited_detail_answer(
    question: str,
    answer: str,
    plan: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    domain_data: Dict[str, Any],
    doc_chunks: Dict[str, List[Any]],
) -> Tuple[str, Dict[str, str]]:
    detail = _detect_detail_limited_request(question, plan)
    if not detail:
        return answer, {}

    answer_text = norm_text_ascii(answer)
    if str(detail.get('label', '') or '') and norm_text_ascii(detail['label']) in answer_text:
        return answer, {}

    identity_answer = _role_identity_answer(question, plan, domain_data, doc_chunks)
    if not identity_answer:
        identity_answer = answer
    identity_answer = _compact_role_identity_answer(identity_answer)
    if not identity_answer:
        return answer, {}

    final_answer = f'{identity_answer} {_detail_limit_notice(detail)}'.strip()
    return final_answer, detail


def _query_coverage(question: str, answer: str) -> float:
    q_tokens = _content_tokens(question)
    if not q_tokens:
        return 0.0
    answer_text = norm_text_ascii(answer)
    hits = sum(1 for tok in q_tokens if tok in answer_text)
    return hits / max(1, len(q_tokens))


def _context_coverage(answer: str, contexts: List[Dict[str, Any]]) -> float:
    if not answer or not contexts:
        return 0.0
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_text = norm_text_ascii(' '.join(c.get('text', '') for c in contexts))
    hits = sum(1 for tok in answer_tokens if tok in context_text)
    return hits / max(1, len(answer_tokens))


def _answer_focus(question: str, answer: str) -> float:
    q_tokens = set(_content_tokens(question))
    a_tokens = _content_tokens(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    hits = sum(1 for tok in a_tokens if tok in q_tokens)
    return hits / max(1, len(a_tokens))


def _verification_query(question: str, answer: str) -> str:
    answer_words = ' '.join(str(answer or '').split()).split()
    answer_hint = ' '.join(answer_words[:42])
    return f'{question} {answer_hint}'.strip()


def _reward_score(question: str, answer: str) -> float:
    try:
        return float(load_reward_model().score(question, answer))
    except Exception:
        return 0.5


def _reward_model_kind() -> str:
    try:
        return str(getattr(load_reward_model(), 'metadata', {}).get('kind', '') or '')
    except Exception:
        return ''


def _verify_answer_with_retrieval(
    question: str,
    answer: str,
    plan: Dict[str, Any],
    store: KnowledgeStore,
    chunk_docs: List[str],
    domain_data: Dict[str, Any],
    doc_chunks: Dict[str, List[Any]],
) -> Tuple[str, Dict[str, Any]]:
    domain = str(plan.get('domain', '') or '')
    if domain not in {'khoa_chuyen_mon', 'phong_ban_va_chuc_nang', 'portal_howto', 'co_so_vat_chat', 'ban_giam_hieu'}:
        return answer, {}
    query = _verification_query(question, answer)
    if not query:
        return answer, {}
    try:
        items = store.chunk_retrieve(query, chunk_docs, top_k=4)
    except Exception as exc:
        return answer, {'error': str(exc)}
    contexts = [
        {
            'kind': x.kind,
            'text': x.text,
            'title': x.title,
            'source_id': x.source_id,
            'metadata': x.metadata,
            'score': x.score,
        }
        for x in items
    ]
    debug = {
        'query': query,
        'contexts': [
            {
                'source_id': ctx.get('source_id', ''),
                'score': round(float(ctx.get('score', 0.0) or 0.0), 6),
                'chunk_type': (ctx.get('metadata') or {}).get('chunk_type', ''),
                'entity_name': (ctx.get('metadata') or {}).get('entity_name', ''),
            }
            for ctx in contexts
        ],
    }
    answer_reward = _reward_score(question, answer)
    debug['reward_scores'] = {
        'answer': round(answer_reward, 6),
        'model_kind': _reward_model_kind(),
    }
    if not contexts or not _has_strong_evidence(contexts):
        return answer, debug

    candidate = str(_rule_answer(question, contexts, plan, domain_data, doc_chunks) or '').strip()
    if not candidate and _has_strong_evidence(contexts):
        candidate = str(_fallback_answer(question, contexts, plan) or '').strip()
    if not candidate or norm_text_ascii(candidate) == norm_text_ascii(answer):
        return answer, debug
    if not answer_guard(candidate, contexts):
        return answer, debug

    candidate_focus = _answer_focus(question, candidate)
    answer_focus = _answer_focus(question, answer)
    candidate_coverage = _query_coverage(question, candidate)
    answer_coverage = _query_coverage(question, answer)
    candidate_reward = _reward_score(question, candidate)
    debug['reward_scores'] = {
        'answer': round(answer_reward, 6),
        'candidate': round(candidate_reward, 6),
        'model_kind': _reward_model_kind(),
    }
    shorter = len(candidate) + 80 < len(answer)
    reward_better = candidate_reward >= answer_reward + 0.12 and candidate_focus + candidate_coverage >= answer_focus + answer_coverage - 0.08
    if candidate_focus >= answer_focus + 0.03 or candidate_coverage >= answer_coverage + 0.15 or shorter or reward_better:
        debug['candidate_applied'] = True
        return candidate, debug
    return answer, debug


def _post_verify_answer(
    question: str,
    answer: str,
    plan: Dict[str, Any],
    contexts: List[Dict[str, Any]],
    domain_data: Dict[str, Any],
    doc_chunks: Dict[str, List[Any]],
) -> str:
    if not answer:
        return answer
    if not contexts:
        return answer

    candidates: List[Tuple[float, str, str]] = []
    intent = str(plan.get('intent', '') or '')
    is_howto_query = _is_howto_like_question(question, intent)
    rule_answer = str(_rule_answer(question, contexts, plan, domain_data, doc_chunks) or '').strip()

    if rule_answer:
        if norm_text_ascii(rule_answer) == norm_text_ascii(answer):
            return answer
        rule_focus = _answer_focus(question, rule_answer)
        answer_focus = _answer_focus(question, answer)
        rule_specificity = _instruction_specificity(rule_answer)
        answer_specificity = _instruction_specificity(answer)
        if intent == 'list':
            return rule_answer
        if intent in {'factoid', 'contact', 'howto'}:
            if not answer_guard(answer, contexts):
                return rule_answer
            if is_howto_query and rule_specificity >= answer_specificity + 0.4:
                return rule_answer
            if rule_focus >= answer_focus + 0.03:
                return rule_answer
            if len(rule_answer) * 2 <= max(1, len(answer)):
                return rule_answer

    def add_candidate(text: str, source: str, bonus: float = 0.0):
        candidate = str(text or '').strip()
        if not candidate:
            return
        if any(existing == candidate for _, existing, _ in candidates):
            return
        support = _context_coverage(candidate, contexts)
        if answer_guard(candidate, contexts):
            support = max(support, 0.85)
        relevance = _query_coverage(question, candidate)
        focus = _answer_focus(question, candidate)
        brevity = 1.0
        specificity = _instruction_specificity(candidate) if is_howto_query else 0.0
        if intent in {'factoid', 'contact', 'howto'}:
            brevity = min(1.0, 220.0 / max(220.0, float(len(candidate))))
        reward = _reward_score(question, candidate)
        score = support * 0.35 + relevance * 0.22 + focus * 0.18 + brevity * 0.13 + (reward - 0.5) * 0.12 + bonus
        if is_howto_query:
            score += min(0.35, specificity * 0.12)
        candidates.append((score, candidate, source))

    add_candidate(answer, 'original', bonus=0.02)
    add_candidate(rule_answer, 'rule', bonus=0.18)
    if _has_strong_evidence(contexts):
        add_candidate(_fallback_answer(question, contexts, plan), 'fallback', bonus=0.02)

    if not candidates:
        return answer

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_answer, best_source = candidates[0]
    original_score = next((score for score, candidate, _ in candidates if candidate == answer), 0.0)

    if best_source != 'original' and best_score >= original_score + 0.08:
        return best_answer
    if best_source == 'original':
        return answer
    if not answer_guard(answer, contexts) and best_score >= 0.3:
        return best_answer
    return answer


def _fallback_answer(question: str, contexts: List[Dict[str, Any]], plan: Dict[str, Any]) -> str:
    if not contexts:
        return ''

    intent = plan.get('intent', 'factoid')
    unique_contexts = []
    seen_text = set()
    for ctx in contexts:
        key = ctx.get('text', '').strip()
        if key and key not in seen_text:
            seen_text.add(key)
            unique_contexts.append(ctx)

    if intent == 'howto':
        return unique_contexts[0]['text']
    if intent == 'list':
        return '\n'.join(f"- {_context_text(ctx)}" for ctx in unique_contexts[:6])
    if intent == 'summary':
        first = unique_contexts[0]['text']
        bullets = [f"- {ctx['text']}" for ctx in unique_contexts[1:4]]
        return first if not bullets else first + '\n' + '\n'.join(bullets)

    return _context_text(_pick_primary_context(unique_contexts, question))


def _answer_from_uneti_web(question: str, web_client: UnetiWebClient) -> Tuple[str, Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    try:
        if is_web_notice_query(question):
            if any(tok in norm_text_ascii(question) for tok in ['moi nhat', 'gan day', 'hom nay', 'thong bao']):
                items = web_client.latest_notices(limit=5)
            else:
                items = web_client.search(question, limit=5)
        else:
            items = web_client.search(question, limit=5)
    except Exception as e:
        debug['used'] = False
        debug['error'] = str(e)
        return '', debug

    if not items:
        debug['used'] = False
        debug['count'] = 0
        return '', debug

    lines = []
    intro = 'Tôi không thấy câu trả lời rõ trong tài liệu hiện có. Dưới đây là thông tin tham khảo từ uneti.edu.vn:'
    for item in items[:5]:
        label = item.title
        if item.published_at:
            label += f" ({item.published_at})"
        lines.append(f"- {label}: {item.link}")
    debug['used'] = True
    debug['count'] = len(items[:5])
    return intro + '\n' + '\n'.join(lines), debug


def _infer_named_unit(domain: str, question: str, plan: Dict[str, Any], doc_chunks: Dict[str, List[Any]], contexts: List[Dict[str, Any]] | None = None) -> str:
    if domain == 'phong_ban_va_chuc_nang':
        chunks = doc_chunks.get('phong-ban-va-chuc-nang', [])
    elif domain == 'khoa_chuyen_mon':
        chunks = doc_chunks.get('khoa-chuyen-mon', [])
    else:
        return ''
    profiles = _extract_unit_profiles(chunks)
    contacts = _extract_unit_contact_index(chunks)
    unit_names = sorted(set(profiles.keys()) | set(contacts.keys()))
    matched = _match_named_unit(str(plan.get('retrieval_query', '') or question), unit_names)
    if matched:
        return matched
    query = norm_text_ascii(str(plan.get('retrieval_query', '') or question))
    for ctx in contexts or []:
        meta = ctx.get('metadata') or {}
        entity_name = str(meta.get('entity_name', '') or '').strip()
        entity_norm = norm_text_ascii(entity_name)
        if not entity_name:
            continue
        if domain == 'khoa_chuyen_mon' and not entity_norm.startswith('khoa '):
            continue
        if domain == 'phong_ban_va_chuc_nang' and not entity_norm.startswith('phong '):
            continue
        short_entity = entity_norm
        for prefix in ['khoa ', 'phong ']:
            if short_entity.startswith(prefix):
                short_entity = short_entity[len(prefix):].strip()
                break
        if entity_norm in query or (short_entity and short_entity in query):
            return entity_name
    return ''


def _infer_entity_from_answer(domain: str, question: str, answer: str, contexts: List[Dict[str, Any]], domain_data: Dict[str, Any]) -> str:
    if domain == 'co_so_vat_chat':
        filters = detect_campus_filter(question)
        if filters.get('city'):
            return filters['city']
        if filters.get('address'):
            return filters['address']
        return ''
    if domain == 'portal_howto':
        return detect_portal_topic(question) or ''
    if domain == 'ban_giam_hieu':
        q = norm_text_ascii(question)
        for rec in domain_data.get('records', []):
            role = norm_text_ascii(rec.get('role', ''))
            if ('pho hieu truong' in q or 'hieu pho' in q) and 'pho hieu truong' in role:
                return rec.get('name', '')
            if ('hieu truong' in q or 'nguoi dung dau' in q) and 'hieu truong' in role and 'pho hieu truong' not in role:
                return rec.get('name', '')
    for c in contexts:
        meta = c.get('metadata', {})
        for key in ['name', 'role', 'address', 'city']:
            if meta.get(key):
                return meta[key]
    return ''
