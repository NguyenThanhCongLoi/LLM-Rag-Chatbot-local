import json
import os
import time
import streamlit as st
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.contribution_store import (
    classify_contribution_reply,
    contribution_decline_reply,
    contribution_invitation,
    contribution_thanks_reply,
    list_contributions,
    should_offer_contribution,
    submit_contribution,
    update_contribution,
)
from core.feedback_store import (
    get_feedback,
    is_feedback_eligible,
    list_feedback,
    upsert_feedback,
)
from core.history import load_history, save_history, DEFAULT_MEMORY
from core.llm import LLMConfig, OllamaClient
from core.pipeline_v4 import UnetiDocumentAgentV4Max
from core.review_store import (
    REVIEW_KIND_LABELS,
    REVIEW_KINDS,
    VERDICT_LABELS,
    VERDICTS,
    all_reviews,
    get_review,
    history_turns,
    list_history_student_ids,
    upsert_review,
)

APP_TITLE = 'Chatbot hỗ trợ sinh viên uneti'
ADMIN_TOKEN = os.getenv('UNETI_ADMIN_TOKEN', 'liostcookin')


def inject_chat_css():
    st.markdown(
        """
        <style>
        .uneti-welcome {
            background: linear-gradient(135deg, #fff8e8 0%, #eef7ff 100%);
            border: 1px solid rgba(12, 74, 110, 0.12);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 14px;
        }
        .uneti-welcome h3 {
            margin: 0 0 8px 0;
            font-size: 1.1rem;
            color: #0f172a;
        }
        .uneti-welcome p {
            margin: 0;
            color: #334155;
            line-height: 1.55;
        }
        .uneti-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
        }
        .uneti-chip {
            border-radius: 999px;
            padding: 6px 10px;
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(15, 23, 42, 0.08);
            color: #0f172a;
            font-size: 0.86rem;
        }
        .uneti-thinking {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 16px;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid rgba(148, 163, 184, 0.24);
            color: #0f172a;
            margin-top: 4px;
        }
        .uneti-thinking-dots {
            display: inline-flex;
            gap: 5px;
        }
        .uneti-thinking-dots span {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #0f766e;
            animation: unetiPulse 1.1s infinite ease-in-out;
        }
        .uneti-thinking-dots span:nth-child(2) { animation-delay: 0.14s; }
        .uneti-thinking-dots span:nth-child(3) { animation-delay: 0.28s; }
        @keyframes unetiPulse {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
            40% { transform: translateY(-4px); opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_state():
    st.markdown(
        """
        <div class="uneti-welcome">
            <h3>Xin chào. Tôi hỗ trợ tra cứu thông tin và tài liệu UNETI.</h3>
            <p>Ưu tiên trả lời ngắn, bám tài liệu và không bịa thêm. Bạn có thể hỏi về ban giám hiệu, cơ sở đào tạo, phòng ban, khoa, cổng sinh viên hoặc hướng dẫn thao tác.</p>
            <div class="uneti-chip-row">
                <span class="uneti-chip">Hiệu trưởng là ai?</span>
                <span class="uneti-chip">Địa chỉ cơ sở Hà Nội ở đâu?</span>
                <span class="uneti-chip">Phòng Đào tạo làm gì?</span>
                <span class="uneti-chip">Cách xem lịch học</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def thinking_placeholder_html(message: str = 'Đang suy nghĩ') -> str:
    return f"""
    <div class="uneti-thinking">
        <div class="uneti-thinking-dots"><span></span><span></span><span></span></div>
        <div>{message}</div>
    </div>
    """


def is_admin_token_valid(token: str) -> bool:
    return bool(token) and token == ADMIN_TOKEN


def parse_debug_text(raw: str):
    try:
        return json.loads(raw) if raw else {}
    except Exception:
        return {}


def feedback_turns(messages: list[dict], student_id: str) -> list[dict]:
    turns = []
    last_user = ''
    last_user_index = -1
    for idx, msg in enumerate(messages):
        role = str(msg.get('role', '') or '')
        if role == 'user':
            last_user = str(msg.get('content', '') or '')
            last_user_index = idx
            continue
        if role != 'assistant':
            continue
        debug = parse_debug_text(str(msg.get('debug', '') or ''))
        if not is_feedback_eligible(debug):
            continue
        turns.append({
            'turn_id': f'{student_id}:{idx}',
            'student_id': student_id,
            'assistant_index': idx,
            'user_index': last_user_index,
            'question': last_user,
            'answer': str(msg.get('content', '') or ''),
            'debug': debug,
        })
    turns.reverse()
    return turns


POLICY_CODE_OPTIONS = [
    '',
    'student_private_info',
    'class_private_info',
    'class_scope_not_supported',
]
POLICY_CODE_LABELS = {
    '': 'Không áp dụng',
    'student_private_info': 'Thông tin riêng tư sinh viên',
    'class_private_info': 'Thông tin chi tiết theo lớp',
    'class_scope_not_supported': 'Thông tin số lớp ngoài phạm vi',
}


def clear_pending_contribution(memory: dict) -> dict:
    updated = dict(memory or {})
    updated['awaiting_user_contribution'] = False
    updated['pending_contribution_question'] = ''
    updated['pending_contribution_answer'] = ''
    updated['pending_contribution_domain'] = ''
    return updated


def set_pending_contribution(memory: dict, question: str, answer: str, debug: dict) -> dict:
    updated = clear_pending_contribution(memory)
    updated['awaiting_user_contribution'] = True
    updated['pending_contribution_question'] = question
    updated['pending_contribution_answer'] = answer
    updated['pending_contribution_domain'] = str((debug.get('plan') or {}).get('domain', '') or '')
    return updated

@st.cache_resource
def get_engine(ollama_embed_model: str, embed_model_ref: str, embed_local_only: bool) -> UnetiDocumentAgentV4Max:
    return UnetiDocumentAgentV4Max()

def login_screen():
    st.markdown('## Đăng nhập')
    student_tab, admin_tab = st.tabs(['Sinh viên', 'Admin'])
    with student_tab:
        st.markdown('Nhập **mã sinh viên** để sử dụng chatbot tài liệu UNETI.')
        with st.form('login_form', clear_on_submit=False):
            student_id = st.text_input('Mã sinh viên', placeholder='Ví dụ: 2217xxxxxxx').strip()
            submitted = st.form_submit_button('Vào chatbot', use_container_width=True)
        if submitted:
            if not student_id:
                st.error('Bạn cần nhập mã sinh viên.')
                return
            st.session_state.student_id = student_id
            st.session_state.admin_mode = False
            hist = load_history(student_id)
            st.session_state.messages = hist.get('messages', [])
            st.session_state.memory = hist.get('memory', dict(DEFAULT_MEMORY))
            st.rerun()
    with admin_tab:
        st.markdown('Nhập **admin token** để duyệt lịch sử chat, đánh giá câu trả lời và lưu đáp án chuẩn.')
        with st.form('admin_login_form', clear_on_submit=False):
            admin_token = st.text_input('Admin token', type='password').strip()
            submitted_admin = st.form_submit_button('Vào khu quản trị', use_container_width=True)
        if submitted_admin:
            if not is_admin_token_valid(admin_token):
                st.error('Admin token không hợp lệ.')
                return
            st.session_state.admin_mode = True
            st.session_state.pop('student_id', None)
            st.session_state.pop('messages', None)
            st.session_state.pop('memory', None)
            st.rerun()

def llm_sidebar(engine: UnetiDocumentAgentV4Max):
    st.sidebar.header('Ollama / Local LLM')
    enabled = st.sidebar.checkbox('Bật LLM', value=st.session_state.get('llm_enabled', False))
    analyzer_model = st.sidebar.text_input('Analyzer model', value=st.session_state.get('llm_analyzer', 'qwen2.5:1.5b'))
    answer_model = st.sidebar.text_input('Answer model', value=st.session_state.get('llm_answer', 'qwen2.5:1.5b'))
    base_url = st.sidebar.text_input('Ollama URL', value=st.session_state.get('llm_url', 'http://127.0.0.1:11434'))
    st.session_state.llm_enabled = enabled
    st.session_state.llm_analyzer = analyzer_model
    st.session_state.llm_answer = answer_model
    st.session_state.llm_url = base_url
    cfg = LLMConfig(enabled=enabled, analyzer_model=analyzer_model, answer_model=answer_model, base_url=base_url)
    engine.set_llm_config(cfg)
    if st.sidebar.button('Ping Ollama', use_container_width=True):
        client = OllamaClient(cfg)
        ok, msg = client.ping()
        st.session_state.ping = {'ok': ok, 'msg': msg}
    if 'ping' in st.session_state:
        if st.session_state.ping['ok']:
            st.sidebar.success('Kết nối Ollama OK')
        else:
            st.sidebar.error(st.session_state.ping['msg'])

def embedding_sidebar():
    st.sidebar.header('Embedding / Vector DB')
    ollama_embed_model = st.sidebar.text_input(
        'Ollama embedding model',
        value=st.session_state.get('ollama_embed_model', os.getenv('UNETI_OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')),
        help='Model embedding local trong Ollama. Gợi ý: nomic-embed-text:latest',
    )
    embed_model = st.sidebar.text_input(
        'Sentence-transformers ref',
        value=st.session_state.get('embed_model_ref', os.getenv('UNETI_EMBED_MODEL', '')),
        placeholder='Có thể để trống nếu chỉ dùng Ollama embeddings',
        help='Tùy chọn fallback nếu bạn có model local/cached của sentence-transformers. Có thể để trống nếu chỉ dùng Ollama embeddings.',
    )
    local_only = st.sidebar.checkbox(
        'Chỉ dùng local cache',
        value=st.session_state.get('embed_local_only', os.getenv('UNETI_EMBED_LOCAL_ONLY', '1') != '0'),
        help='Bật để tránh app cố tải model từ mạng.',
    )
    st.session_state.ollama_embed_model = ollama_embed_model
    st.session_state.embed_model_ref = embed_model
    st.session_state.embed_local_only = local_only
    os.environ['UNETI_OLLAMA_EMBED_MODEL'] = ollama_embed_model
    os.environ['UNETI_EMBED_MODEL'] = embed_model
    os.environ['UNETI_EMBED_LOCAL_ONLY'] = '1' if local_only else '0'

def top_bar(student_id: str):
    c1, c2, c3 = st.columns([6,2,1])
    with c1:
        st.title(APP_TITLE)
        st.caption('Local-only, grounded by documents, login by student ID, text-only answers.')
    with c2:
        label = 'ADMIN' if st.session_state.get('admin_mode') else student_id
        prefix = 'Mode' if st.session_state.get('admin_mode') else 'MSV'
        st.markdown(f'**{prefix}:** `{label}`')
    with c3:
        if st.button('Đăng xuất', use_container_width=True):
            for k in ['student_id', 'messages', 'memory', 'admin_mode']:
                st.session_state.pop(k, None)
            st.rerun()


def admin_dashboard():
    st.subheader('Đánh giá lịch sử chat')
    student_ids = list_history_student_ids()
    if not student_ids:
        st.info('Chưa có lịch sử chat nào để đánh giá.')
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_student = st.selectbox('Chọn mã sinh viên', student_ids, key='admin_selected_student')
    with c2:
        verdict_filter_options = ['all', *VERDICTS]
        verdict_filter_labels = ['Tất cả', *[VERDICT_LABELS[item] for item in VERDICTS]]
        verdict_filter = st.selectbox(
            'Lọc review',
            verdict_filter_options,
            format_func=lambda value: verdict_filter_labels[verdict_filter_options.index(value)],
            key='admin_verdict_filter',
        )

    turns = history_turns(selected_student)
    if not turns:
        st.info('Mã sinh viên này chưa có lượt assistant nào.')
        return

    turn_labels = [f"Lượt {idx + 1} | {turn['question'][:80] or '(không có câu hỏi)'}" for idx, turn in enumerate(turns)]
    selected_turn_label = st.selectbox('Chọn lượt cần đánh giá', turn_labels, key='admin_selected_turn')
    selected_turn = turns[turn_labels.index(selected_turn_label)]
    existing_review = get_review(selected_turn['turn_id'])

    left, right = st.columns(2)
    with left:
        st.markdown('**Câu hỏi người dùng**')
        st.text_area('question', value=selected_turn['question'], height=120, disabled=True, label_visibility='collapsed')
        st.markdown('**Câu trả lời chatbot**')
        st.text_area('answer', value=selected_turn['answer'], height=220, disabled=True, label_visibility='collapsed')
    with right:
        st.markdown('**Debug / evidence**')
        debug_data = parse_debug_text(selected_turn.get('debug', ''))
        if debug_data:
            st.json(debug_data)
        else:
            st.code(selected_turn.get('debug', '') or '(không có debug)')

    default_verdict = existing_review.get('verdict', 'partial') if existing_review else 'partial'
    default_answer = existing_review.get('approved_answer', '') if existing_review else ''
    if not default_answer and default_verdict == 'correct':
        default_answer = selected_turn['answer']
    default_matches = '\n'.join(existing_review.get('match_questions', []) if existing_review else [selected_turn['question']])
    default_domain = existing_review.get('domain_hint', '') if existing_review else ''
    default_review_kind = existing_review.get('review_kind', 'answer') if existing_review else ('policy' if debug_data.get('policy_code') else 'answer')
    default_policy_code = existing_review.get('policy_code', '') if existing_review else str(debug_data.get('policy_code', '') or '')
    domain_options = ['', 'ban_giam_hieu', 'co_so_vat_chat', 'lich_su_hinh_thanh', 'hoi_dong_truong', 'portal_howto', 'khoa_chuyen_mon', 'phong_ban_va_chuc_nang', 'general_docs']

    with st.form('admin_review_form', clear_on_submit=False):
        verdict = st.selectbox(
            'Đánh giá',
            VERDICTS,
            index=VERDICTS.index(default_verdict) if default_verdict in VERDICTS else VERDICTS.index('partial'),
            format_func=lambda value: VERDICT_LABELS.get(value, value),
        )
        review_kind = st.selectbox(
            'Loại review',
            REVIEW_KINDS,
            index=REVIEW_KINDS.index(default_review_kind) if default_review_kind in REVIEW_KINDS else 0,
            format_func=lambda value: REVIEW_KIND_LABELS.get(value, value),
        )
        policy_code = st.selectbox(
            'Policy code',
            POLICY_CODE_OPTIONS,
            index=POLICY_CODE_OPTIONS.index(default_policy_code) if default_policy_code in POLICY_CODE_OPTIONS else 0,
            format_func=lambda value: POLICY_CODE_LABELS.get(value, value),
            help='Dùng cho review loại Chính sách để tái sử dụng cho các câu hỏi cùng nhóm.',
        )
        approved_answer = st.text_area(
            'Câu trả lời chuẩn / đã sửa',
            value=default_answer,
            height=180,
            help='Nếu để trống và đánh dấu correct, hệ thống sẽ dùng lại câu trả lời hiện tại của chatbot.',
        )
        retrieval_hint = st.text_area(
            'Gợi ý cho chatbot tìm tri thức',
            value=existing_review.get('retrieval_hint', '') if existing_review else '',
            height=100,
            help='Ví dụ: thêm từ khóa, section, tên đơn vị hoặc logic cần ưu tiên.',
        )
        domain_hint = st.selectbox(
            'Gợi ý domain',
            domain_options,
            index=domain_options.index(default_domain) if default_domain in domain_options else 0,
        )
        match_questions = st.text_area(
            'Các câu hỏi áp dụng lại, mỗi dòng 1 câu',
            value=default_matches,
            height=120,
            help='Bot sẽ tái sử dụng đáp án/hint này cho các câu hỏi khớp ở các lượt sau.',
        )
        notes = st.text_area('Ghi chú admin', value=existing_review.get('notes', '') if existing_review else '', height=100)
        active = st.checkbox('Kích hoạt review này', value=existing_review.get('active', True) if existing_review else True)
        submitted = st.form_submit_button('Lưu đánh giá', use_container_width=True)

    if submitted:
        review = upsert_review({
            'turn_id': selected_turn['turn_id'],
            'student_id': selected_turn['student_id'],
            'assistant_index': selected_turn['assistant_index'],
            'user_question': selected_turn['question'],
            'assistant_answer': selected_turn['answer'],
            'verdict': verdict,
            'review_kind': review_kind,
            'policy_code': policy_code,
            'approved_answer': approved_answer,
            'retrieval_hint': retrieval_hint,
            'domain_hint': domain_hint,
            'notes': notes,
            'active': active,
            'match_questions': [line.strip() for line in match_questions.splitlines() if line.strip()],
        })
        st.success(f"Đã lưu review `{review['turn_id']}`.")
        st.rerun()

    st.markdown('**Thư viện review đã lưu**')
    reviews = all_reviews()
    if verdict_filter != 'all':
        reviews = [item for item in reviews if item.get('verdict') == verdict_filter]
    rows = [{
        'turn_id': item.get('turn_id', ''),
        'student_id': item.get('student_id', ''),
        'loại': REVIEW_KIND_LABELS.get(item.get('review_kind', 'answer'), item.get('review_kind', 'answer')),
        'verdict': VERDICT_LABELS.get(item.get('verdict', ''), item.get('verdict', '')),
        'policy_code': item.get('policy_code', ''),
        'so_lan_dung': item.get('usage_count', 0),
        'bien_the_tu_hoc': len(item.get('auto_match_questions', []) or []),
        'active': item.get('active', True),
        'question': item.get('user_question', '')[:100],
        'updated_at': item.get('updated_at', ''),
    } for item in reviews[:200]]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info('Chưa có review phù hợp với bộ lọc hiện tại.')

    st.markdown('**Bổ sung từ người dùng chờ duyệt**')
    pending_items = list_contributions('pending')
    if not pending_items:
        st.info('Chưa có bổ sung nào chờ admin kiểm tra.')
        return

    contribution_labels = [
        f"{item.get('contribution_id', '')} | {str(item.get('original_question', '') or '')[:80]}"
        for item in pending_items
    ]
    selected_contribution_label = st.selectbox(
        'Chọn bổ sung cần duyệt',
        contribution_labels,
        key='admin_selected_contribution',
    )
    selected_contribution = pending_items[contribution_labels.index(selected_contribution_label)]

    left, right = st.columns(2)
    with left:
        st.markdown('**Câu hỏi gốc**')
        st.text_area('orig_q', value=selected_contribution.get('original_question', ''), height=100, disabled=True, label_visibility='collapsed')
        st.markdown('**Câu trả lời trước đó của chatbot**')
        st.text_area('orig_a', value=selected_contribution.get('bot_answer', ''), height=140, disabled=True, label_visibility='collapsed')
    with right:
        st.markdown('**Thông tin người dùng bổ sung**')
        st.text_area('user_contrib', value=selected_contribution.get('user_contribution', ''), height=180, disabled=True, label_visibility='collapsed')
        st.caption(f"MSV: {selected_contribution.get('student_id', '')} | Domain: {selected_contribution.get('domain_hint', '')}")

    with st.form('admin_contribution_form', clear_on_submit=False):
        approved_answer = st.text_area(
            'Câu trả lời sau khi admin xác nhận',
            value=selected_contribution.get('user_contribution', ''),
            height=160,
        )
        domain_hint = st.selectbox(
            'Domain cho review mới',
            domain_options,
            index=domain_options.index(selected_contribution.get('domain_hint', '')) if selected_contribution.get('domain_hint', '') in domain_options else 0,
            key='contribution_domain_hint',
        )
        notes = st.text_area('Ghi chú admin cho bổ sung này', height=80)
        approve = st.form_submit_button('Duyệt và lưu thành review', use_container_width=True)
        reject = st.form_submit_button('Từ chối bổ sung', use_container_width=True)

    if approve:
        review = upsert_review({
            'turn_id': f"contribution:{selected_contribution.get('contribution_id', '')}",
            'student_id': selected_contribution.get('student_id', ''),
            'assistant_index': 0,
            'user_question': selected_contribution.get('original_question', ''),
            'assistant_answer': selected_contribution.get('bot_answer', ''),
            'verdict': 'missing',
            'review_kind': 'answer',
            'policy_code': '',
            'approved_answer': approved_answer,
            'retrieval_hint': '',
            'domain_hint': domain_hint,
            'notes': notes,
            'active': True,
            'match_questions': [selected_contribution.get('original_question', '')],
        })
        update_contribution(
            str(selected_contribution.get('contribution_id', '') or ''),
            status='approved',
            admin_note=notes,
            linked_review_id=review.get('turn_id', ''),
        )
        st.success('Đã duyệt bổ sung và lưu thành review.')
        st.rerun()

    if reject:
        update_contribution(
            str(selected_contribution.get('contribution_id', '') or ''),
            status='rejected',
            admin_note=notes,
        )
        st.warning('Đã từ chối bổ sung này.')
        st.rerun()

    st.markdown('**Đánh giá từ người dùng**')
    feedback_rows = [{
        'turn_id': item.get('turn_id', ''),
        'student_id': item.get('student_id', ''),
        'rating': item.get('rating', 0),
        'route': item.get('route', ''),
        'domain': item.get('domain', ''),
        'question': str(item.get('question', '') or '')[:100],
        'suggestion': str(item.get('answer_suggestion', '') or '')[:120],
        'updated_at': item.get('updated_at', ''),
    } for item in list_feedback()[:200]]
    if feedback_rows:
        st.dataframe(feedback_rows, use_container_width=True, hide_index=True)
    else:
        st.info('Chưa có đánh giá nào từ người dùng.')


def render_feedback_section(student_id: str, messages: list[dict]) -> None:
    turns = feedback_turns(messages, student_id)
    if not turns:
        return

    st.markdown('**Góp Ý Câu Trả Lời**')
    labels = [f"Lượt {turn['assistant_index']} | {turn['question'][:80] or '(không có câu hỏi)'}" for turn in turns]
    selected_label = st.selectbox('Chọn câu trả lời để góp ý', labels, key='user_feedback_turn')
    turn = turns[labels.index(selected_label)]
    existing = get_feedback(turn['turn_id'])
    default_rating = int(existing.get('rating', 3) or 3)

    left, right = st.columns(2)
    with left:
        st.markdown('**Câu hỏi của bạn**')
        st.text_area('feedback_question', value=turn['question'], height=90, disabled=True, label_visibility='collapsed')
        st.markdown('**Câu trả lời hiện tại**')
        st.text_area('feedback_answer', value=turn['answer'], height=180, disabled=True, label_visibility='collapsed')
    with right:
        with st.form(f"user_feedback_form_{turn['turn_id']}", clear_on_submit=False):
            rating = st.slider('Điểm câu trả lời', min_value=1, max_value=5, value=default_rating)
            answer_suggestion = st.text_area(
                'Nếu điểm thấp, bạn mong muốn câu trả lời như thế nào?',
                value=existing.get('answer_suggestion', '') if existing else '',
                height=140,
                help='Dùng để gợi ý câu trả lời lại hoặc nêu phần còn thiếu/sai.',
            )
            notes = st.text_area(
                'Góp ý thêm',
                value=existing.get('notes', '') if existing else '',
                height=80,
            )
            submitted = st.form_submit_button('Gửi góp ý', use_container_width=True)
        if submitted:
            feedback = upsert_feedback({
                'turn_id': turn['turn_id'],
                'student_id': student_id,
                'assistant_index': turn['assistant_index'],
                'question': turn['question'],
                'answer': turn['answer'],
                'rating': rating,
                'answer_suggestion': answer_suggestion,
                'notes': notes,
                'route': str(turn['debug'].get('route', '') or ''),
                'domain': str((turn['debug'].get('plan') or {}).get('domain', '') or ''),
            })
            st.success(f"Đã ghi nhận góp ý cho `{feedback['turn_id']}`.")
            st.rerun()

def main():
    st.set_page_config(page_title=APP_TITLE, page_icon='🎓', layout='wide')
    inject_chat_css()
    if st.session_state.get('admin_mode'):
        top_bar('ADMIN')
        admin_dashboard()
        return
    if 'student_id' not in st.session_state:
        login_screen(); return
    if 'messages' not in st.session_state:
        hist = load_history(st.session_state.student_id)
        st.session_state.messages = hist.get('messages', [])
    if 'memory' not in st.session_state:
        hist = load_history(st.session_state.student_id)
        st.session_state.memory = hist.get('memory', dict(DEFAULT_MEMORY))
    if 'show_debug' not in st.session_state:
        st.session_state.show_debug = False
    embedding_sidebar()
    engine = get_engine(
        st.session_state.get('ollama_embed_model', os.getenv('UNETI_OLLAMA_EMBED_MODEL', 'nomic-embed-text:latest')),
        st.session_state.get('embed_model_ref', os.getenv('UNETI_EMBED_MODEL', '')),
        st.session_state.get('embed_local_only', os.getenv('UNETI_EMBED_LOCAL_ONLY', '1') != '0'),
    )
    llm_sidebar(engine)
    top_bar(st.session_state.student_id)
    with st.sidebar.expander('Tài liệu seed đang nạp', expanded=False):
        for doc in engine.available_docs():
            st.write('-', doc)
    with st.expander('Tùy chọn debug', expanded=False):
        st.checkbox('Hiện debug', key='show_debug')
        st.json(st.session_state.memory)
    if not st.session_state.messages:
        render_welcome_state()
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            if msg.get('debug') and st.session_state.show_debug:
                st.code(msg['debug'], language='json')
    render_feedback_section(st.session_state.student_id, st.session_state.messages)
    user_input = st.chat_input('Hỏi về tài liệu UNETI...')
    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)
        pending_memory = dict(st.session_state.memory or {})
        if pending_memory.get('awaiting_user_contribution'):
            contribution_kind = classify_contribution_reply(user_input)
            if contribution_kind == 'provide':
                submit_contribution({
                    'student_id': st.session_state.student_id,
                    'original_question': pending_memory.get('pending_contribution_question', ''),
                    'bot_answer': pending_memory.get('pending_contribution_answer', ''),
                    'user_contribution': user_input,
                    'domain_hint': pending_memory.get('pending_contribution_domain', ''),
                    'route': 'user_contribution',
                })
                st.session_state.memory = clear_pending_contribution(pending_memory)
                answer = contribution_thanks_reply()
                debug = {'route': 'user_contribution_submitted'}
                debug_text = json.dumps(debug, ensure_ascii=False, indent=2)
                reply = {'role': 'assistant', 'content': answer, 'debug': debug_text}
                st.session_state.messages.append(reply)
                save_history(st.session_state.student_id, st.session_state.messages, st.session_state.memory)
                with st.chat_message('assistant'):
                    st.markdown(answer)
                    if st.session_state.show_debug:
                        st.code(debug_text, language='json')
                return
            if contribution_kind == 'decline':
                st.session_state.memory = clear_pending_contribution(pending_memory)
                answer = contribution_decline_reply()
                debug = {'route': 'user_contribution_declined'}
                debug_text = json.dumps(debug, ensure_ascii=False, indent=2)
                reply = {'role': 'assistant', 'content': answer, 'debug': debug_text}
                st.session_state.messages.append(reply)
                save_history(st.session_state.student_id, st.session_state.messages, st.session_state.memory)
                with st.chat_message('assistant'):
                    st.markdown(answer)
                    if st.session_state.show_debug:
                        st.code(debug_text, language='json')
                return
            st.session_state.memory = clear_pending_contribution(pending_memory)

        with st.chat_message('assistant'):
            thinking_slot = st.empty()
            started_at = time.perf_counter()
            thinking_slot.markdown(thinking_placeholder_html(), unsafe_allow_html=True)
            answer, debug, new_memory = engine.answer(user_input, st.session_state.memory)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            thinking_slot.empty()
            if should_offer_contribution(answer, debug):
                base_answer = answer
                answer = contribution_invitation(base_answer)
                new_memory = set_pending_contribution(new_memory, user_input, base_answer, debug)
                debug['contribution_requested'] = True
            else:
                new_memory = clear_pending_contribution(new_memory)
            st.session_state.memory = new_memory
            debug['latency_ms'] = elapsed_ms
            debug_text = json.dumps(debug, ensure_ascii=False, indent=2)
            reply = {'role': 'assistant', 'content': answer, 'debug': debug_text}
            st.session_state.messages.append(reply)
            save_history(st.session_state.student_id, st.session_state.messages, st.session_state.memory)
            st.markdown(answer)
            st.caption(f'Hoàn tất trong khoảng {elapsed_ms} ms')
            if st.session_state.show_debug:
                st.code(debug_text, language='json')

if __name__ == '__main__':
    main()
