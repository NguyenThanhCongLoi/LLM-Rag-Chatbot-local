from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import CONTRIBUTION_DB_PATH, FEEDBACK_DB_PATH, HISTORY_DIR, INDEX_DIR, REVIEW_DB_PATH, RLHF_DIR
from core.contribution_store import (
    classify_contribution_reply,
    contribution_decline_reply,
    contribution_invitation,
    contribution_thanks_reply,
    load_contribution_db,
    should_offer_contribution,
    submit_contribution,
    update_contribution,
)
from core.feedback_store import get_feedback, is_feedback_eligible, load_feedback_db, upsert_feedback
from core.normalize import norm_text_ascii
from core.pipeline_v4 import UnetiDocumentAgentV4Max
from core.reward_model import load_reward_model, train_reward_model
from scripts.build_rlhf_dataset import build_rlhf_dataset
from scripts.export_ppo_dataset import export_ppo_dataset
from scripts.export_rlhf_formats import export_rlhf_formats
from scripts.train_ppo import dry_run as ppo_dry_run


def assert_contains(text: str, expected: str):
    assert norm_text_ascii(expected) in norm_text_ascii(text), text


def assert_not_contains(text: str, unexpected: str):
    assert norm_text_ascii(unexpected) not in norm_text_ascii(text), text


initial_review_backup = REVIEW_DB_PATH.read_text(encoding='utf-8') if REVIEW_DB_PATH.exists() else ''
REVIEW_DB_PATH.write_text(json.dumps({'reviews': []}, ensure_ascii=False, indent=2), encoding='utf-8')


agent = UnetiDocumentAgentV4Max()
assert 'huong-dan-chuc-nang-cong-thong-tin-sv' in agent.available_docs()

ans0, debug0, mem0 = agent.answer('xin chao', {})
assert debug0.get('route') == 'meta'
assert debug0.get('meta_intent') == 'greeting'
assert_contains(ans0, 'Xin chao')
assert_contains(ans0, 'UNETI')

ans0id, debug0id, mem0id = agent.answer('ban la ai', {})
assert debug0id.get('route') == 'meta'
assert debug0id.get('meta_intent') == 'bot_identity'
assert_contains(ans0id, 'chatbot')
assert_contains(ans0id, 'UNETI')

ans0id2, debug0id2, _ = agent.answer('What you name ?', {})
assert debug0id2.get('route') == 'meta'
assert debug0id2.get('meta_intent') == 'bot_identity'
assert_contains(ans0id2, 'chatbot')

ans0cap, debug0cap, mem0cap = agent.answer('ban co the lam gi', {})
assert debug0cap.get('route') == 'meta'
assert debug0cap.get('meta_intent') == 'bot_capability'
assert_contains(ans0cap, 'phong ban')
assert_contains(ans0cap, 'cong sinh vien')

ans0cap2, debug0cap2, _ = agent.answer('ban giup gi duoc cho toi', {})
assert debug0cap2.get('route') == 'meta'
assert debug0cap2.get('meta_intent') == 'bot_capability'

ans0help, debug0help, mem0help = agent.answer('toi nen hoi gi', {})
assert debug0help.get('route') == 'meta'
assert debug0help.get('meta_intent') == 'bot_help'
assert_contains(ans0help, 'phong dao tao o dau')

ans0thanks, debug0thanks, mem0thanks = agent.answer('cam on', {})
assert debug0thanks.get('route') == 'meta'
assert debug0thanks.get('meta_intent') == 'thanks'
assert_contains(ans0thanks, 'Khong co gi')

ans0bye, debug0bye, mem0bye = agent.answer('tam biet', {})
assert debug0bye.get('route') == 'meta'
assert debug0bye.get('meta_intent') == 'farewell'
assert_contains(ans0bye, 'Chao ban')

ans0meta2, debug0meta2, mem0meta2 = agent.answer('sao may ho tro gi', {})
assert debug0meta2.get('route') == 'meta'
assert debug0meta2.get('meta_intent') == 'bot_capability'
assert_contains(ans0meta2, 'UNETI')

ans0meta3, debug0meta3, mem0meta3 = agent.answer('ho tro gi', {})
assert debug0meta3.get('route') == 'meta'
assert debug0meta3.get('meta_intent') == 'bot_capability'
assert_contains(ans0meta3, 'phong ban')

ans0complaint, debug0complaint, _ = agent.answer('sao ban lai khong biet', {})
assert debug0complaint.get('route') == 'meta'
assert debug0complaint.get('meta_intent') == 'bot_complaint'
assert_contains(ans0complaint, 'Xin loi')
assert_contains(ans0complaint, 'bo sung tri thuc')

ans0complaint2, debug0complaint2, _ = agent.answer('sao m ngu the', {})
assert debug0complaint2.get('route') == 'meta'
assert debug0complaint2.get('meta_intent') == 'bot_complaint'
assert_contains(ans0complaint2, 'admin kiem tra')

ans0complaint3, debug0complaint3, _ = agent.answer('toi khong hai long ve tinh nang search cua chatbot', {})
assert debug0complaint3.get('route') == 'meta'
assert debug0complaint3.get('meta_intent') == 'bot_complaint'
assert_contains(ans0complaint3, 'gui link')

ans0clarify, debug0clarify, mem0clarify = agent.answer('lo vay', {})
assert debug0clarify.get('route') == 'clarification'
assert_contains(ans0clarify, 'noi ro hon')

ans0clarify2, debug0clarify2, _ = agent.answer('toi khong biet', {})
assert debug0clarify2.get('route') == 'clarification'
assert_contains(ans0clarify2, 'noi ro hon')

ans0clarify3, debug0clarify3, _ = agent.answer('tiep di', {})
assert debug0clarify3.get('route') == 'clarification'
assert_contains(ans0clarify3, 'noi ro hon')

ans0b, debug0b, mem0b = agent.answer('thoi tiet hom nay the nao', {})
assert debug0b.get('route') == 'out_of_scope'
assert_contains(ans0b, 'ngoai pham vi ho tro cua chatbot')
assert_contains(ans0b, 'UNETI')

ans0c, debug0c, mem0c = agent.answer('thu do cua phap la gi', {})
assert debug0c.get('route') == 'out_of_scope'
assert_contains(ans0c, 'ngoai pham vi ho tro cua chatbot')

ans1, debug1, mem1 = agent.answer('các cơ sở của trường', {})
assert_contains(ans1, '4 địa điểm đào tạo')
assert_contains(ans1, 'Hà Nội')
assert_contains(ans1, 'Nam Định')

school_intro, school_intro_debug, _ = agent.answer('gioi thieu ve truong', {})
assert school_intro_debug['plan']['domain'] == 'lich_su_hinh_thanh'
assert_contains(school_intro, 'UNETI')
assert_contains(school_intro, 'thanh lap')
assert_contains(school_intro, 'dia diem dao tao')

ans2, debug2, mem2 = agent.answer('cơ sở hà nội', mem1)
assert_contains(ans2, 'Hà Nội')
assert_contains(ans2, 'Minh Khai')
assert_contains(ans2, '218 Lĩnh Nam')
assert_not_contains(ans2, '353 Trần Hưng Đạo')

ans3, debug3, mem3 = agent.answer('hiệu trưởng là ai', mem2)
assert_contains(ans3, 'Trần Hoàng Long')
assert_not_contains(ans3, 'Nguyễn Hữu Quang')

ans4, debug4, mem4 = agent.answer('cách đăng ký lịch học', mem3)
assert_contains(ans4, 'Lịch theo tuần')
assert_not_contains(ans4, 'Đăng ký học phần')

ans5, debug5, mem5 = agent.answer('có bao nhiêu khoa', {})
assert_contains(ans5, 'Trong tài liệu hiện có')
assert_contains(ans5, 'Khoa Cơ khí')
assert_contains(ans5, 'Khoa Công nghệ thông tin')

ans6, debug6, mem6 = agent.answer('mỗi lớp bao nhiêu sinh viên', {})
assert_contains(ans6, 'ngoài phạm vi chatbot hiện tại')
assert debug6.get('policy_code') == 'class_private_info'

ans6b, debug6b, mem6b = agent.answer('khoa cntt có bao nhiêu lớp', {})
assert debug6b.get('route') == 'sensitive_block'
assert debug6b.get('policy_code') == 'class_scope_not_supported'
assert_contains(ans6b, 'ngoài phạm vi chatbot hiện tại')
assert_contains(ans6b, 'số lớp chi tiết theo từng khoa')

ans6c, debug6c, mem6c = agent.answer('lớp cntt có bao nhiêu sinh viên', {})
assert debug6c.get('route') == 'sensitive_block'
assert debug6c.get('policy_code') == 'class_private_info'
assert_contains(ans6c, 'ngoài phạm vi chatbot hiện tại')
assert_contains(ans6c, 'thông tin chi tiết theo từng lớp')

ans6d, debug6d, mem6d = agent.answer('dhkl16a2hn có bao nhiêu sinh viên', {})
assert debug6d.get('route') == 'sensitive_block'
assert debug6d.get('policy_code') == 'class_private_info'
assert_contains(ans6d, 'mã lớp')
assert_contains(ans6d, 'điểm hay thông tin chi tiết theo từng lớp')

ans6e, debug6e, mem6e = agent.answer('điểm của sinh viên 22174600063', {})
assert debug6e.get('route') == 'sensitive_block'
assert debug6e.get('policy_code') == 'student_private_info'
assert_contains(ans6e, 'thông tin cá nhân của sinh viên')
assert_contains(ans6e, 'theo mã sinh viên')

ans6f, debug6f, mem6f = agent.answer('thông tin sinh viên 22174600063', {})
assert debug6f.get('route') == 'sensitive_block'
assert debug6f.get('policy_code') == 'student_private_info'
assert_contains(ans6f, 'thông tin cá nhân của sinh viên')

ans6g, debug6g, mem6g = agent.answer('điểm lớp dhkl16a2hn', {})
assert debug6g.get('route') == 'sensitive_block'
assert debug6g.get('policy_code') == 'class_private_info'
assert_contains(ans6g, 'mã lớp')

ans6h, debug6h, mem6h = agent.answer('số điện thoại sinh viên cntt', {})
assert debug6h.get('route') == 'sensitive_block'
assert debug6h.get('policy_code') == 'student_private_info'
assert_contains(ans6h, 'thông tin cá nhân của sinh viên')

ans6i, debug6i, _ = agent.answer('toi hoc lop nao', {})
assert debug6i.get('route') == 'policy_block'
assert debug6i.get('policy_code') == 'self_profile_not_supported'
assert_contains(ans6i, 'du lieu ca nhan')

ans6j, debug6j, _ = agent.answer('mon xac suat thong ke do ai day o khoa khoa hoc ung dung', {})
assert debug6j.get('route') == 'policy_block'
assert debug6j.get('policy_code') == 'teaching_assignment_not_supported'
assert_contains(ans6j, 'phan cong giang day')

ans6k, debug6k, _ = agent.answer('t muon dot truong', {})
assert debug6k.get('route') == 'policy_block'
assert debug6k.get('policy_code') == 'harmful_or_violent'
assert_contains(ans6k, 'bao luc')

ans6l, debug6l, _ = agent.answer('giao trinh hach toan ke toan', {})
assert debug6l.get('route') == 'policy_block'
assert debug6l.get('policy_code') == 'academic_material_not_supported'
assert_contains(ans6l, 'giao trinh')

ans6m, debug6m, _ = agent.answer('cac mon loai A', {})
assert debug6m.get('route') == 'policy_block'
assert debug6m.get('policy_code') == 'academic_detail_not_supported'
assert_contains(ans6m, 'hoc vu chi tiet')

ans6n, debug6n, _ = agent.answer('hach toan ke toan', {})
assert debug6n.get('route') == 'clarification'
assert debug6n.get('query_family', {}).get('detail') == 'academic_reference_needs_clarification'
assert_contains(ans6n, 'noi ro hon')

ans6o, debug6o, _ = agent.answer('lap trinh plc', {})
assert debug6o.get('route') == 'clarification'
assert debug6o.get('query_family', {}).get('detail') == 'academic_reference_needs_clarification'

ans6p, debug6p, _ = agent.answer('Co Than Thi Thuong', {})
assert debug6p.get('route') == 'clarification'
assert debug6p.get('query_family', {}).get('detail') == 'person_reference_needs_clarification'
assert_contains(ans6p, 'noi ro hon')

ans7, debug7, mem7 = agent.answer('phong dao tao o dau', {})
assert_contains(ans7, 'Phong 306, 308 H.A11')
assert_contains(ans7, 'Linh Nam')
assert_contains(ans7, 'Phong 104 H.A3')
assert_contains(ans7, '353 Tran Hung Dao')
assert_not_contains(ans7, 'Trong thoi gian toi')
assert debug7['plan']['domain'] == 'phong_ban_va_chuc_nang'

ans8, debug8, mem8 = agent.answer('khoa khoa hoc ung dung o dau', {})
assert_contains(ans8, '218 Linh Nam')
assert_contains(ans8, '353 Tran Hung Dao')
assert_not_contains(ans8, 'de tai nghien cuu')
assert debug8['plan']['domain'] == 'khoa_chuyen_mon'

ans8b, debug8b, mem8b = agent.answer('khoa khoa hoc ung dung day mon gi', {})
assert_contains(ans8b, 'Toan hoc')
assert_contains(ans8b, 'Logic hoc')
assert_contains(ans8b, 'Vat ly')
assert_contains(ans8b, 'Hoa hoc')
assert_not_contains(ans8b, 'Lich su hinh thanh')
assert debug8b['plan']['domain'] == 'khoa_chuyen_mon'
assert (debug8b.get('answer_verification') or {}).get('contexts')

ans8c, debug8c, mem8c = agent.answer('khoa khoa hoc ung dung dao tao gi', {})
assert_contains(ans8c, 'KHDL')
assert_contains(ans8c, '7460108')
assert_not_contains(ans8c, 'dieu chinh chuong trinh')

ans9, debug9, mem9 = agent.answer('dia chi khoa khoa hoc ung dung', {})
assert_contains(ans9, '218 Linh Nam')
assert_contains(ans9, '353 Tran Hung Dao')
assert debug9['plan']['domain'] == 'khoa_chuyen_mon'
assert mem9.get('last_named_unit')
assert mem9.get('last_assistant_answer')

ans10, debug10, mem10 = agent.answer('chuc nang phong khoa hoc cong nghe la gi', {})
assert_contains(ans10, 'tham muu giup viec cho Hieu truong')
assert_contains(ans10, 'khoa hoc va cong nghe')
assert debug10['plan']['domain'] == 'phong_ban_va_chuc_nang'

ans10b, debug10b, mem10b = agent.answer('noi ve phong cong nghe thong tin', {})
assert debug10b['plan']['domain'] == 'phong_ban_va_chuc_nang'
assert_contains(ans10b, 'Phong Cong nghe Thong tin')
assert_not_contains(ans10b, 'sv.phongdaotao@uneti.edu.vn')

ans11, debug11, mem11 = agent.answer('truong phong khoa hoc cong nghe la ai', {})
assert_contains(ans11, 'Nguyen Anh Tuan')

ans12, debug12, mem12 = agent.answer('email truong phong khoa hoc cong nghe', {})
assert_contains(ans12, 'natuan@uneti.edu.vn')

ans13, debug13, mem13 = agent.answer('phong khoa hoc cong nghe', {})
ans14, debug14, mem14 = agent.answer('truong phong la ai', mem13)
assert_contains(ans14, 'Nguyen Anh Tuan')
assert mem14.get('last_named_unit')

ans15, debug15, mem15 = agent.answer('chuc nang la gi', mem13)
assert_contains(ans15, 'tham muu giup viec cho Hieu truong')

ans16, debug16, mem16 = agent.answer('website khoa cong nghe thong tin', {})
assert_contains(ans16, 'khoacntt.uneti.edu.vn')

faculty_office, faculty_office_debug, _ = agent.answer('van phong khoa cntt o dau', {})
assert faculty_office_debug['plan']['domain'] == 'khoa_chuyen_mon'
assert_contains(faculty_office, 'khoacntt@uneti.edu.vn')
assert_not_contains(faculty_office, 'Phong cong nghe thong tin')

tourism_contact, tourism_debug, tourism_mem = agent.answer('khoa du lich o dau', {})
assert_contains(tourism_contact, 'Khoa Du lich va Khach san')
assert_contains(tourism_contact, '454 Minh Khai')
assert_not_contains(tourism_contact, 'Khoa Cong nghe thong tin')

cntt_contact, cntt_debug, cntt_mem = agent.answer('khoa cntt o dau', {})
tourism_after_cntt, tourism_after_cntt_debug, _ = agent.answer('khoa du lich o dau', cntt_mem)
assert tourism_after_cntt_debug['plan']['use_history'] is False
assert_contains(tourism_after_cntt, 'Khoa Du lich va Khach san')
assert_contains(tourism_after_cntt, '454 Minh Khai')
assert_not_contains(tourism_after_cntt, 'khoacntt.uneti.edu.vn')
assert_not_contains(tourism_after_cntt, 'Khoa Cong nghe thong tin')

ans17, debug17, mem17 = agent.answer('email truong khoa quan tri marketing', {})
assert_contains(ans17, 'lkcuong@uneti.edu.vn')

ans17b, debug17b, _ = agent.answer('co truong khoa thuong mai la ai', {})
assert_contains(ans17b, 'Nguyen Thi Chi')

ans17c, debug17c, _ = agent.answer('co truong khoa khoa hoc ung dung la ai', {})
assert_not_contains(ans17c, 'Khoa Cong nghe thong tin')
assert_contains(ans17c, 'chua thay thong tin')
assert_contains(ans17c, 'Truong khoa Khoa Hoc Ung Dung')

science_office, science_debug, science_mem = agent.answer('phong khoa hoc cong nghe', {})
dao_tao_after_science, dao_tao_after_science_debug, _ = agent.answer('phong dao tao o dau', science_mem)
assert dao_tao_after_science_debug['plan']['use_history'] is False
assert_contains(dao_tao_after_science, '353 Tran Hung Dao')
assert_not_contains(dao_tao_after_science, 'Nguyen Anh Tuan')

marketing_after_cntt, marketing_after_cntt_debug, _ = agent.answer('email truong khoa quan tri marketing', cntt_mem)
assert marketing_after_cntt_debug['plan']['use_history'] is False
assert_contains(marketing_after_cntt, 'lkcuong@uneti.edu.vn')
assert_not_contains(marketing_after_cntt, 'khoacntt.uneti.edu.vn')

follow_mem = {
    'active_domain': 'phong_ban_va_chuc_nang',
    'last_user_query': 'phong khoa hoc cong nghe',
    'last_assistant_answer': 'Phong Khoa hoc Cong nghe',
    'last_named_unit': 'phong khoa hoc cong nghe',
    'context_turns': 0,
}
follow_answer, follow_debug, _ = agent.answer('email', follow_mem)
assert follow_debug['plan']['use_history'] is False

ans18, debug18, mem18 = agent.answer('mo cong thong tin sinh vien o dau', {})
assert_contains(ans18, 'sinhvien.uneti.edu.vn')

ans18b, debug18b, mem18b = agent.answer('dang nhap cong thong tin sinh vien nhu the nao', {})
assert_contains(ans18b, 'Username la ma sinh vien')
assert_contains(ans18b, 'nhan nut dang nhap')

ans19, debug19, mem19 = agent.answer('xem thong tin sinh vien o dau', {})
assert_contains(ans19, 'Thong tin sinh vien')
assert_contains(ans19, 'ho so sinh vien')

ans19b, debug19b, mem19b = agent.answer('xem lich hoc hoac lich thi theo tuan o dau', {})
assert_contains(ans19b, 'Dashboard -> Hoc tap -> Lich theo tuan')
assert_contains(ans19b, 'tuan hien tai')

ans19c, debug19c, mem19c = agent.answer('xem lich theo tien do o dau', {})
assert_contains(ans19c, 'Dashboard -> Hoc tap -> Lich theo tien do')
assert_contains(ans19c, 'chon dot/hoc ky can xem')

ans20, debug20, mem20 = agent.answer('xem ket qua hoc tap o dau', {})
assert_contains(ans20, 'Ket qua hoc tap')

ans20a, debug20a, mem20a = agent.answer('xem chuong trinh khung o dau', {})
assert_contains(ans20a, 'Dashboard -> Dang ky hoc phan -> Chuong trinh khung')
assert_contains(ans20a, 'toan bo chuong trinh hoc')

ans20b, debug20b, mem20b = agent.answer('cach dang ky hoc phan', {})
assert_contains(ans20b, 'Dashboard -> Dang ky hoc phan -> Dang ky hoc phan')
assert_contains(ans20b, 'chon tien do')
assert_contains(ans20b, 'chon lop hoc phan')

ans21, debug21, mem21 = agent.answer('dang ky hoc phan tren cong sinh vien nhu the nao', {})
assert_contains(ans21, 'Dashboard -> Dang ky hoc phan -> Dang ky hoc phan')
assert_contains(ans21, 'chon tien do')
assert_contains(ans21, 'chon lop hoc phan')

ans22, debug22, mem22 = agent.answer('dang ky thi lai nhu the nao', {})
assert_contains(ans22, 'Dashboard -> Dang ky hoc phan -> Dang ky thi lai')
assert_contains(ans22, 'chon mon hoc phan')
assert_contains(ans22, 'xac nhan dang ky thi lai')

ans23, debug23, mem23 = agent.answer('phong ctsv o ha noi o dau', {})
assert_contains(ans23, '454 Minh Khai')
assert_contains(ans23, '218 Linh Nam')
assert debug23['plan']['domain'] == 'phong_ban_va_chuc_nang'

ans24, debug24, mem24 = agent.answer('hanh chinh mot cua lam viec khi nao', {})
assert_contains(ans24, '8h00-12h00')
assert_contains(ans24, '13h00-17h00')

ans25, debug25, mem25 = agent.answer('sinh vien can ho tro ky thuat tai khoan hoac loi he thong thi lien he dau', {})
assert_contains(ans25, 'sv.phongcntt@uneti.edu.vn')
assert debug25['plan']['domain'] == 'phong_ban_va_chuc_nang'

ans26, debug26, mem26 = agent.answer('khoa co khi dao tao nhung nganh nao', {})
assert_contains(ans26, 'Cong nghe ky thuat Co khi')
assert_contains(ans26, 'Cong nghe ky thuat Co dien tu')
assert_contains(ans26, 'Cong nghe ky thuat O to')

ans27, debug27, mem27 = agent.answer('truong phong dao tao la ai', {})
assert_contains(ans27, 'Hoang Anh Tuan')
assert_contains(ans27, 'hatuan@uneti.edu.vn')

fee_memories = [
    {},
    {'active_domain': 'co_so_vat_chat', 'last_user_query': 'dia chi cac co so', 'last_assistant_answer': 'Truong co 4 dia diem dao tao...'},
    {'active_domain': 'phong_ban_va_chuc_nang', 'last_user_query': 'phong ctsv o dau', 'last_assistant_answer': 'Phong CTSV o 454 Minh Khai...', 'last_named_unit': 'phong cong tac sinh vien'},
    {'active_domain': 'portal_howto', 'last_user_query': 'dang nhap cong sv', 'last_assistant_answer': 'vao sinhvien.uneti.edu.vn', 'last_topic': 'login'},
]
for fee_memory in fee_memories:
    fee_answer, fee_debug, _ = agent.answer('cach nop hoc phi', fee_memory)
    assert fee_debug['plan']['domain'] == 'portal_howto'
    assert_contains(fee_answer, 'Tra cuu cong no')
    assert_contains(fee_answer, 'Phiếu thu tổng hợp')
    assert_not_contains(fee_answer, '454 Minh Khai')
    assert_not_contains(fee_answer, '4 dia diem dao tao')

ans28, debug28, mem28 = agent.answer('hoc phi', {})
assert debug28['plan']['domain'] == 'portal_howto'
assert_contains(ans28, 'Tra cuu cong no')

ans28b, debug28b, mem28b = agent.answer('tra cuu cong no o dau', {})
assert_contains(ans28b, 'Dashboard -> Hoc phi -> Tra cuu cong no')
assert_contains(ans28b, 'chon hoc ky can xem')

ans29, debug29, mem29 = agent.answer('dong hoc phi nhu the nao', {'active_domain': 'co_so_vat_chat', 'last_user_query': 'co so ha noi', 'last_assistant_answer': '454 Minh Khai'})
assert debug29['plan']['domain'] == 'portal_howto'
assert_contains(ans29, 'Tra cuu cong no')
assert_contains(ans29, 'Phieu thu tong hop')

ans30, debug30, mem30 = agent.answer('hieu truong la ai', {'active_domain': 'portal_howto', 'last_user_query': 'dang nhap cong sv', 'last_assistant_answer': 'vao sinhvien.uneti.edu.vn', 'last_topic': 'login'})
assert debug30['plan']['domain'] == 'ban_giam_hieu'
assert_contains(ans30, 'Tran Hoang Long')

ans31, debug31, mem31 = agent.answer('phong dao tao o dau', {'active_domain': 'portal_howto', 'last_user_query': 'hoc phi', 'last_assistant_answer': 'Tra cuu cong no'})
assert debug31['plan']['domain'] == 'phong_ban_va_chuc_nang'
assert_contains(ans31, '353 Tran Hung Dao')

deep1, deep1_debug, _ = agent.answer('ngay sinh hieu truong la gi', {})
assert deep1_debug.get('route') == 'partial_scope'
assert_contains(deep1, 'Hieu truong')
assert_contains(deep1, 'Tran Hoang Long')
assert_contains(deep1, 'ngay sinh')
assert_contains(deep1, 'khong co trong tri thuc')

deep2, deep2_debug, _ = agent.answer('so dien thoai hieu truong', {})
assert deep2_debug.get('route') == 'partial_scope'
assert_contains(deep2, 'Tran Hoang Long')
assert_contains(deep2, 'so dien thoai ca nhan')
assert_contains(deep2, 'thong tin ca nhan rieng tu')

deep3, deep3_debug, _ = agent.answer('ngay sinh truong phong dao tao', {})
assert deep3_debug.get('route') == 'partial_scope'
assert_contains(deep3, 'Hoang Anh Tuan')
assert_contains(deep3, 'ngay sinh')
assert_contains(deep3, 'khong co trong tri thuc')

deep4, deep4_debug, _ = agent.answer('tuoi truong khoa thuong mai', {})
assert deep4_debug.get('route') == 'partial_scope'
assert_contains(deep4, 'Nguyen Thi Chi')
assert_contains(deep4, 'tuoi')
assert_contains(deep4, 'khong co trong tri thuc')

assert should_offer_contribution(
    'Toi chua thay thong tin ve Truong khoa Khoa Hoc Ung Dung trong tri thuc hien co.',
    {'route': ''},
) is True
assert should_offer_contribution(
    'Xin loi, day la thong tin ca nhan rieng tu nen chatbot khong cung cap.',
    {'route': 'policy_block'},
) is False
assert classify_contribution_reply('Truong khoa la TS. Nguyen Van A, email nva@uneti.edu.vn.') == 'provide'
assert classify_contribution_reply('toi khong biet') == 'decline'
assert classify_contribution_reply('ban la ai') == 'other'
assert_contains(contribution_invitation('Toi chua thay thong tin trong tri thuc hien co.'), 'neu ban co thong tin chinh xac')
assert_contains(contribution_thanks_reply(), 'Cam on')
assert_contains(contribution_decline_reply(), 'Khong sao')

contribution_backup = CONTRIBUTION_DB_PATH.read_text(encoding='utf-8') if CONTRIBUTION_DB_PATH.exists() else ''
CONTRIBUTION_DB_PATH.write_text(json.dumps({'items': []}, ensure_ascii=False, indent=2), encoding='utf-8')
contribution = submit_contribution({
    'student_id': 'test',
    'original_question': 'co truong khoa khoa hoc ung dung la ai',
    'bot_answer': 'Toi chua thay thong tin trong tri thuc hien co.',
    'user_contribution': 'Truong khoa la TS. Nguyen Van A.',
    'domain_hint': 'khoa_chuyen_mon',
    'route': 'user_contribution',
})
assert contribution.get('status') == 'pending'
updated_contribution = update_contribution(contribution.get('contribution_id', ''), status='approved', admin_note='ok')
assert updated_contribution.get('status') == 'approved'
assert load_contribution_db().get('items', [])[0].get('admin_note') == 'ok'

assert is_feedback_eligible({'route': 'partial_scope', 'plan': {'domain': 'khoa_chuyen_mon'}}) is True
assert is_feedback_eligible({'route': 'out_of_scope'}) is False
feedback_backup = FEEDBACK_DB_PATH.read_text(encoding='utf-8') if FEEDBACK_DB_PATH.exists() else ''
FEEDBACK_DB_PATH.write_text(json.dumps({'items': []}, ensure_ascii=False, indent=2), encoding='utf-8')
feedback_item = upsert_feedback({
    'turn_id': 'test-feedback:1',
    'student_id': 'test',
    'assistant_index': 1,
    'question': 'phong dao tao o dau',
    'answer': 'Phong dao tao o 353 Tran Hung Dao.',
    'rating': 2,
    'answer_suggestion': 'Can them co so Ha Noi.',
    'notes': 'Thieu 1 co so',
    'route': 'docs',
    'domain': 'phong_ban_va_chuc_nang',
})
assert feedback_item.get('rating') == 2
assert get_feedback('test-feedback:1').get('answer_suggestion') == 'Can them co so Ha Noi.'
assert load_feedback_db().get('items', [])[0].get('domain') == 'phong_ban_va_chuc_nang'

history_test_path = HISTORY_DIR / 'rlhf_test_user.json'
history_backup = history_test_path.read_text(encoding='utf-8') if history_test_path.exists() else ''
history_test_path.write_text(json.dumps({
    'messages': [
        {'role': 'user', 'content': 'phong dao tao o dau'},
        {'role': 'assistant', 'content': 'Sai dia chi.', 'debug': json.dumps({'route': 'docs', 'plan': {'domain': 'phong_ban_va_chuc_nang'}}, ensure_ascii=False)},
    ],
    'memory': {},
}, ensure_ascii=False, indent=2), encoding='utf-8')
REVIEW_DB_PATH.write_text(json.dumps({
    'reviews': [{
        'turn_id': 'rlhf_test_user:1',
        'student_id': 'rlhf_test_user',
        'assistant_index': 1,
        'user_question': 'phong dao tao o dau',
        'assistant_answer': 'Sai dia chi.',
        'verdict': 'incorrect',
        'review_kind': 'answer',
        'policy_code': '',
        'approved_answer': 'Phong Đào tạo có cơ sở Hà Nội và Nam Định.',
        'retrieval_hint': '',
        'domain_hint': 'phong_ban_va_chuc_nang',
        'notes': '',
        'active': True,
        'match_questions': ['phong dao tao o dau'],
        'auto_match_questions': [],
        'usage_count': 0,
        'last_used_at': '',
        'updated_at': '2026-04-09T00:00:00Z',
        'created_at': '2026-04-09T00:00:00Z',
    }]
}, ensure_ascii=False, indent=2), encoding='utf-8')
feedback_item = upsert_feedback({
    'turn_id': 'rlhf_test_user:1',
    'student_id': 'rlhf_test_user',
    'assistant_index': 1,
    'question': 'phong dao tao o dau',
    'answer': 'Sai dia chi.',
    'rating': 1,
    'answer_suggestion': 'Cần nêu đủ cả hai cơ sở.',
    'notes': '',
    'route': 'docs',
    'domain': 'phong_ban_va_chuc_nang',
})
summary = build_rlhf_dataset(output_prefix='test_rlhf')
assert summary.get('preference_rows', 0) >= 1
pref_path = RLHF_DIR / 'test_rlhf_preferences.jsonl'
assert pref_path.exists()
pref_lines = [json.loads(line) for line in pref_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert any(norm_text_ascii(row.get('question', '')) == norm_text_ascii('phong dao tao o dau') for row in pref_lines)
assert any(norm_text_ascii(row.get('chosen', '')) == norm_text_ascii('Phong Đào tạo có cơ sở Hà Nội và Nam Định.') for row in pref_lines)
assert any(row.get('meta', {}).get('domain') == 'phong_ban_va_chuc_nang' for row in pref_lines)
assert any('incorrect' in (row.get('meta', {}).get('review_verdicts') or []) for row in pref_lines)
assert any(int(row.get('meta', {}).get('feedback_rating', 0) or 0) == 1 for row in pref_lines)
export_summary = export_rlhf_formats(
    dataset_prefix='test_rlhf',
    export_prefix='test_rlhf_export',
    frameworks=['trl', 'axolotl', 'unsloth'],
    domains=['phong_ban_va_chuc_nang'],
    verdicts=['incorrect'],
    min_rating=1,
    max_rating=2,
    val_ratio=0.0,
)
assert export_summary.get('source_counts', {}).get('preferences', 0) >= 1
trl_pref_path = RLHF_DIR / 'exports' / 'test_rlhf_export' / 'trl' / 'preference_train.jsonl'
axolotl_sft_path = RLHF_DIR / 'exports' / 'test_rlhf_export' / 'axolotl' / 'sft_train.jsonl'
unsloth_sft_path = RLHF_DIR / 'exports' / 'test_rlhf_export' / 'unsloth' / 'sft_train.jsonl'
assert trl_pref_path.exists()
assert axolotl_sft_path.exists()
assert unsloth_sft_path.exists()
trl_pref_rows = [json.loads(line) for line in trl_pref_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert any(norm_text_ascii(row.get('prompt', '')) == norm_text_ascii('phong dao tao o dau') for row in trl_pref_rows)
assert any(row.get('meta', {}).get('chosen_source') == 'review' for row in trl_pref_rows)
axolotl_sft_rows = [json.loads(line) for line in axolotl_sft_path.read_text(encoding='utf-8').splitlines() if line.strip()]
assert any(norm_text_ascii(row.get('instruction', '')) == norm_text_ascii('phong dao tao o dau') for row in axolotl_sft_rows)
assert any(row.get('meta', {}).get('domain') == 'phong_ban_va_chuc_nang' for row in axolotl_sft_rows)
reward_model_path = RLHF_DIR / 'test_reward_model.pkl'
reward_summary = train_reward_model(pref_path, reward_model_path)
assert reward_summary.get('example_count', 0) >= 2
reward_scorer = load_reward_model(str(reward_model_path))
assert reward_scorer.score('phong dao tao o dau', 'Phong Dao tao co co so Ha Noi va Nam Dinh.') > reward_scorer.score('phong dao tao o dau', 'Sai dia chi.')
ppo_summary = export_ppo_dataset(dataset_prefix='test_rlhf', output_prefix='test_rlhf_ppo', build_if_missing=False)
ppo_path = RLHF_DIR / 'test_rlhf_ppo_ppo_prompts.jsonl'
assert ppo_path.exists()
assert ppo_summary.get('prompt_rows', 0) >= 1
ppo_check = ppo_dry_run(ppo_path, reward_model_path)
assert ppo_check.get('rows', 0) >= 1
if contribution_backup:
    CONTRIBUTION_DB_PATH.write_text(contribution_backup, encoding='utf-8')
elif CONTRIBUTION_DB_PATH.exists():
    CONTRIBUTION_DB_PATH.unlink()
if feedback_backup:
    FEEDBACK_DB_PATH.write_text(feedback_backup, encoding='utf-8')
elif FEEDBACK_DB_PATH.exists():
    FEEDBACK_DB_PATH.unlink()
if history_backup:
    history_test_path.write_text(history_backup, encoding='utf-8')
elif history_test_path.exists():
    history_test_path.unlink()
for suffix in ['candidates', 'preferences', 'sft', 'summary']:
    rlhf_path = RLHF_DIR / f'test_rlhf_{suffix}.jsonl'
    if suffix == 'summary':
        rlhf_path = RLHF_DIR / 'test_rlhf_summary.json'
    if rlhf_path.exists():
        rlhf_path.unlink()
export_root = RLHF_DIR / 'exports' / 'test_rlhf_export'
if export_root.exists():
    shutil.rmtree(export_root)
for path in [
    reward_model_path,
    reward_model_path.with_suffix('.meta.json'),
    RLHF_DIR / 'test_rlhf_ppo_ppo_prompts.jsonl',
    RLHF_DIR / 'test_rlhf_ppo_ppo_summary.json',
]:
    if path.exists():
        path.unlink()

review_backup = REVIEW_DB_PATH.read_text(encoding='utf-8') if REVIEW_DB_PATH.exists() else ''
try:
    REVIEW_DB_PATH.write_text(json.dumps({
        'reviews': [{
            'turn_id': 'test:1',
            'student_id': 'test',
            'assistant_index': 1,
            'user_question': 'cau hoi admin override',
            'assistant_answer': 'old answer',
            'verdict': 'correct',
            'review_kind': 'answer',
            'policy_code': '',
            'approved_answer': 'Đáp án chuẩn từ admin review.',
            'retrieval_hint': '',
            'domain_hint': 'portal_howto',
            'notes': '',
            'active': True,
            'match_questions': ['cau hoi admin override'],
            'updated_at': '2026-04-08T00:00:00Z',
            'created_at': '2026-04-08T00:00:00Z',
        }]
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    ans32, debug32, mem32 = agent.answer('cau hoi admin override', {})
    assert_contains(ans32, 'Đáp án chuẩn từ admin review')
    assert debug32.get('route') == 'admin_review_override'

    REVIEW_DB_PATH.write_text(json.dumps({
        'reviews': [{
            'turn_id': 'policy:1',
            'student_id': 'admin',
            'assistant_index': 1,
            'user_question': 'so dien thoai sinh vien cntt',
            'assistant_answer': 'old policy answer',
            'verdict': 'correct',
            'review_kind': 'policy',
            'policy_code': 'student_private_info',
            'approved_answer': 'Xin lỗi, đây là thông tin riêng tư của sinh viên nên chatbot không được phép cung cấp.',
            'retrieval_hint': '',
            'domain_hint': '',
            'notes': '',
            'active': True,
            'match_questions': ['so dien thoai sinh vien cntt'],
            'updated_at': '2026-04-08T00:00:00Z',
            'created_at': '2026-04-08T00:00:00Z',
        }]
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    ans33, debug33, mem33 = agent.answer('so dien thoai sinh vien ke toan', {})
    assert debug33.get('route') == 'sensitive_block'
    assert debug33.get('policy_code') == 'student_private_info'
    assert_contains(ans33, 'thông tin riêng tư của sinh viên')
    saved_policy = json.loads(REVIEW_DB_PATH.read_text(encoding='utf-8'))['reviews'][0]
    assert saved_policy.get('usage_count', 0) >= 1
    assert any(norm_text_ascii(x) == norm_text_ascii('so dien thoai sinh vien ke toan') for x in saved_policy.get('auto_match_questions', []))
finally:
    if review_backup:
        REVIEW_DB_PATH.write_text(review_backup, encoding='utf-8')
    elif REVIEW_DB_PATH.exists():
        REVIEW_DB_PATH.unlink()

vector_root = INDEX_DIR / 'vectors'
assert vector_root.exists(), str(vector_root)

if initial_review_backup:
    REVIEW_DB_PATH.write_text(initial_review_backup, encoding='utf-8')
elif REVIEW_DB_PATH.exists():
    REVIEW_DB_PATH.unlink()

if contribution_backup:
    CONTRIBUTION_DB_PATH.write_text(contribution_backup, encoding='utf-8')
elif CONTRIBUTION_DB_PATH.exists():
    CONTRIBUTION_DB_PATH.unlink()
