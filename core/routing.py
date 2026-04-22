from __future__ import annotations

import re
from typing import Dict, List, Optional

from .normalize import norm_text_ascii
from .seed_loader import load_seed_knowledge

CONTROL_WORDS = {'hết', 'dừng', 'stop'}
GREETING_PHRASES = {
    'xin chao', 'chao', 'chao ban', 'chao bot', 'hello', 'hi', 'hey', 'alo',
    'xin chao ban', 'xin chao bot', 'chao anh', 'chao chi', 'hello bot',
}
GREETING_TOKENS = {
    'xin', 'chao', 'hello', 'hi', 'hey', 'alo', 'ban', 'bot', 'anh', 'chi', 'em', 'oi', 'nhe', 'a', 'ah',
}
BOT_IDENTITY_HINTS = [
    'ban la ai', 'bot la ai', 'chatbot la ai', 'gioi thieu ve ban', 'gioi thieu ban than',
    'ban ten gi', 'ten ban la gi', 'what your name', 'what you name', 'who are you',
]
BOT_CAPABILITY_HINTS = [
    'ban lam duoc gi', 'ban co the lam gi', 'ban giup duoc gi', 'ban ho tro gi',
    'chatbot lam duoc gi', 'chatbot ho tro gi', 'co the hoi gi', 'hoi duoc gi',
    'ho tro gi', 'lam duoc gi', 'giup duoc gi', 'may ho tro gi', 'sao may ho tro gi',
    'sao ban ho tro gi', 'bot ho tro gi', 'uneti ho tro gi', 'ban giup gi duoc cho toi', 'ban giup gi duoc',
    'ban co the tra loi gi', 'tra loi gi', 'tra loi duoc gi', 'nhung cau hoi ve linh vuc gi',
    'tra loi nhung gi', 'ban tra loi gi',
]
BOT_HELP_HINTS = [
    'huong dan toi hoi', 'goi y toi hoi', 'toi nen hoi gi', 'bat dau tu dau',
    'giup toi voi', 'ho tro toi voi', 'chi toi cach hoi',
]
THANKS_HINTS = ['cam on', 'cảm ơn', 'thanks', 'thank you', 'cam on ban', 'cam on bot']
FAREWELL_HINTS = ['tam biet', 'bye', 'goodbye', 'hen gap lai', 'chao tam biet', 'bye bot']
BOT_COMPLAINT_TARGETS = [
    'ban', 'may', 'bot', 'chatbot', 'tro ly', 'cau tra loi', 'dap an', 'phan hoi',
    'he thong', 'app', 'ung dung', 'san pham', 'tinh nang', 'chuc nang', 'search', 'tim kiem',
]
BOT_COMPLAINT_ISSUES = [
    'khong hai long', 'khong vua y', 'te', 'kem', 'chan', 'khong tot', 'khong on',
    'khong dung', 'khong chinh xac', 'sai', 'noi sai', 'tra loi sai', 'linh tinh',
    'khong giup duoc', 'khong huu ich', 'vo ich', 'khong biet', 'khong hieu',
    'khong tim duoc', 'tim khong ra', 'search khong ra', 'loi', 'bi loi',
    'phan nan', 'gop y', 'feedback', 'ngu', 'dan don', 'do dot',
]
BOT_COMPLAINT_DIRECT_HINTS = [
    'sao ban lai khong biet', 'sao ban khong biet', 'ban khong biet gi',
    'sao may lai khong biet', 'sao may khong biet', 'may khong biet gi',
    'sao m lai khong biet', 'sao m khong biet', 'm khong biet gi',
    'sao ban ngu', 'sao may ngu', 'sao m ngu', 'ban ngu', 'may ngu', 'm ngu',
    'bot ngu', 'chatbot ngu', 'bot te', 'chatbot te', 'tro ly te',
    'tra loi linh tinh', 'tra loi qua te', 'cau tra loi sai', 'dap an sai',
    'toi khong hai long ve chatbot', 'toi khong hai long ve bot',
    'khong hai long ve chatbot', 'khong hai long ve bot',
    'phan nan ve chatbot', 'phan nan ve bot', 'gop y ve chatbot', 'gop y ve bot',
]
BOT_COMPLAINT_STANDALONE_INSULTS = {
    'ngu', 'ngu the', 'ngu vay', 'qua ngu', 'dan don', 'do dot', 'te qua', 'chan qua',
}
ACK_HINTS = ['ok', 'oke', 'okela', 'duoc roi', 'ro roi', 'hieu roi']
CLARIFICATION_HINTS = ['lo vay', 'la vay', 'gi vay', 'sao vay', 'ui vay', 'u la vay']
HARMFUL_HINTS = ['dot truong', 'pha truong', 'danh bom', 'giet', 'tan cong', 'dot phong', 'doa giang vien']
SELF_PROFILE_HINTS = [
    'toi hoc lop nao', 'em hoc lop nao', 'minh hoc lop nao', 'lop cua toi', 'lop cua em',
    'toi hoc khoa nao', 'em hoc khoa nao', 'nganh cua toi', 'ma sinh vien cua toi', 'ho so cua toi',
]
TEACHING_ASSIGNMENT_HINTS = [
    'do ai day', 'ai day mon', 'mon nao do ai day', 'giang vien nao day', 'co nao day mon',
    'thay nao day mon', 'lich day mon', 'phan cong giang day', 'mon nay ai phu trach',
]
ACADEMIC_MATERIAL_HINTS = [
    'giao trinh', 'de cuong', 'tai lieu mon', 'hoc lieu', 'slide bai giang', 'tai lieu hoc tap',
]
ACADEMIC_DETAIL_HINTS = [
    'mon loai a', 'mon loai b', 'mon loai c', 'cac mon loai a', 'cac mon loai b', 'cac mon loai c',
    'hoc phan loai a', 'hoc phan loai b',
]
ACADEMIC_TERM_TOKENS = {
    'hach', 'toan', 'ke', 'toan', 'lap', 'trinh', 'plc', 'xac', 'suat', 'thong', 'ke',
    'mon', 'hoc', 'phan', 'giao', 'trinh', 'de', 'cuong', 'tai', 'lieu',
}
PERSON_TITLE_TOKENS = {'co', 'thay', 'giang', 'vien', 'ts', 'ths', 'pgs', 'pgs.ts'}
QUESTION_INTENT_PHRASES = [
    'la ai', 'la gi', 'o dau', 'email', 'website', 'web', 'link', 'sdt', 'so dien thoai',
    'nhu the nao', 'cach', 'lam the nao', 'dang ky', 'xem', 'tra cuu', 'bao nhieu',
    'liet ke', 'tom tat', 'gioi thieu', 'noi ve', 'co khong',
]
SCHOOL_OVERVIEW_HINTS = [
    'gioi thieu ve truong', 'gioi thieu ve uneti', 'noi ve truong', 'noi ve uneti',
    'thong tin ve truong', 'tong quan ve truong', 'truong uneti la truong nao',
]
DOMAIN_HINTS = {
    'ban_giam_hieu': ['hieu truong', 'pho hieu truong', 'hieu pho', 'ban giam hieu', 'lanh dao', 'nguoi dung dau', 'lanh dao cao nhat'],
    'co_so_vat_chat': ['co so', 'dia chi', 'dia diem dao tao', 'minh khai', 'linh nam', 'tran hung dao', 'my xa', 'ha noi', 'nam dinh'],
    'lich_su_hinh_thanh': ['lich su', 'tien than', 'thanh lap', 'giai doan phat trien'],
    'hoi_dong_truong': ['hoi dong truong', 'chu tich hdt', 'chu tich hoi dong truong', 'thanh vien hoi dong truong'],
    'portal_howto': ['cong sinh vien', 'dang ky hoc phan', 'dkhp', 'lich hoc', 'lich thi', 'doi mat khau', 'cong no', 'phieu thu', 'hoc phi', 'nop hoc phi', 'dong hoc phi', 'thanh toan hoc phi', 'dang nhap', 'hoc tap', 'thi lai', 'thong tin sinh vien', 'nhac nho', 'ket qua hoc tap', 'ket qua ren luyen', 'chuong trinh khung', 'diem danh', 'lich toan truong', 'lich theo tien do', 'lich theo tuan'],
    'khoa_chuyen_mon': ['khoa ', 'truong khoa', 'pho truong khoa', 'cntt', 'co khi', 'thuong mai', 'du lich', 'cong nghe thuc pham', 'quan tri marketing', 'ngoai ngu', 'khoa hoc ung dung'],
    'phong_ban_va_chuc_nang': ['phong ', 'chuc nang', 'phong cong nghe thong tin', 'phong to chuc can bo', 'phong khoa hoc', 'phong dao tao', 'ctsv', 'cong tac sinh vien', 'hanh chinh mot cua', 'mot cua', 'phong tai chinh ke toan', 'phong cntt', 'trung tam ngoai ngu va tin hoc', 'ho tro ky thuat', 'loi he thong', 'tai khoan'],
}
FOLLOW_UP_HINTS = ['nguoi do', 'nguoi ay', 'co so do', 'don vi do', 'khoa do', 'phong do', 'email cua nguoi do', 'dia chi do', 'con lai', 'nguoi nay', 'co so nay']
GENERIC_FOLLOW_UPS = {'email', 'sdt', 'so dien thoai', 'dia chi', 'o dau', 'con lai', 'nguoi do'}
QUERY_ALIASES = {
    'nguoi dung dau': ['hieu truong'],
    'lanh dao cao nhat': ['hieu truong'],
    'ban lanh dao': ['ban giam hieu'],
    'campus': ['co so', 'dia diem dao tao'],
    'xuong thuc hanh': ['my xa'],
    'dang ky mon': ['dang ky hoc phan'],
    'dkhp': ['dang ky hoc phan'],
    'portal': ['cong sinh vien'],
    'ctsv': ['cong tac sinh vien'],
    'tccb': ['to chuc can bo'],
    'dbcl': ['dam bao chat luong'],
    'tckt': ['tai chinh ke toan'],
    'cntt': ['cong nghe thong tin'],
    'nop hoc phi': ['hoc phi', 'tra cuu cong no', 'phieu thu tong hop', 'dashboard hoc phi'],
    'dong hoc phi': ['hoc phi', 'tra cuu cong no', 'phieu thu tong hop', 'dashboard hoc phi'],
    'thanh toan hoc phi': ['hoc phi', 'tra cuu cong no', 'phieu thu tong hop', 'dashboard hoc phi'],
}
PORTAL_TOPICS = {
    'homepage': ['trang chu', 'trang chủ', 'website sinh vien', 'cong thong tin sinh vien', 'mo cong thong tin sinh vien', 'vao cong thong tin sinh vien'],
    'login': ['dang nhap', 'login', 'tai khoan', 'username', 'mat khau'],
    'change_password': ['doi mat khau', 'doi password', 'mat khau moi'],
    'student_info': ['xem thong tin sinh vien', 'ho so sinh vien', 'thong tin ca nhan'],
    'schedule_week': ['lich hoc', 'lich thi', 'lich theo tuan'],
    'schedule_progress': ['lich theo tien do'],
    'schedule_global': ['lich toan truong'],
    'reminders': ['nhac nho', 'thong bao cho sinh vien'],
    'study_results': ['ket qua hoc tap', 'bang diem'],
    'conduct_results': ['ket qua ren luyen', 'diem ren luyen'],
    'attendance': ['diem danh'],
    'curriculum': ['chuong trinh khung'],
    'course_registration': ['dang ky hoc phan', 'dkhp', 'dang ky mon hoc phan'],
    'cancel_registration': ['huy lop hoc phan', 'huy dang ky hoc phan'],
    'conditional_course': ['mon hoc dieu kien'],
    'retake_registration': ['thi lai', 'dang ky thi lai'],
    'debt_lookup': ['cong no', 'hoc phi', 'nop hoc phi', 'dong hoc phi', 'thanh toan hoc phi'],
    'receipt': ['phieu thu'],
}
PORTAL_TOPIC_EXPANSIONS = {
    'homepage': ['trang chu', 'cong thong tin sinh vien', 'sinhvien.uneti.edu.vn'],
    'login': ['dang nhap', 'username', 'mat khau'],
    'change_password': ['doi mat khau'],
    'student_info': ['xem thong tin sinh vien', 'ho so sinh vien'],
    'schedule_week': ['lich hoc', 'lich thi', 'lich theo tuan'],
    'schedule_progress': ['lich theo tien do'],
    'schedule_global': ['lich toan truong'],
    'reminders': ['nhac nho', 'thong bao'],
    'study_results': ['ket qua hoc tap', 'bang diem'],
    'conduct_results': ['ket qua ren luyen', 'diem ren luyen'],
    'attendance': ['diem danh'],
    'curriculum': ['chuong trinh khung'],
    'course_registration': ['dang ky hoc phan', 'dkhp'],
    'cancel_registration': ['huy dang ky hoc phan'],
    'conditional_course': ['mon hoc dieu kien'],
    'retake_registration': ['dang ky thi lai', 'thi lai'],
    'debt_lookup': ['cong no', 'hoc phi', 'nop hoc phi', 'dong hoc phi', 'thanh toan hoc phi'],
    'receipt': ['phieu thu tong hop', 'phieu thu'],
}
WEB_NOTICE_HINTS = ['thong bao', 'tin tuc', 'su kien', 'lich cong tac', 'tuyen sinh', 'hoc bong', 'diem san', 'xet tuyen', 'tuyen dung']
WEB_NOTICE_TIME_HINTS = ['moi nhat', 'gan day', 'hom nay']
EXPLICIT_QUERY_ANCHORS = [
    'phong dao tao', 'phong tai chinh ke toan', 'phong cong tac sinh vien', 'ctsv',
    'hanh chinh mot cua', 'phong cntt', 'trung tam ngoai ngu va tin hoc',
    'hieu truong', 'pho hieu truong', 'ban giam hieu', 'hoi dong truong',
    'co so', 'minh khai', 'linh nam', 'tran hung dao', 'my xa', 'ha noi', 'nam dinh',
    'cong sinh vien', 'dang ky hoc phan', 'lich hoc', 'lich thi', 'cong no', 'hoc phi', 'phieu thu',
    'thi lai', 'ket qua hoc tap', 'ket qua ren luyen', 'thong tin sinh vien',
]
UNETI_SCOPE_HINTS = [
    'uneti', 'truong', 'nha truong', 'sinh vien', 'hoc vien', 'giang vien',
    'phong ', 'khoa ', 'ban giam hieu', 'hoi dong truong', 'hieu truong', 'pho hieu truong',
    'co so', 'dia diem dao tao', 'minh khai', 'linh nam', 'tran hung dao', 'my xa', 'ha noi', 'nam dinh',
    'cong thong tin sinh vien', 'cong sinh vien', 'dang ky hoc phan', 'hoc phi', 'cong no', 'phieu thu',
    'lich hoc', 'lich thi', 'thi lai', 'ket qua hoc tap', 'ket qua ren luyen',
    'ctsv', 'dao tao', 'to chuc can bo', 'khoa hoc cong nghe', 'hanh chinh mot cua',
]
OUT_OF_SCOPE_HINTS = [
    'thoi tiet', 'du bao thoi tiet', 'gia vang', 'gia usd', 'ty gia', 'bitcoin', 'crypto', 'chung khoan',
    'bong da', 'bong ro', 'ket qua tran dau', 'ca si', 'dien vien', 'phim', 'am nhac',
    'thu do', 'tong thong', 'thu tuong', 'lich su viet nam', 'toan hoc', 'lap trinh', 'python', 'java',
    'benh', 'trieu chung', 'thuoc', 'nau an', 'du lich', 'mua hang',
]
SENSITIVE_HINTS = [
    'ten sinh vien', 'danh sach sinh vien', 'mssv', 'ma sinh vien cua', 'thong tin sinh vien cua',
    'moi lop bao nhieu sinh vien', 'bao nhieu sinh vien moi lop', 'si so tung lop', 'si so lop',
    'moi khoa co bao nhieu lop', 'bao nhieu lop moi khoa', 'danh sach lop', 'lop nao co bao nhieu sinh vien',
]
CLASS_CODE_RE = re.compile(r'\b(?:dh|cd|tc)[a-z]{2,6}\d{2}[a-z]\d+(?:hn|nd)\b')
STUDENT_CODE_RE = re.compile(r'\b\d{8,12}\b')
GRADE_HINTS = ['diem', 'bang diem', 'ket qua hoc tap cua', 'ket qua hoc tap', 'diem thi', 'hoc luc', 'ren luyen cua']
STUDENT_INFO_HINTS = [
    'thong tin sinh vien', 'ho so sinh vien', 'mssv', 'ma sinh vien', 'sinh vien nao', 'sinh vien ma',
    'sinh vien co mssv', 'email sinh vien', 'so dien thoai sinh vien',
]
CLASS_INFO_HINTS = [
    'lop', 'si so', 'bao nhieu sinh vien', 'danh sach sinh vien', 'danh sach lop', 'bao nhieu lop',
    'gom nhung lop nao', 'lop nao', 'sinh vien lop',
]
POLICY_MESSAGES = {
    'student_private_info': (
        'Xin lỗi, đây là nhóm thông tin cá nhân của sinh viên và ngoài phạm vi chatbot hiện tại. '
        'Trong dữ liệu hiện có không cung cấp điểm, hồ sơ hay thông tin chi tiết theo mã sinh viên.'
    ),
    'class_private_info': (
        'Xin lỗi, truy vấn này liên quan tới mã lớp hoặc thông tin chi tiết của lớp và thuộc nhóm thông tin ngoài phạm vi chatbot hiện tại. '
        'Trong dữ liệu hiện có không có sĩ số, danh sách sinh viên, điểm hay thông tin chi tiết theo từng lớp.'
    ),
    'class_scope_not_supported': (
        'Xin lỗi, trong dữ liệu hiện có không có thông tin số lớp chi tiết theo từng khoa/ngành hoặc chi tiết lớp, '
        'và đây cũng là nhóm thông tin ngoài phạm vi chatbot hiện tại.'
    ),
    'self_profile_not_supported': (
        'Xin lỗi, chatbot hiện không truy cập dữ liệu cá nhân theo từng tài khoản sinh viên như lớp đang học, ngành đang học, hồ sơ hay mã sinh viên của riêng bạn.'
    ),
    'teaching_assignment_not_supported': (
        'Xin lỗi, trong dữ liệu hiện có không có phân công giảng dạy chi tiết theo từng môn, từng lớp hay từng giảng viên, nên chatbot chưa thể xác nhận ai đang dạy môn này.'
    ),
    'harmful_or_violent': (
        'Tôi không hỗ trợ các yêu cầu có nội dung bạo lực, phá hoại hoặc đe dọa. Nếu bạn cần, tôi có thể hỗ trợ các câu hỏi an toàn liên quan đến UNETI.'
    ),
    'academic_material_not_supported': (
        'Xin lỗi, tri thức hiện có chưa cung cấp giáo trình, đề cương hay học liệu chi tiết cho từng môn học, nên chatbot chưa thể trả lời chính xác nội dung này.'
    ),
    'academic_detail_not_supported': (
        'Xin lỗi, tri thức hiện có chưa có thông tin học vụ chi tiết như phân loại môn học, nhóm môn hay thuộc tính chi tiết của từng học phần.'
    ),
}


def is_control(text: str) -> bool:
    return norm_text_ascii(text) in {norm_text_ascii(x) for x in CONTROL_WORDS}


def is_greeting(text: str) -> bool:
    q = norm_text_ascii(text)
    if not q:
        return False
    if q in GREETING_PHRASES:
        return True
    tokens = q.split()
    if len(tokens) > 6:
        return False
    if 'chao' not in tokens and not any(tok in {'hello', 'hi', 'hey', 'alo'} for tok in tokens):
        return False
    return all(tok in GREETING_TOKENS for tok in tokens)


def greeting_reply() -> str:
    return 'Xin chào. Tôi hỗ trợ trả lời các câu hỏi về thông tin và tài liệu UNETI. Bạn cần hỏi nội dung gì?'


def _is_bot_complaint_query(q: str) -> bool:
    if not q:
        return False
    if q in BOT_COMPLAINT_STANDALONE_INSULTS:
        return True
    if any(hint in q for hint in BOT_COMPLAINT_DIRECT_HINTS):
        return True
    def has_phrase(phrase: str) -> bool:
        phrase_norm = re.escape(norm_text_ascii(phrase))
        return bool(re.search(rf'(?<![a-z0-9]){phrase_norm}(?![a-z0-9])', q))

    tokens = set(re.findall(r'[a-z0-9]+', q))
    if any(hint in q for hint in ['lien he', 'o dau', 'ho tro ky thuat', 'tai khoan', 'phong cntt', 'phong cong nghe thong tin']):
        return False
    has_target = any(has_phrase(target) for target in BOT_COMPLAINT_TARGETS) or 'm' in tokens
    if not has_target:
        return False
    return any(has_phrase(issue) for issue in BOT_COMPLAINT_ISSUES)


def detect_meta_query(text: str) -> str:
    q = norm_text_ascii(text)
    if not q:
        return ''
    if _is_bot_complaint_query(q):
        return 'bot_complaint'
    if is_greeting(q):
        return 'greeting'
    if q in {norm_text_ascii(x) for x in THANKS_HINTS}:
        return 'thanks'
    if q in {norm_text_ascii(x) for x in FAREWELL_HINTS}:
        return 'farewell'
    if q in {norm_text_ascii(x) for x in ACK_HINTS}:
        return 'ack'
    if any(hint in q for hint in BOT_IDENTITY_HINTS):
        return 'bot_identity'
    if any(hint in q for hint in BOT_CAPABILITY_HINTS):
        return 'bot_capability'
    if any(hint in q for hint in BOT_HELP_HINTS):
        return 'bot_help'
    return ''


def meta_reply(meta_intent: str) -> str:
    if meta_intent == 'greeting':
        return 'Xin chào. Tôi là chatbot hỗ trợ tra cứu thông tin và tài liệu UNETI. Bạn muốn hỏi nội dung gì?'
    if meta_intent == 'bot_identity':
        return 'Tôi là chatbot hỗ trợ tra cứu thông tin và tài liệu UNETI. Tôi tập trung trả lời các câu hỏi về nhà trường, phòng ban, khoa, cơ sở, cổng sinh viên và các hướng dẫn liên quan.'
    if meta_intent == 'bot_capability':
        return 'Tôi có thể hỗ trợ tra cứu thông tin UNETI như phòng ban, khoa, cơ sở đào tạo, ban giám hiệu, hướng dẫn trên cổng sinh viên và một số thông báo liên quan của trường.'
    if meta_intent == 'bot_help':
        return 'Bạn có thể hỏi ngắn gọn theo ý cần tìm, ví dụ: phòng đào tạo ở đâu, hiệu trưởng là ai, cách xem lịch học, cách tra cứu học phí, khoa công nghệ thông tin liên hệ thế nào.'
    if meta_intent == 'bot_complaint':
        return (
            'Xin l\u1ed7i v\u00ec \u0111\u00e3 kh\u00f4ng gi\u00fap \u0111\u01b0\u1ee3c b\u1ea1n. '
            'T\u00f4i c\u00f3 th\u1ec3 b\u1ecb thi\u1ebfu tri th\u1ee9c, hi\u1ec3u sai c\u00e2u h\u1ecfi ho\u1eb7c ch\u01b0a t\u00ecm \u0111\u01b0\u1ee3c ngu\u1ed3n ph\u00f9 h\u1ee3p. '
            'B\u1ea1n c\u00f3 th\u1ec3 h\u1ecfi l\u1ea1i c\u1ee5 th\u1ec3 h\u01a1n, g\u1eedi link/t\u00e0i li\u1ec7u li\u00ean quan, '
            'ho\u1eb7c b\u1ed5 sung tri th\u1ee9c tr\u1ef1c ti\u1ebfp trong \u00f4 chat \u0111\u1ec3 admin ki\u1ec3m tra tr\u01b0\u1edbc khi d\u00f9ng cho chatbot.'
        )
    if meta_intent == 'thanks':
        return 'Không có gì. Nếu cần, bạn cứ tiếp tục hỏi về thông tin hoặc tài liệu UNETI.'
    if meta_intent == 'farewell':
        return 'Chào bạn. Khi cần tra cứu thông tin UNETI, bạn cứ quay lại hỏi tiếp.'
    if meta_intent == 'ack':
        return 'Vâng. Nếu cần tra cứu thêm thông tin UNETI, bạn cứ hỏi tiếp.'
    return ''


def _tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r'[a-z0-9]+', norm_text_ascii(text)) if tok]


def _looks_like_person_reference(q: str) -> bool:
    tokens = _tokenize(q)
    if not 2 <= len(tokens) <= 6:
        return False
    if any(phrase in q for phrase in QUESTION_INTENT_PHRASES):
        return False
    if len(tokens) >= 2 and tokens[1] in {'so', 'khoa', 'phong', 'truong', 'ban', 'vien', 'nganh', 'lop'}:
        return False
    if tokens[0] in PERSON_TITLE_TOKENS and all(tok.isalpha() for tok in tokens[1:]):
        return True
    return False


def _looks_like_academic_term_fragment(q: str) -> bool:
    tokens = _tokenize(q)
    if not 2 <= len(tokens) <= 5:
        return False
    if any(phrase in q for phrase in QUESTION_INTENT_PHRASES):
        return False
    return all(tok in ACADEMIC_TERM_TOKENS for tok in tokens)


def _matches_harmful_query(q: str) -> bool:
    if any(hint in q for hint in HARMFUL_HINTS):
        return True
    tokens = set(_tokenize(q))
    destructive_verbs = {'dot', 'pha', 'danh', 'tan', 'chet', 'giet', 'bom'}
    destructive_targets = {'truong', 'phong', 'lop', 'giang', 'vien', 'sinh'}
    return bool(tokens & destructive_verbs) and bool(tokens & destructive_targets)


def is_low_signal_query(text: str) -> bool:
    q = norm_text_ascii(text)
    if not q:
        return False
    if any(hint == q for hint in CLARIFICATION_HINTS):
        return True
    if detect_meta_query(q) or is_greeting(q) or is_policy_query(q) or is_sensitive_query(q):
        return False
    if _looks_like_person_reference(q) or _looks_like_academic_term_fragment(q):
        return True
    tokens = q.split()
    if len(tokens) > 3:
        return False
    if any(tok in q for tok in ['khoa', 'phong', 'uneti', 'hoc phi', 'cong no', 'lich', 'diem', 'cong sinh vien', 'dang nhap']):
        return False
    return q in {'hoi gi', 'gi nua', 'sao nua', 'roi sao', 'xong sao', 'toi khong biet', 'tiep di'}


def clarification_reply() -> str:
    return 'Bạn nói rõ hơn một chút được không? Tôi đang hỗ trợ các câu hỏi về thông tin và tài liệu UNETI.'


def detect_question_type(text: str) -> str:
    q = norm_text_ascii(text)
    if any(x in q for x in ['liet ke', 'danh sach', 'gom', 'bao nhieu co so', 'bao nhieu khoa', 'cac co so', 'dia diem nao']):
        return 'list'
    if any(x in q for x in ['email', 'mail', 'so dien thoai', 'sdt', 'website', 'link', 'dia chi', 'o dau']):
        return 'contact'
    if any(x in q for x in ['so sanh', 'khac nhau']):
        return 'compare'
    if any(x in q for x in ['tom tat', 'gioi thieu', 'lich su', 'qua trinh', 'noi ve', 'thong tin ve']):
        return 'summary'
    if any(x in q for x in ['nhu the nao', 'lam the nao', 'cach', 'dang ky', 'xem ', 'tra cuu', 'mo o dau']):
        return 'howto'
    return 'factoid'


def detect_domain(text: str) -> Optional[str]:
    q = norm_text_ascii(text)
    if any(hint in q for hint in SCHOOL_OVERVIEW_HINTS):
        return 'lich_su_hinh_thanh'
    if any(x in q for x in ['truong phong', 'pho truong phong']):
        return 'phong_ban_va_chuc_nang'
    if any(x in q for x in ['truong khoa', 'pho truong khoa', 'tro ly khoa']):
        return 'khoa_chuyen_mon'
    is_contact_query = any(x in q for x in ['email', 'mail', 'so dien thoai', 'sdt', 'website', 'link', 'dia chi', 'o dau', 'van phong'])
    if is_contact_query:
        if 'khoa ' in q:
            return 'khoa_chuyen_mon'
        if re.search(r'\bphong\s+', q) and 'van phong khoa' not in q:
            return 'phong_ban_va_chuc_nang'
    if 'khoa' in q and any(x in q for x in ['bao nhieu', 'co bao nhieu', 'cac khoa', 'danh sach khoa']):
        return 'khoa_chuyen_mon'
    if any(x in q for x in ['hoc phi', 'cong no', 'phieu thu']) and not any(x in q for x in ['phong tai chinh', 'tai chinh ke toan', 'phong tckt']):
        return 'portal_howto'

    best = None
    best_len = -1
    for dom, hints in DOMAIN_HINTS.items():
        for h in hints:
            a = norm_text_ascii(h)
            if a in q and len(a) > best_len:
                best = dom
                best_len = len(a)
    return best


def detect_portal_topic(text: str) -> Optional[str]:
    q = norm_text_ascii(text)
    if any(tok in q for tok in ['dang nhap', 'login', 'username', 'mat khau']):
        return 'login'
    if 'doi mat khau' in q or 'doi password' in q:
        return 'change_password'
    best = None
    best_score = -1
    for topic, hints in PORTAL_TOPICS.items():
        score = 0
        for hint in hints:
            if norm_text_ascii(hint) in q:
                score += max(1, len(hint.split()))
        if topic.startswith('schedule') and ('lich hoc' in q or 'lich thi' in q):
            score += 3
        if score > best_score:
            best = topic
            best_score = score
    return best if best_score > 0 else None


def detect_campus_filter(text: str) -> Dict[str, str]:
    q = norm_text_ascii(text)
    out: Dict[str, str] = {}
    if 'ha noi' in q or 'minh khai' in q or 'linh nam' in q:
        out['city'] = 'ha noi'
    if 'nam dinh' in q or 'tran hung dao' in q or 'my xa' in q:
        out['city'] = 'nam dinh'
    if 'minh khai' in q:
        out['address'] = 'minh khai'
    if 'linh nam' in q:
        out['address'] = 'linh nam'
    if 'tran hung dao' in q:
        out['address'] = 'tran hung dao'
    if 'my xa' in q:
        out['address'] = 'my xa'
    return out


def classify_sensitive_query(text: str) -> Dict[str, str]:
    q = norm_text_ascii(text)
    if any(x in q for x in [
        'cong thong tin sinh vien', 'mo cong thong tin sinh vien', 'vao cong thong tin sinh vien',
        'xem thong tin sinh vien o dau', 'ho so sinh vien o dau',
    ]):
        return {}
    has_class_code = bool(CLASS_CODE_RE.search(q))
    has_student_code = bool(STUDENT_CODE_RE.search(q))
    asks_grade = any(x in q for x in GRADE_HINTS)
    asks_student_info = any(x in q for x in STUDENT_INFO_HINTS)
    asks_class_info = any(x in q for x in CLASS_INFO_HINTS)

    if any(hint in q for hint in SENSITIVE_HINTS):
        if 'lop' in q:
            if any(x in q for x in ['bao nhieu lop', 'so lop', 'danh sach lop']):
                return {'policy_code': 'class_scope_not_supported', 'message': POLICY_MESSAGES['class_scope_not_supported']}
            return {'policy_code': 'class_private_info', 'message': POLICY_MESSAGES['class_private_info']}
        if any(x in q for x in ['mssv', 'ma sinh vien', 'sinh vien']):
            return {'policy_code': 'student_private_info', 'message': POLICY_MESSAGES['student_private_info']}
        return {'policy_code': 'student_private_info', 'message': POLICY_MESSAGES['student_private_info']}
    if has_student_code and (asks_grade or asks_student_info):
        return {'policy_code': 'student_private_info', 'message': POLICY_MESSAGES['student_private_info']}
    if has_student_code:
        return {'policy_code': 'student_private_info', 'message': POLICY_MESSAGES['student_private_info']}
    if has_class_code and (asks_grade or asks_class_info):
        return {'policy_code': 'class_private_info', 'message': POLICY_MESSAGES['class_private_info']}
    if has_class_code:
        return {'policy_code': 'class_private_info', 'message': POLICY_MESSAGES['class_private_info']}

    has_student_scope = any(x in q for x in ['sinh vien', 'mssv', 'ma sinh vien', 'danh sach sinh vien'])
    has_class_scope = 'lop' in q
    asks_count = any(x in q for x in ['bao nhieu', 'si so', 'so luong', 'liet ke', 'danh sach', 'gom nhung lop nao'])
    has_department_scope = any(x in q for x in ['khoa', 'nganh', 'chuyen nganh', 'cntt', 'cong nghe thong tin', 'co khi', 'thuong mai'])

    if has_student_scope and (asks_grade or asks_student_info):
        return {'policy_code': 'student_private_info', 'message': POLICY_MESSAGES['student_private_info']}
    if has_class_scope and has_student_scope:
        return {'policy_code': 'class_private_info', 'message': POLICY_MESSAGES['class_private_info']}
    if has_class_scope and asks_count and has_department_scope:
        return {'policy_code': 'class_scope_not_supported', 'message': POLICY_MESSAGES['class_scope_not_supported']}
    if has_class_scope and asks_count:
        return {'policy_code': 'class_private_info', 'message': POLICY_MESSAGES['class_private_info']}
    return {}


def sensitive_query_reason(text: str) -> str:
    return str(classify_sensitive_query(text).get('message', '') or '')


def is_sensitive_query(text: str) -> bool:
    return bool(sensitive_query_reason(text))


def classify_policy_query(text: str) -> Dict[str, str]:
    q = norm_text_ascii(text)
    if _matches_harmful_query(q):
        return {'policy_code': 'harmful_or_violent', 'message': POLICY_MESSAGES['harmful_or_violent']}
    if any(hint in q for hint in SELF_PROFILE_HINTS):
        return {'policy_code': 'self_profile_not_supported', 'message': POLICY_MESSAGES['self_profile_not_supported']}
    if any(hint in q for hint in TEACHING_ASSIGNMENT_HINTS):
        return {'policy_code': 'teaching_assignment_not_supported', 'message': POLICY_MESSAGES['teaching_assignment_not_supported']}
    if any(hint in q for hint in ACADEMIC_MATERIAL_HINTS):
        return {'policy_code': 'academic_material_not_supported', 'message': POLICY_MESSAGES['academic_material_not_supported']}
    if any(hint in q for hint in ACADEMIC_DETAIL_HINTS):
        return {'policy_code': 'academic_detail_not_supported', 'message': POLICY_MESSAGES['academic_detail_not_supported']}
    return {}


def policy_query_reason(text: str) -> str:
    return str(classify_policy_query(text).get('message', '') or '')


def is_policy_query(text: str) -> bool:
    return bool(policy_query_reason(text))


def is_web_notice_query(text: str) -> bool:
    q = norm_text_ascii(text)
    if any(hint in q for hint in WEB_NOTICE_HINTS):
        return True
    return any(hint in q for hint in WEB_NOTICE_TIME_HINTS) and any(anchor in q for anchor in ['uneti', 'truong', 'nha truong'])


def is_out_of_scope_query(text: str, memory: Optional[Dict[str, str]] = None) -> bool:
    q = norm_text_ascii(text)
    if not q:
        return False
    if is_control(q) or detect_meta_query(q) or is_sensitive_query(q) or is_policy_query(q) or is_web_notice_query(q):
        return False
    if detect_domain(q):
        return False
    if _should_use_history(q, memory):
        return False
    if any(hint in q for hint in UNETI_SCOPE_HINTS):
        return False

    content_tokens = [tok for tok in q.split() if len(tok) >= 3]
    if len(content_tokens) < 2:
        return False
    if any(hint in q for hint in OUT_OF_SCOPE_HINTS):
        return True
    return True


def out_of_scope_reply() -> str:
    return 'Tôi chỉ hỗ trợ các câu hỏi liên quan đến thông tin, tài liệu và nghiệp vụ của UNETI. Câu hỏi này hiện nằm ngoài phạm vi hỗ trợ của chatbot.'


def detect_query_family(text: str, memory: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    q = norm_text_ascii(text)
    meta_intent = detect_meta_query(q)
    if meta_intent:
        return {'family': 'meta', 'meta_intent': meta_intent}
    if is_low_signal_query(q):
        detail = 'clarification'
        if _looks_like_person_reference(q):
            detail = 'person_reference_needs_clarification'
        elif _looks_like_academic_term_fragment(q):
            detail = 'academic_reference_needs_clarification'
        return {'family': 'clarification', 'detail': detail}
    policy = classify_policy_query(q)
    if policy:
        return {'family': 'policy', 'policy_code': str(policy.get('policy_code', '') or '')}
    sensitive = classify_sensitive_query(q)
    if sensitive:
        return {'family': 'sensitive', 'policy_code': str(sensitive.get('policy_code', '') or '')}
    domain = detect_domain(q)
    if domain:
        return {'family': 'domain', 'domain': domain}
    if is_web_notice_query(q):
        return {'family': 'web_notice'}
    if is_out_of_scope_query(q, memory):
        return {'family': 'out_of_scope'}
    return {'family': 'general'}


def _should_use_history(q: str, memory: Optional[Dict[str, str]]) -> bool:
    if not memory:
        return False
    if int(memory.get('context_turns', 0) or 0) <= 0:
        return False
    explicit_unit_match = re.search(r'\b(?:khoa|phong)\s+([a-z0-9][a-z0-9\s&.-]*)', q)
    if explicit_unit_match:
        explicit_unit = explicit_unit_match.group(1)
        explicit_unit = re.split(
            r'\b(?:o dau|dia chi|email|website|web|link|so dien thoai|sdt|chuc nang|nhiem vu|truong khoa|pho truong khoa|truong phong|pho truong phong|la gi|nhu the nao)\b',
            explicit_unit,
            maxsplit=1,
        )[0].strip()
        explicit_tokens = [
            tok for tok in explicit_unit.split()
            if tok not in {
                'do', 'nay', 'ay', 'kia', 'la', 'gi', 'ai', 'o', 'dau', 'email', 'website', 'web',
                'link', 'dien', 'thoai', 'sdt', 'chuc', 'nang', 'nhiem', 'vu',
            }
        ]
        if explicit_tokens:
            return False
    if any(hint in q for hint in FOLLOW_UP_HINTS):
        return True
    if any(anchor in q for anchor in EXPLICIT_QUERY_ANCHORS):
        return False
    has_explicit_anchor = any(anchor in q for anchor in EXPLICIT_QUERY_ANCHORS)
    if any(hint in q for hint in ['email', 'website', 'web', 'link', 'dien thoai', 'sdt', 'so dien thoai', 'dia chi', 'o dau',
                                  'chuc nang', 'nhiem vu', 'truong khoa', 'pho truong khoa', 'truong phong', 'pho truong phong',
                                  'lich su', 'thanh tich', 'dinh huong']):
        if has_explicit_anchor:
            return False
        return True
    if q in GENERIC_FOLLOW_UPS:
        return True
    if len(q.split()) <= 3 and q in {'email', 'dia chi', 'o dau', 'con lai', 'tiep di'}:
        return True
    return False


def should_use_history(text: str, memory: Optional[Dict[str, str]]) -> bool:
    return _should_use_history(norm_text_ascii(text), memory)


def _compact_history_hint(text: str, max_tokens: int = 18) -> str:
    q = norm_text_ascii(text)
    if not q:
        return ''
    q = re.split(
        r'\b(?:email|website|web|link|dia chi|o dau|chuc nang|nhiem vu|la ai|la gi|nhu the nao)\b',
        q,
        maxsplit=1,
    )[0].strip() or q
    tokens = q.split()
    return ' '.join(tokens[:max_tokens])


def expand_query(text: str, domain: Optional[str] = None, memory: Optional[Dict[str, str]] = None) -> str:
    q = norm_text_ascii(text)
    terms: List[str] = [q]

    for alias, expansions in QUERY_ALIASES.items():
        if alias in q:
            terms.extend(expansions)

    if domain == 'ban_giam_hieu':
        if 'pho hieu truong' in q or 'hieu pho' in q:
            terms.extend(['pho hieu truong'])
        elif 'hieu truong' in q or 'nguoi dung dau' in q or 'lanh dao cao nhat' in q:
            terms.extend(['hieu truong'])
        else:
            terms.extend(['ban giam hieu'])

    elif domain == 'co_so_vat_chat':
        campus_filter = detect_campus_filter(q)
        if campus_filter.get('address') == 'minh khai':
            terms.extend(['ha noi', 'minh khai'])
        elif campus_filter.get('address') == 'linh nam':
            terms.extend(['ha noi', 'linh nam'])
        elif campus_filter.get('address') == 'tran hung dao':
            terms.extend(['nam dinh', 'tran hung dao'])
        elif campus_filter.get('address') == 'my xa':
            terms.extend(['nam dinh', 'my xa'])
        elif campus_filter.get('city') == 'ha noi':
            terms.extend(['ha noi', 'minh khai', 'linh nam'])
        elif campus_filter.get('city') == 'nam dinh':
            terms.extend(['nam dinh', 'tran hung dao', 'my xa'])
        else:
            terms.extend(['co so', 'dia diem dao tao', 'ha noi', 'nam dinh'])

    elif domain == 'portal_howto':
        topic = detect_portal_topic(q)
        if topic and topic in PORTAL_TOPIC_EXPANSIONS:
            terms.extend(PORTAL_TOPIC_EXPANSIONS[topic])
        else:
            terms.extend(['cong sinh vien'])

    memory = memory or {}
    if _should_use_history(q, memory):
        last_entity = str(memory.get('last_entity', '') or '').strip()
        last_named_unit = str(memory.get('last_named_unit', '') or '').strip()
        last_user_query = str(memory.get('last_user_query', '') or '').strip()
        last_assistant_answer = str(memory.get('last_assistant_answer', '') or '').strip()
        last_topic = str(memory.get('last_topic', '') or '').strip()
        if last_topic:
            terms.append(last_topic)
        if last_named_unit:
            terms.append(last_named_unit)
        if last_entity:
            terms.append(last_entity)
        if last_user_query:
            terms.append(_compact_history_hint(last_user_query))
        if last_assistant_answer and not (last_named_unit or last_entity):
            terms.append(_compact_history_hint(last_assistant_answer, max_tokens=12))

    unique_terms: List[str] = []
    seen = set()
    for term in terms:
        norm = norm_text_ascii(term)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        unique_terms.append(term.strip())
    return ' '.join(unique_terms)


def domain_titles() -> Dict[str, str]:
    seed = load_seed_knowledge()['domains']
    return {k: v['title'] for k, v in seed.items()}
