from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'seed'
STORAGE_DIR = ROOT / 'storage'
UPLOAD_DIR = STORAGE_DIR / 'uploads'
PARSED_DIR = STORAGE_DIR / 'parsed'
WEB_PARSED_DIR = PARSED_DIR / 'uneti_site'
INDEX_DIR = STORAGE_DIR / 'indexes'
HISTORY_DIR = STORAGE_DIR / 'chat_histories'
REVIEW_DIR = STORAGE_DIR / 'reviews'
REVIEW_DB_PATH = REVIEW_DIR / 'admin_review_db.json'
CONTRIBUTION_DB_PATH = REVIEW_DIR / 'pending_contributions.json'
FEEDBACK_DB_PATH = REVIEW_DIR / 'user_feedback_db.json'
RLHF_DIR = STORAGE_DIR / 'rlhf'
for p in [UPLOAD_DIR, PARSED_DIR, WEB_PARSED_DIR, INDEX_DIR, HISTORY_DIR, REVIEW_DIR, RLHF_DIR]:
    p.mkdir(parents=True, exist_ok=True)
