# UNETI Document Agent V4 Max

Bản tối ưu hóa theo hướng local-only, giữ UI Streamlit, đăng nhập bằng mã sinh viên, lưu lịch sử chat, và trả lời dựa trên tài liệu PDF/DOCX.

## Điểm chính
- 2 LLM local qua Ollama:
  - LLM 1: planner/analyzer
  - LLM 2: grounded answerer
- 3 lớp tri thức:
  - fact records
  - portal QA
  - adaptive chunks
- Rule routing ưu tiên:
  - hiệu trưởng / hiệu phó -> Ban giám hiệu
  - cơ sở / địa chỉ / Minh Khai / Lĩnh Nam / Trần Hưng Đạo / Mỹ Xá -> Cơ sở vật chất
- Chỉ trả lời text
- Có lịch sử chat theo mã sinh viên

## Chạy
```bash
pip install -r requirements.txt
bash run_v4.sh
```

## Kiến trúc
- `app/streamlit_app_v4.py`: giao diện
- `core/pipeline_v4.py`: luồng chính
- `core/ingest.py`: đọc PDF/DOCX
- `core/chunking.py`: adaptive chunking
- `core/retrieval.py`: hybrid retrieval lexical + BM25 + rule boosts
- `core/routing.py`: domain routing
- `core/llm.py`: Ollama client, planner, answerer
- `core/guards.py`: chống hallucination cơ bản
- `core/history.py`: lưu chat theo mã sinh viên

## Ghi chú
- Có sẵn `data/seed/knowledge_seed.json` cho các phần Ban giám hiệu, Cơ sở vật chất, Lịch sử, Hội đồng trường, Portal.
- Có thể bấm nạp tài liệu seed trong UI để parse thêm các file DOCX.
