from flask import Flask, request, jsonify, send_from_directory
import openai
import os
from pathlib import Path
import faiss
import pickle
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DOC_EMBED_FILE = EMBEDDINGS_DIR / "doc_embeddings.pkl"
INDEX_FILE = EMBEDDINGS_DIR / "faiss.index"

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# 문서 벡터화 모델(서버가 뜰 때 1회만 로드)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents():
    documents = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        try:
            text = extract_text(pdf_path)
            if text.strip():
                documents.append((str(pdf_path), text.strip()))
        except Exception:
            continue
    return documents

def build_vector_db(model, documents):
    texts = [text for _, text in documents]
    embeddings = model.encode(texts)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOC_EMBED_FILE, "wb") as f:
        pickle.dump(documents, f)
    faiss.write_index(index, str(INDEX_FILE))

def retrieve_similar_docs(model, query, top_k=3):
    if not INDEX_FILE.exists() or not DOC_EMBED_FILE.exists():
        # 인덱스가 없으면 생성
        documents = load_documents()
        if documents:
            build_vector_db(model, documents)
        else:
            return []
    index = faiss.read_index(str(INDEX_FILE))
    with open(DOC_EMBED_FILE, "rb") as f:
        documents = pickle.load(f)
    query_vec = model.encode([query])
    D, I = index.search(query_vec, top_k)
    return [documents[i][1] for i in I[0] if i < len(documents)]

EXAMPLE_DOC = """
────────────────────────────
제목: 2025학년도 학교교육력 제고 연구팀 중간 발표회 및 전시회 참관 안내

1. 관련: 서울특별시교육청학생체육관 교육지원과-73(2025. 2. 5.)
2. 2025학년도 학교교육력 제고 <청소년(수련) 활동> 연구팀의 연구 결과 중간 발표회 및 전시회를 다음과 같이 실시하고자 하오니, 귀교의 희망 교원이 참관할 수 있도록 안내 부탁드립니다.
   가. 연구 주제: 스카우팅을 통한 사회정서역량 키우기
   나. 중간발표회 및 전시회 개요
     1) 일시: 2025. 6. 18.(수) 14:30~16:00
     2) 장소: 본교 2층 과학실
     3) 일정
   다. 참관 신청 및 안내 사항
     1) 2025. 6. 16.(월) 12:00까지 아래 링크를 통해 참관 신청
        - 신청링크: https://forms.gle/
     2) 신청교사는 참관 후 참관록 작성 협조
   라. 비고: 본 발표회 및 전시회는 수업 나눔 문화 확산을 위한 [2025 상반기 강서양천 내게 다가온 수업 한마당]과 연계하여 진행함. 

붙임: 2025학년도 학교교육력 제고 연구팀 중간 발표회 및 전시회 포스터 1부. 끝.

────────────────────────────
"""
def build_prompt(user_input, examples):
    example_text = "\n\n---\n\n".join(examples[:1])
    prompt = f"""
아래 예시의 아래 예시의 '목차(1., 2., 3., 가., 나., 다.), 구분선, 결재선, 붙임' 등 서식을 반드시 그대로 따라 새 공문을 작성하세요.
※ 회사명, 사장, 부사장, 업체명, 담당자 등 **비(非)교육청/학교/공문 스타일**은 절대 사용하지 마세요.
※ 오로지 학교 또는 교육청 공문 서식만 허용합니다.
[대표 예시]
{EXAMPLE_DOC}

(참고용 예시)
{example_text}

공문 스타일에 맞게 제목, 본문, 붙임, 결재선 등을 포함한 문서를 작성하세요.
위 예시와 같은 형식으로 작성하세요:

- 키워드: {user_input["keyword"]}
- 행사명: {user_input["event"]}
- 첨부파일: {user_input["attachments"]}

공문 스타일에 맞게 제목, 본문, 붙임, 결재선 등을 포함한 문서를 작성하세요.
"""
    return prompt
    
# 2. 메인 페이지
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
    
#3. 공문 생성 API
@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("[DEBUG] /generate called")
        
    data = request.get_json()
    keyword = data.get("keyword")
    event = data.get("event")
    attachments = data.get("attachments")
    print(f"[DEBUG] 입력값: keyword={keyword}, event={event}, attachments={attachments}")

    if not all([keyword, event, attachments]):
        print("[ERROR] 입력값 누락")
        return jsonify({"error": "모든 입력 필드(keyword, event, attachments)가 필요합니다."}), 400

    user_input = {"keyword": keyword, "event": event, "attachments": attachments}
    query = f"{keyword} {event} {attachments}"
  
    # PDF 기반 유사 예시 추출
    examples = retrieve_similar_docs(sentence_model, query)
    print(f"[DEBUG] 예시 개수: {len(examples)}")
    
    prompt = build_prompt(user_input, examples)
    print("[DEBUG] prompt 길이:", len(prompt))
    print("[DEBUG] prompt 미리보기:\n", prompt[:300])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        result = response["choices"][0]["message"]["content"]
        print("[DEBUG] 생성 결과 미리보기:", result[:200])
        return jsonify({"result": result})

    except Exception as e:
        print("[EXCEPTION]", e)
        return jsonify({"error": str(e)}), 500

