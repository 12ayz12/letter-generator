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

# 프롬프트 입력(실제 예시만 사용하도록)
def build_prompt(user_input, examples):
    # 실제 예시(교육청 공문)만 사용!
    EXAMPLE_DOC = """────────────────────────────
제목: 2025학년도 교원 연수 안내

1. 관련: 서울특별시교육청-1234(2025.3.1.)
2. 2025학년도 하계 교원 연수 운영과 관련하여 아래와 같이 연수를 실시하오니, 협조 부탁드립니다.
  가. 연수명: 2025학년도 하계 교원 연수
  나. 대상: 관내 초·중등 교원
  다. 기간: 2025.7.20.~2025.7.22.
  라. 장소: ○○연수원
  마. 기타: 붙임 참조

붙임: 교원 연수 안내문 1부.  끝.
────────────────────────────
"""

    prompt = f"""
아래 공문 예시의 '번호(1., 2., 3.), 소항목(가., 나., 다.), 붙임, 끝' 형식과 구분선을 반드시 그대로 따라 새 공문을 작성하세요.  
아래 예시와 같은 형식 외에는 절대 다른 형식(회사, 업체, 담당자, 자유 인사말 등)이나 표현을 사용하지 마세요.

[공문 예시]
{EXAMPLE_DOC}

[입력 정보]
- 행사명: {user_input["event"]}
- 키워드: {user_input["keyword"]}
- 첨부파일: {user_input["attachments"]}

위 예시와 입력 정보를 바탕으로, 반드시 예시 형식의 공문을 작성하세요.
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

