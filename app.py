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

EXAMPLE_PROMPT = """
제목: {event}

수신: 각급 학교장
발신: OO교육지원청 교육장

1. {event}와 관련하여 다음과 같이 안내합니다.

주요 안내사항
  가. 주요 키워드: {keyword}
  나. 일정 및 장소: ○○○
  다. 유의사항: 붙임 참조

붙임: {attachments} 1부. 끝.
"""

def build_prompt(user_input):
    return EXAMPLE_PROMPT.format(
        event=user_input["event"],
        keyword=user_input["keyword"],
        attachments=user_input["attachments"]
    )

    
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

