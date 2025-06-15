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

def retrieve_similar_docs(model, query, top_k=1):
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

# 대표 공문 포맷 + RAG 예시(최대 1개, 400자 이내)
EXAMPLE_PROMPT = """
제목: {event}

수신: 각급 학교장
발신: OO교육지원청 교육장

1. {summary}

주요 안내사항
  가. 일시: {date} {time}
  나. 장소: {location}
  다. 대상: {target}
  라. 신청 방법: {application}
  마. 문의처: {contact}

붙임: {attachments} 1부. 끝.

[참고 예시]
{example}

※ 참고 예시 내용을 반드시 반영하여, 위와 동일한 서식(가. 나. 다.)으로 공문을 작성하세요.
"""

def build_prompt(user_input, examples):
    # 예시 중 첫 1개만, 400자 이내로 잘라서 사용
    example_text = (examples[0][:400]) if examples else ""
    return EXAMPLE_PROMPT.format(
        event=user_input.get("event", ""),
        summary=user_input.get("summary", ""),
        date=user_input.get("date", ""),
        time=user_input.get("time", ""),
        location=user_input.get("location", ""),
        target=user_input.get("target", ""),
        application=user_input.get("application", ""),
        contact=user_input.get("contact", ""),
        attachments=user_input.get("attachments", ""),
        example=example_text
    )

# 메인 페이지
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# 공문 생성 API
@app.route("/generate", methods=["POST"])
def generate():
    try:
        print("[DEBUG] /generate called")
        data = request.get_json()
        # 입력값 세부 항목으로 분리(폼에서 받아야 함)
        event = data.get("event", "")
        summary = data.get("summary", "")
        date = data.get("date", "")
        time = data.get("time", "")
        location = data.get("location", "")
        target = data.get("target", "")
        application = data.get("application", "")
        contact = data.get("contact", "")
        attachments = data.get("attachments", "")

        # 모든 항목 필수면 체크 (옵션화 가능)
        if not all([event, summary, date, time, location, target, application, attachments]):
            print("[ERROR] 입력값 누락")
            return jsonify({"error": "모든 입력 필드를 입력해 주세요."}), 400

        user_input = {
            "event": event,
            "summary": summary,
            "date": date,
            "time": time,
            "location": location,
            "target": target,
            "application": application,
            "contact": contact,
            "attachments": attachments
        }
        # 검색 쿼리에는 대표 정보만 모아서
        query = f"{event} {summary} {date} {location} {target} {application}"

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


