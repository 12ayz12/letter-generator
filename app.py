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

# 공문 프롬프트 템플릿
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
"""
# 아래 참고 예시 내용을 반드시 참고하여, 위와 동일한 서식(가. 나. 다.)으로 공문을 작성하세요.
# [참고 예시]
# {example}
# """

def build_prompt(user_input, examples):
    # 예시 중 첫 1개만 아주 짧게 (길면 자르기)
    EXAMPLE_DOC = """
────────────────────────────
제목: 2025학년도 교원 연수 안내

1. 관련: ○○교육청-2025(2025.6.15.)
2. 2025학년도 하계 교원 연수 운영과 관련하여 아래와 같이 연수를 실시하오니, 협조 부탁드립니다.
   가. 연수명: 2025학년도 하계 연수
   나. 대상: 관내 교원
   다. 기간: 2025.7.20.~2025.7.22.
   라. 장소: ○○연수원
   마. 신청 방법: 붙임 참조

붙임: 교원 연수 안내문 1부.  끝.
────────────────────────────
"""
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
        keyword = data.get("keyword", "")
        event = data.get("event", "")
        summary = data.get("summary", "")
        date_ = data.get("date", "")
        time_ = data.get("time", "")
        location = data.get("location", "")
        target = data.get("target", "")
        application = data.get("application", "")
        contact = data.get("contact", "")
        attachments = data.get("attachments", "")
        print(f"[DEBUG] 입력값: keyword={keyword}, event={event}, attachments={attachments}")

        # 필수값 체크
        if not all([event, summary, date_, time_, location, target, application, attachments]):
            print("[ERROR] 입력값 누락")
            return jsonify({"error": "필수 입력값이 누락되었습니다."}), 400

        user_input = {
            "keyword": keyword,
            "event": event,
            "summary": summary,
            "date": date_,
            "time": time_,
            "location": location,
            "target": target,
            "application": application,
            "contact": contact,
            "attachments": attachments,
        }
        query = f"{event} {summary} {attachments}"

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

# 생활기록부 생성 API
@app.route("/generate_saenggi", methods=["POST"])
def generate_saenggi():
    try:
        data = request.get_json()
        subject = data.get("subject")
        area = data.get("area")
        element = data.get("element")
        criterion = data.get("criterion")
        level = data.get("level")
        character = data.get("character")

        if not all([subject, area, element, criterion, level, character]):
            return jsonify({"error": "모든 필드를 입력해주세요."}), 400

        # 프롬프트 구성
        prompt = f"""
다음 정보를 바탕으로 학생 생활기록부용 한두 문장 평가를 작성해줘.
- 과목명: {subject}
- 평가영역: {area}
- 평가요소: {element}
- 성취기준: {criterion}
- 평가등급: {level}
- 학생 특성: {character}

예시는 불필요하고, 실제 평가처럼 자연스럽게, 단 한두 문장만 생성하고 '-함', '-할 수 있음.'과 같은 말투로 끝맺음해줘.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        result = response["choices"][0]["message"]["content"].strip()
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
