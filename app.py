from flask import Flask, request, jsonify
import openai
import os

app = Flask(__name__)

# 1. 환경변수에서 API 키 가져오기
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# 2. 공문 생성 API 엔드포인트
@app.route("/generate", methods=["POST"])
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

def generate_document():
    data = request.get_json()

    keyword = data.get("keyword")
    event = data.get("event")
    attachments = data.get("attachments")

    if not all([keyword, event, attachments]):
        return jsonify({"error": "모든 입력 필드(keyword, event, attachments)가 필요합니다."}), 400

    prompt = f"""
당신은 공문 작성 전문가입니다. 다음 조건에 맞는 새로운 공문을 작성하세요:

- 키워드: {keyword}
- 행사명: {event}
- 첨부파일: {attachments}

공문 스타일에 맞게 제목, 본문, 붙임, 결재선 등을 포함한 문서를 작성하세요.
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        result = response["choices"][0]["message"]["content"]
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. 로컬 실행용 (배포용에선 gunicorn 사용)
if __name__ == "__main__":
    app.run(debug=True)
