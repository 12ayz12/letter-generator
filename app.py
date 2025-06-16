from flask import Flask, request, jsonify, send_from_directory
import openai
import os

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다.")

# 공문 생성 프롬프트 함수
def build_prompt(user_input):
    EXAMPLE_PROMPT = f"""
제목: {user_input.get('event','')}

수신: 각급 학교장
발신: OO교육지원청 교육장

1. {user_input.get('summary','')}

주요 안내사항
  가. 일시: {user_input.get('date','')} {user_input.get('time','')}
  나. 장소: {user_input.get('location','')}
  다. 대상: {user_input.get('target','')}
  라. 신청 방법: {user_input.get('application','')}
  마. 문의처: {user_input.get('contact','')}

붙임: {user_input.get('attachments','')} 1부. 끝.
"""
    return EXAMPLE_PROMPT

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        user_input = {
            "event": data.get("event", ""),
            "summary": data.get("summary", ""),
            "date": data.get("date", ""),
            "time": data.get("time", ""),
            "location": data.get("location", ""),
            "target": data.get("target", ""),
            "application": data.get("application", ""),
            "contact": data.get("contact", ""),
            "attachments": data.get("attachments", ""),
        }
        prompt = build_prompt(user_input)
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

        prompt = f"""
다음 정보를 바탕으로 학생 생활기록부용 한두 문장 평가를 작성해줘.
- 과목명: {subject}
- 평가영역: {area}
- 평가요소: {element}
- 성취기준: {criterion}
- 평가등급: {level}
- 학생 특성: {character}

예시는 불필요하고, 실제 평가처럼 자연스럽게, 단 한두 문장만 생성해줘.
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

