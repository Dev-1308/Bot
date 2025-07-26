from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import json
import difflib

app = Flask(__name__)
CORS(app)

# 🔑 Set your OpenAI API Key here
openai.api_key = "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"

# Load FAQs
with open("faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

def find_faq_answer(user_msg, role):
    questions = [faq["q"] for faq in faq_data.get(role, [])]
    matches = difflib.get_close_matches(user_msg, questions, n=1, cutoff=0.6)
    if matches:
        for faq in faq_data[role]:
            if faq["q"] == matches[0]:
                return faq["a"]
    return None

def ask_gpt(message, lang):
    prompt = f"उत्तर हिंदी में दें:\n{message}" if lang == "hi" else message
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant helping users with agriculture logistics queries."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "GPT सेवा उपलब्ध नहीं है। कृपया बाद में प्रयास करें।"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "")
    role = data.get("role", "Grower")
    lang = data.get("lang", "hi")

    # First, try FAQ match
    faq_answer = find_faq_answer(msg, role)
    if faq_answer:
        return jsonify({"reply": faq_answer})

    # Else, fallback to GPT
    gpt_reply = ask_gpt(msg, lang)
    return jsonify({"reply": gpt_reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render dynamically sets PORT
    print(f"🚀 Starting Flask on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)

    app.run(debug=True)
# If you want to run this server, make sure to set your OpenAI API key in the code above.   