from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import json
import difflib
import os
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
CORS(app)

# 🔑 OpenAI API Key
openai.api_key = "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"

# Load FAQ data
with open("faqs.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

def find_faq_answer(user_msg):
    questions = [faq["q"] for faq in faq_data]
    matches = difflib.get_close_matches(user_msg, questions, n=1, cutoff=0.6)
    if matches:
        for faq in faq_data:
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
        return "सेवा उपलब्ध नहीं है। कृपया बाद में प्रयास करें।"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "")
    lang = data.get("lang", "en")
    faq_answer = find_faq_answer(msg)
    if faq_answer:
        return jsonify({"reply": faq_answer})
    gpt_reply = ask_gpt(msg, lang)
    return jsonify({"reply": gpt_reply})

@app.route("/chart", methods=["POST"])
def generate_chart():
    data = request.get_json()
    try:
        categories = data['categories']
        weights = data['weights']
        boxes = data['boxes']
    except (KeyError, TypeError):
        return jsonify({"error": "Missing or invalid 'categories', 'weights', or 'boxes'."}), 400

    if not (len(categories) == len(weights) == len(boxes)):
        return jsonify({"error": "All lists must be of the same length."}), 400

    # Create figure and plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvas(fig)

    ax1.set_xlabel("Categories")
    ax1.set_ylabel("Weight (Kg)", color="green")
    ax1.plot(categories, weights, color="green", marker='o', label="Weight (Kg)")
    ax1.tick_params(axis='y', labelcolor="green")
    ax1.set_xticklabels(categories, rotation=45)

    ax2 = ax1.twinx()
    ax2.set_ylabel("No. of Boxes", color="blue")
    ax2.plot(categories, boxes, color="blue", marker='s', label="No. of Boxes")
    ax2.tick_params(axis='y', labelcolor="blue")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Weight and Box Count per Category")
    plt.tight_layout()
    plt.grid(True)

    # Save to in-memory buffer
    buf = io.BytesIO()
    canvas.print_png(buf)
    buf.seek(0)

    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
