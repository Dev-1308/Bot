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
    """
    Generates a styled chart from POST request data and returns it as a PNG image.
    """
    # 1. Get data from the POST request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400
            
        categories = data['categories']
        quality = data['quality']
        weights = data['weights']
        boxes = data['boxes']
    except (KeyError, TypeError):
        return jsonify({"error": "Missing or invalid 'categories', 'weights', or 'boxes' keys."}), 400

    # Validate that all lists have the same length
    if not (len(categories) == len(weights) == len(boxes)):
        return jsonify({"error": "All lists ('categories', 'weights', 'boxes') must be of the same length."}), 400

    # 2. Create the chart
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('gold') # Set outer background

    # Set plot area background and border
    ax1.set_facecolor('#F0F8FF') 
    for spine in ax1.spines.values():
        spine.set_edgecolor('c')
        spine.set_linewidth(1.5)

    # Plot the data lines
    ax1.plot(categories, weights, color='blue', marker='o', linewidth=2.5, label="Weight (Kg)")
    ax1.set_ylabel("Weight (Kg)", color="blue", fontsize=12, weight='bold')
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(categories, boxes, color='red', marker='s', linewidth=2.5, label="No. of Boxes")
    ax2.set_ylabel("No. of Boxes", color="red", fontsize=12, weight='bold')
    ax2.tick_params(axis='y', labelcolor="red")

    # 3. Add the data table at the bottom
    ax1.set_xticks([]) # Hide original x-axis ticks
    
    the_table = plt.table(cellText=[weights, boxes],
                          rowLabels=['Weight (Kg)', 'No. of Boxes'],
                          colLabels=categories,
                          loc='bottom',
                          cellLoc='center',
                          rowLoc='center')
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.8)

    # Adjust layout to make room for the table
    plt.subplots_adjust(left=0.1, bottom=0.25)

    # 4. Final touches
    plt.title(f"Consignment Wise Procurement Analytics {quality}", fontsize=16, weight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 5. Save chart to an in-memory buffer
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    plt.close(fig) # Close the figure to free up memory
    buf.seek(0)

    # 6. Return the image as a response
    return send_file(buf, mimetype='image/png')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
