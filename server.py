from flask import Flask, request, jsonify, send_file, current_app
from flask_cors import CORS
import openai
import json
import difflib
import os
import logging
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Patch
import matplotlib.patheffects as path_effects
import seaborn as sns

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


openai.api_key = "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"


# Load FAQ data
def load_faq_data():
    try:
        with open("faqs.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        app.logger.error(f"Error loading FAQ data: {str(e)}")
        return []

faq_data = load_faq_data()

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
        app.logger.error(f"OpenAI API error: {str(e)}")
        return "सेवा उपलब्ध नहीं है। कृपया बाद में प्रयास करें।" if lang == "hi" else "Service unavailable. Please try again later."

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400
            
        msg = data.get("message", "")
        lang = data.get("lang", "en")
        
        faq_answer = find_faq_answer(msg)
        if faq_answer:
            return jsonify({"reply": faq_answer})
            
        gpt_reply = ask_gpt(msg, lang)
        return jsonify({"reply": gpt_reply})
        
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/chart", methods=["POST"])
def generate_chart():
    """Generate a modern styled bar-line chart and return as raw image bytes (UInt8List)"""

        # 1. Get and validate data
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        categories = data.get('categories', ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
        quality = data.get('quality', 'Standard')
        weights = data.get('weights', [1200, 1550, 950, 1800, 1300, 1650])
        boxes = data.get('boxes', [80, 100, 65, 120, 90, 110])
    except Exception as e:
        # Catch potential issues if data is not a dict or keys are missing
        return jsonify({"error": f"Invalid JSON data: {e}"}), 400

    # Validate that all lists have the same length
    if not (len(categories) == len(weights) == len(boxes)):
        return jsonify({"error": "All lists ('categories', 'weights', 'boxes') must be of the same length."}), 400


        # 2. Setup modern style
        sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.color': '0.8'})
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        
        # Create figure with golden background
        fig, ax1 = plt.subplots(figsize=(14, 8.5), facecolor='gold')
        fig.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.25)
        
        # 3. Line Plot (Weights) - Dark Blue
        line_color = '#22258d'
        ax1.plot(categories, weights, 
                 color=line_color,
                 marker='D',
                 markersize=8,
                 markeredgecolor='white',
                 markeredgewidth=1.5,
                 linewidth=3, 
                 label="Weight (Kg)",
                 path_effects=[path_effects.withStroke(linewidth=5, foreground='white')])

        # Add data labels
        for x, y in zip(categories, weights):
            ax1.annotate(f"{y:,} kg", (x, y),
                         textcoords="offset points", xytext=(0,15), ha='center',
                         fontsize=10, fontweight='bold', color='white',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor=line_color, edgecolor='white', alpha=0.9))

        ax1.set_ylabel("Weight (Kg)", color=line_color, fontsize=13, labelpad=15)
        ax1.tick_params(axis='y', colors=line_color, labelsize=11)
        ax1.set_ylim(0, max(weights) * 1.25)
        
        # 4. Bar Plot (Boxes)
        ax2 = ax1.twinx()
       

        bar_colors = ['#d90429', '#d90429', '#f97316', '#f97316', '#22c55e', '#f472b6']
        if len(categories) != len(bar_colors):
            bar_colors = ['#3a7bd5'] * len(categories)

        bars = ax2.bar(categories, boxes, 
                       color=bar_colors,
                       alpha=0.9, 
                       width=0.5,
                       edgecolor='white',
                       linewidth=1.5,
                       label="No. of Boxes")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height}",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom',
                         fontsize=10,
                         fontweight='bold',
                         color=bar.get_facecolor(),
                         path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

        ax2.set_ylabel("No. of Boxes", color='#22258d', fontsize=13, labelpad=15)
        ax2.tick_params(axis='y', colors='#22258d', labelsize=11)
        ax2.set_ylim(0, max(boxes) * 1.4)
        
        # 5. Title and labels
        fig.suptitle(f"PROCUREMENT ANALYTICS - {quality} Quality", 
                    fontsize=18, y=0.98, color='#000000', fontweight='bold')
        plt.title("Weight vs Box Count by Consignment", 
                 fontsize=12, pad=20, color='#000000', fontweight='bold')
        ax1.tick_params(axis='x', rotation=0, labelsize=11, colors='#555555')

        # 6. Data Table with colored columns
        cell_text = [[f"{w:,}" for w in weights], boxes]
        table = plt.table(cellText=cell_text,
                          rowLabels=['WEIGHT (Kg)', 'BOXES'],
                          rowColours=[line_color, '#808080'],
                          colLabels=categories,
                          colColours=bar_colors,
                          cellLoc='center',
                          loc='bottom',
                          bbox=[0, -0.3, 1, 0.2])
        
        # Style table cells
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('w')
            row_idx, col_idx = key
            if row_idx > -1 and col_idx > -1:  # Data cells
                cell.set_facecolor(bar_colors[col_idx])
                cell.set_text_props(color='white', weight='bold')
            elif col_idx == -1 and row_idx > -1:  # Row headers
                cell.set_text_props(color='white')
            elif row_idx == -1 and col_idx > -1:  # Column headers
                cell.set_text_props(weight='bold', color='white')

        # 7. Legend with grade categories
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2 = [Patch(facecolor=color, label=label) for color, label in 
                    [('#d90429', 'AAA Grade'), ('#f97316', 'AA Grade'), 
                     ('#22c55e', 'GP Grade'), ('#f472b6', 'Mix/Pear')]]
        labels2 = [h.get_label() for h in handles2]

        fig.legend(handles1 + handles2, labels1 + labels2,
                  title="Legend",
                  loc='upper right',
                  bbox_to_anchor=(0.87, 0.88),
                  frameon=True,
                  framealpha=0.95,
                  facecolor='white',
                  edgecolor='#dddddd')
        
        # 8. Final touches
        ax1.set_xticks([])
        ax1.grid(False)
        ax2.grid(False)

        # Add watermark
        fig.text(0.5, 0.5, 'FASCORP', 
                 fontsize=100, color='grey', 
                 ha='center', va='center', alpha=0.1, rotation=30)
        
        # 9. Save to bytes buffer and return as raw bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig) # Close the figure to free up memory
        buf.seek(0)

    # 9. Return the image as a response
        return send_file(buf, mimetype='image/png')
        
        # Return as raw bytes (UInt8List)
        

def health_check():
    return jsonify({"status": "healthy", "version": "1.0.0"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.logger.info(f"🚀 Starting server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)


