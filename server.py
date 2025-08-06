from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import openai
import json
import difflib
import os
import io
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import matplotlib
matplotlib.use('Agg')
import numpy as np
import io
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for web servers
import seaborn as sns
import matplotlib.patheffects as path_effects

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

# Define the chart generation route
@app.route("/chart", methods=["POST"])
def generate_chart():
    """
    Generates an attractive, modern styled bar-line chart from POST request data 
    and returns it as a PNG image.
    Expects a JSON payload with 'categories', 'quality', 'weights', and 'boxes'.
    """
    # 1. Get data from the POST request
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

    # 2. Setup Modern Style from your function
    sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.color': '0.8'})
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # Create figure with modern aesthetics and a golden background
    fig, ax1 = plt.subplots(figsize=(14, 8.5), facecolor='gold') # Set outer background to gold
    fig.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.1)
    
    # Set a subtle background color for the plot area (graph bg)
    ax1.set_facecolor('#f5f5f5')

    # 3. Enhanced Line Plot (Weights)
    ax1.plot(categories, weights, 
             color='#007f5f',       # A slightly darker green for better contrast
             marker='D',            # Diamond markers
             markersize=8,
             markeredgecolor='white',
             markeredgewidth=1.5,
             linewidth=3, 
             label="Weight (Kg)",
             zorder=3,
             path_effects=[path_effects.withStroke(linewidth=5, foreground='white')])
    
    # Add data labels for the line plot
    for x, y in zip(categories, weights):
        ax1.annotate(f"{y:,} kg", (x, y),
                     textcoords="offset points", xytext=(0,15), ha='center',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#007f5f', edgecolor='white', alpha=0.9))

    ax1.set_ylabel("Weight (Kg)", color='#007f5f', fontsize=13, labelpad=15)
    ax1.tick_params(axis='y', colors='#007f5f', labelsize=11)
    ax1.set_ylim(0, max(weights) * 1.25)
    
    # 4. Modern Bar Plot (Boxes) on a shared X-axis with custom colors
    ax2 = ax1.twinx()
    
    # Define the custom color sequence for the bars
    # NOTE: This assumes there will be 6 categories to match the color list.
    bar_colors = ['#d90429', '#d90429', '#f97316', '#f97316', '#22c55e', '#f472b6']
    if len(categories) != len(bar_colors):
        # Fallback to a single color if the number of categories is not 6
        bar_colors = ['#3a7bd5'] * len(categories)

    bars = ax2.bar(categories, boxes, 
                   color=bar_colors,
                   alpha=0.9, 
                   width=0.5,
                   edgecolor='white',
                   linewidth=1.5,
                   label="No. of Boxes",
                   zorder=2)

    # Add value labels for the bar plot ("No of boxes")
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), # text offset
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=10,
                     fontweight='bold',
                     color=bar.get_facecolor(), # Label color matches bar color
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])


    ax2.set_ylabel("No. of Boxes", color='#555555', fontsize=13, labelpad=15)
    ax2.tick_params(axis='y', colors='#555555', labelsize=11)
    ax2.set_ylim(0, max(boxes) * 1.4)
    
    # 5. Modern Title and X-axis labels
    fig.suptitle(f"PROCUREMENT ANALYTICS", fontsize=18, y=0.98, color='#333333')
    plt.title("Weight vs Box Count by Consignment", fontsize=12, pad=20, color='#777777')
    ax1.tick_params(axis='x', rotation=45, labelsize=11, colors='#555555')

    # 6. Enhanced Data Table
    cell_text = [[f"{w:,}" for w in weights], boxes]
    row_labels = ['WEIGHT (Kg)', 'BOXES']
    # Use neutral colors for table rows since bars have many colors
    row_colours = ['#007f5f', '#808080'] 
    col_colours = ['#f5f5f5'] * len(categories)

    table = plt.table(cellText=cell_text,
                      rowLabels=row_labels,
                      rowColours=row_colours,
                      colLabels=categories,
                      colColours=col_colours,
                      cellLoc='center',
                      loc='bottom',
                      bbox=[0, -0.35, 1, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('w')
        cell.set_text_props(color='white' if key[1] == -1 else '#333333')
        if key[0] == -1: # Header row
             cell.set_text_props(weight='bold', color='#555555')

    # 7. Modern Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Create custom legend handles for the multi-colored bars
    from matplotlib.patches import Patch
    handles2 = [Patch(facecolor=color, label=label) for color, label in 
                [('#d90429', 'Group 1'), ('#f97316', 'Group 2'), ('#22c55e', 'Group 3'), ('#f472b6', 'Group 4')]]
    labels2 = [h.get_label() for h in handles2]


    legend = fig.legend(handles1 + handles2, labels1 + labels2,
                        title="Legend",
                        loc='upper right',
                        bbox_to_anchor=(0.87, 0.88),
                        frameon=True,
                        framealpha=0.95,
                        facecolor='white',
                        edgecolor='#dddddd')
    
    # 8. Final Polish & Save to Buffer
    ax1.set_xticks([]) # Hide original x-axis ticks
    ax1.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
    ax2.grid(False)

    # Add watermark
    fig.text(0.5, 0.5, 'FASCORP', 
             fontsize=100, color='grey', 
             ha='center', va='center', alpha=0.1, rotation=30, zorder=0)
    
    # Save chart to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig) # Close the figure to free up memory
    buf.seek(0)

    # 9. Return the image as a response
    return send_file(buf, mimetype='image/png')





# Changed method from GET to POST to allow for a request body
@app.route('/generate_chart', methods=['POST'])
def generate_chart_module():
    """
    Generates the styled chart from data provided in a POST request.
    """
    # 1. Get data from the POST request body
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be JSON."}), 400

        categories = data['categories']
        weights = data['weights']
        per_kg_prices = data['per_kg_prices']
        landing_costs = data['landing_costs']
    except (KeyError, TypeError):
        return jsonify({"error": "Request body must contain 'categories', 'weights', 'per_kg_prices', and 'landing_costs'."}), 400

    # The rest of the chart generation code remains the same
    x_values = np.arange(len(categories))

    # 2. Create the chart
    fig, ax1 = plt.subplots(figsize=(17, 8))
    ax2 = ax1.twinx()

    fig.set_facecolor('gold')
    ax1.set_facecolor('#F0F8FF')

    ax1.plot(x_values, weights, color='blue', marker='o', linewidth=3)
    ax2.plot(x_values, per_kg_prices, color='red', marker='o', linewidth=3)
    ax2.plot(x_values, landing_costs, color='green', marker='o', linewidth=3)

    ax1.set_ylabel('Fruit Category Wise Weight in Kg', color='black', fontsize=12, weight='bold')
    max_weight = max(weights) if weights else 0
    max_price = max(max(per_kg_prices) if per_kg_prices else 0, 
                max(landing_costs) if landing_costs else 0)
    weight_axis_limit = max_weight * 1.2 if max_weight > 0 else 2500
    price_axis_limit = max_price * 1.2 if max_price > 0 else 100
    ax1.set_ylim(0, weight_axis_limit)
    ax2.set_ylim(0, price_axis_limit)

    plt.title(
        'Consignment Wise Procurement Analytics...\nfor Buyers to pick up or leave the Consignment Bidding',
        fontsize=14, weight='bold'
    )

    ax1.set_xticks([])
    cell_text = [weights, per_kg_prices, landing_costs]
    row_labels = ['Weight in Kg', 'Per Kg Price', 'Landing Cost']
    row_colors = ['blue', 'red', 'green']

    the_table = plt.table(
        cellText=cell_text, colLabels=categories, rowLabels=row_labels,
        rowColours=row_colors, loc='bottom', cellLoc='center'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 2.5)

    cells = the_table.get_celld()
    for j in range(len(categories)):
        cell = cells[(0, j)]
        cell.set_height(cell.get_height() * 1.2)
        cell.set_text_props(va='center')

    for i in range(1, len(row_labels) + 1):
        for j in range(len(categories)):
            cells[(i, j)].set_height(cells[(i, j)].get_height() * 1.2)

    plt.subplots_adjust(left=0.1, bottom=0.3)
    ax1.grid(True, color='c', linestyle='--')
    for spine in ax1.spines.values():
        spine.set_edgecolor('c')
        spine.set_linewidth(2)

    # 3. Save chart to an in-memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # 4. Convert to uint8 list and return
    image_uint8_list = list(buf.getvalue())
    return jsonify({"image": image_uint8_list})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
