import io
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend to prevent GUI display

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch

def generate_chart(categories, weights, boxes, quality='Standard', output_filename='chart.png'):
    """
    Generates and saves a modern styled bar-line chart.

    Args:
        categories (list): A list of strings for the x-axis categories.
        weights (list): A list of numbers (int or float) for the line plot values.
        boxes (list): A list of numbers (int or float) for the bar chart values.
        quality (str, optional): A string to be included in the chart title. Defaults to 'Standard'.
        output_filename (str, optional): The name of the file to save the chart as. Defaults to 'chart.png'.
    """
    # 1. Validate data length
    if not (len(categories) == len(weights) == len(boxes)):
        print("Error: All lists (categories, weights, boxes) must be of the same length.")
        return

    # 2. Setup modern style for the plot
    sns.set_style("whitegrid", {'grid.linestyle': ':', 'grid.color': '0.8'})
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    
    # Create figure and primary axis
    # --- CHANGE: Set figure background to gold ---
    fig, ax1 = plt.subplots(figsize=(14, 8.5), facecolor="gold")
    fig.subplots_adjust(left=0.08, right=0.88, top=0.9, bottom=0.25) # Adjust bottom for table
    
    # 3. Line Plot (for Weights)
    line_color = "#22258d"
    # Plot the line with markers and effects for better visibility
    ax1.plot(categories, weights, 
             color=line_color,
             marker='D',
             markersize=8,
             markeredgecolor='white',
             markeredgewidth=1.5,
             linewidth=3, 
             label="Weight (Kg)",
             path_effects=[path_effects.withStroke(linewidth=5, foreground='white')])

    # Add data labels above the line plot points
    for x, y in zip(categories, weights):
        ax1.annotate(f"{y:,} kg", (x, y),
                     textcoords="offset points", xytext=(0,15), ha='center',
                     fontsize=10, fontweight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=line_color, edgecolor='white', alpha=0.9))

    ax1.set_ylabel("Weight (Kg)", color=line_color, fontsize=13, labelpad=15)
    ax1.tick_params(axis='y', colors=line_color, labelsize=11)
    ax1.set_ylim(0, max(weights) * 1.25)
    
    # 4. Bar Plot (for Boxes) on a secondary axis
    ax2 = ax1.twinx() # Create a secondary y-axis sharing the same x-axis
   

    
    # --- CHANGE: Make ax2 background transparent to show ax1's white grid ---


    # Define bar colors. Use a default if lengths don't match.
    bar_colors = ['#d90429', '#d90429', '#f97316', '#f97316', '#22c55e', '#f472b6']
    if len(categories) != len(bar_colors):
        bar_colors = ['#3a7bd5'] * len(categories)

    # Plot the bars
    bars = ax2.bar(categories, boxes, 
                   color=bar_colors,
                   alpha=0.9, 
                   width=0.5,
                   edgecolor='white',
                   linewidth=1.5,
                   label="No. of Boxes")

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), # 5 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=10,
                     fontweight='bold',
                     color=bar.get_facecolor(),
                     path_effects=[path_effects.withStroke(linewidth=2, foreground='white')])

    ax2.set_ylabel("No. of Boxes", color="#22258d", fontsize=13, labelpad=15)
    ax2.tick_params(axis='y', colors='#22258d', labelsize=11)
    ax2.set_ylim(0, max(boxes) * 1.4)
    
    # 5. Titles and X-axis labels
    # --- CHANGE: Set title color to white and bold ---
    fig.suptitle(f"PROCUREMENT ANALYTICS - {quality} Quality", fontsize=18, y=0.98, color="#000000", fontweight='bold')
    plt.title("Weight vs Box Count by Consignment", fontsize=12, pad=20, color='#000000', fontweight='bold')
    ax1.tick_params(axis='x', rotation=0, labelsize=11, colors='#555555')

    # 6. Data Table at the bottom of the chart
    cell_text = [[f"{w:,}" for w in weights], boxes]
    row_labels = ['WEIGHT (Kg)', 'BOXES']
    row_colors = [line_color, '#808080']
    col_colors = bar_colors

    table = plt.table(cellText=cell_text,
                      rowLabels=row_labels,
                      rowColours=row_colors,
                      colLabels=categories,
                      colColours=col_colors, # Use the bar colors for column backgrounds
                      cellLoc='center',
                      loc='bottom',
                      bbox=[0, -0.3, 1, 0.2]) # Position the table
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # --- CHANGE: Color full columns and adjust text for contrast ---
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('w')
        row_idx, col_idx = key

        # This condition handles the data cells (not headers)
        if row_idx > -1 and col_idx > -1:
            cell.set_facecolor(bar_colors[col_idx])
            cell.set_text_props(color='white', weight='bold')
        # This condition handles the row headers (e.g., 'WEIGHT', 'BOXES')
        elif col_idx == -1 and row_idx > -1:
            cell.set_text_props(color='white')
        # This condition handles the column headers (e.g., 'CON-001')
        elif row_idx == -1 and col_idx > -1:
            cell.set_text_props(weight='bold', color='white')

    # 7. Custom Legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Create custom patches for the bar chart categories
    handles2 = [Patch(facecolor=color, label=label) for color, label in 
                [('#d90429', 'AAA Grade'), ('#f97316', 'AA Grade'), ('#22c55e', 'GP Grade'), ('#f472b6', 'Mix/Pear')]]
    labels2 = [h.get_label() for h in handles2]

    # Combine legends and place them on the figure
    fig.legend(handles1 + handles2, labels1 + labels2,
               title="Legend",
               loc='upper right',
               bbox_to_anchor=(0.87, 0.88),
               frameon=True,
               framealpha=0.95,
               facecolor='white',
               edgecolor='#dddddd')
    
    # 8. Final Touches
    ax1.set_xticks([]) # Hide original x-axis ticks, as they are in the table
    ax1.grid(False)
    ax2.grid(False) # Turn off grid for the secondary axis

    # Add a watermark
    fig.text(0.5, 0.5, 'FASCORP', 
             fontsize=100, color='grey', 
             ha='center', va='center', alpha=0.1, rotation=30)
    
    # 9. Save the figure to a file
    try:
        plt.savefig(output_filename, format='png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"Chart successfully saved to {output_filename}")
    except Exception as e:
        print(f"Error saving chart: {e}")
    finally:
        plt.close(fig) # Close the figure to free up memory


if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It serves as a test case for the module.
    
    # --- Sample Data ---
    sample_categories = ['CON-001', 'CON-002', 'CON-003', 'CON-004', 'CON-005', 'CON-006']
    sample_weights = [1200, 1550, 950, 1800, 1300, 1650]
    sample_boxes = [80, 100, 65, 120, 90, 110]
    sample_quality = 'Premium'
    
    # --- Generate the chart using the function ---
    generate_chart(
        categories=sample_categories,
        weights=sample_weights,
        boxes=sample_boxes,
        quality=sample_quality,
        output_filename="procurement_chart.png" # Specify a custom output file name
    )
