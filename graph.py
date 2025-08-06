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
