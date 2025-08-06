import requests
import matplotlib.pyplot as plt
from io import BytesIO

# Sample request data
chart_data = {
    "categories": ["CON-001", "CON-002", "CON-003", "CON-004", "CON-005", "CON-006"],
    "weights": [1250, 1800, 950, 2100, 1500, 1750],
    "boxes": [85, 120, 65, 140, 95, 115],
    "quality": "Premium"
}

# Make the POST request
try:
    response = requests.post(
        "http://localhost:5000/chart",  # Replace with your server URL
        json=chart_data,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        # Option 1: Save to file
        with open("procurement_chart.png", "wb") as f:
            f.write(response.content)
        print("Chart saved as procurement_chart.png")

        # Option 2: Display directly using matplotlib
        img = plt.imread(BytesIO(response.content))
        plt.figure(figsize=(12, 7))
        plt.imshow(img)
        plt.axis('off')
        plt.title("Generated Procurement Chart")
        plt.show()

    else:
        print(f"Error {response.status_code}: {response.json().get('error', 'Unknown error')}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {str(e)}")
except Exception as e:
    print(f"Error processing response: {str(e)}")